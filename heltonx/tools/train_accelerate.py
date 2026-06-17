# coding=utf-8
"""
HeltonX 训练框架 - Accelerate 加速实现
提供基于 Hugging Face Accelerate 的分布式训练支持
"""
import os
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from functools import partial
from torch.utils.data import DataLoader
from accelerate import Accelerator

from heltonx.optimization import *
from heltonx.utils.utils import seed_everything, accelerate_worker_init_fn, worker_init_fn, get_args, dynamic_import_class, to_device
from heltonx.utils.ckpts_utils import train_resume
from heltonx.utils.log_utils import *
from heltonx.utils.hooks_accelerate import NecessaryHook
from heltonx.utils.register import MODELS, DATASETS, OPTIMIZERS, SCHEDULERS, EVALPIPELINES



class Trainer:
    """训练器类，整合训练/验证/推理时的抽象流程（基于 Accelerate 实现）"""

    def __init__(
        self,
        mode,
        epoch,
        seed,
        log_dir,
        log_interval,
        eval_interval,
        resume_path,
        model_cfgs,
        dataset_cfgs,
        optimizer_cfgs,
        scheduler_cfgs,
        grad_accumulate=None,
        grad_clip=None
    ):
        """初始化训练器模块

        Args:
            mode (str): 训练模式，支持 'train', 'train_ddp' 等
            epoch (int): 训练轮数
            seed (int): 全局随机种子，用于复现实验结果
            log_dir (str): 日志文件保存目录
            log_interval (int): 日志打印间隔，每隔多少个iteration打印一次
            eval_interval (int): 评估间隔，每隔多少个epoch进行一次评估
            resume_path (str): 断点恢复路径，若为None则从头开始训练
            model_cfgs (dict): 网络模型配置参数，包含模型类型和初始化参数
            dataset_cfgs (dict): 数据集配置参数，包含数据集路径、batch_size等
            optimizer_cfgs (dict): 优化器配置参数
            scheduler_cfgs (dict): 学习率调度器配置参数
            grad_accumulate (int, optional): 梯度累积步数，用于增大等效batch_size，默认None
            grad_clip (float, optional): 梯度裁剪阈值，用于防止梯度爆炸，默认None
        """
        self.mode = mode
        self.log_dir = log_dir
        self.eval_interval = eval_interval
        self.epoch = epoch
        self.cur_epoch = 1
        self.start_epoch = 1
        self.cur_step = 0
        self.losses = None
        self.grad_accumulate = grad_accumulate
        self.grad_clip = grad_clip
        self.seed = seed
        # 设置全局种子
        seed_everything(self.seed)

        '''确定 CPU/单卡/DDP 训练策略'''
        # ✅ 初始化 accelerate
        self.accelerator = Accelerator()  
        self.device = self.accelerator.device  # 自动选择设备

        '''导入网络'''
        self.model = self.accelerator.prepare(MODELS.build_from_cfg(model_cfgs))

        '''导入数据集'''
        self.train_dataset = DATASETS.build_from_cfg(dataset_cfgs["train_dataset_cfg"])
        self.valid_dataset = DATASETS.build_from_cfg(dataset_cfgs["valid_dataset_cfg"]) if dataset_cfgs["valid_dataset_cfg"] else None
        # ✅ 不再手动创建 DistributedSampler，accelerator.prepare 会自动管理
        self.train_dataloader = self.accelerator.prepare(
            DataLoader(
            dataset=self.train_dataset,
            batch_size=dataset_cfgs["train_bs"],
            num_workers=dataset_cfgs["num_workers"],
            shuffle=True,
            collate_fn=self.train_dataset.dataset_collate,
            worker_init_fn=accelerate_worker_init_fn,
            pin_memory=True # CPU → GPU 数据拷贝速度加速
        ))
        self.valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=dataset_cfgs["valid_bs"],
            num_workers=dataset_cfgs["num_workers"],
            shuffle=dataset_cfgs["valid_shuffle"],
            collate_fn=self.valid_dataset.dataset_collate,
            pin_memory=True # CPU → GPU 数据拷贝速度加速
        ) if dataset_cfgs["valid_dataset_cfg"] else None
        # 一个epoch包含多少batch
        self.train_batch_num = len(self.train_dataloader)

        '''优化器'''
        # 支持 backbone 单独设置学习率倍率 (如 DETR 中 backbone_lr_mult=0.1)
        optimizer_cfgs_copy = optimizer_cfgs.copy()
        backbone_lr_mult = optimizer_cfgs_copy.pop('backbone_lr_mult', None)
        if backbone_lr_mult is not None:
            backbone_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if 'backbone' in name and param.requires_grad:
                    backbone_params.append(param)
                elif param.requires_grad:
                    other_params.append(param)
            params = [
                {'params': backbone_params, 'lr': optimizer_cfgs_copy['lr'] * backbone_lr_mult},
                {'params': other_params, 'lr': optimizer_cfgs_copy['lr']},
            ]
        else:
            params = self.model.parameters()
        self.optimizer = self.accelerator.prepare(OPTIMIZERS.build_from_cfg(optimizer_cfgs_copy, params=params))
        # 学习率衰减策略(+warmup)
        base_scheduler = SCHEDULERS.build_from_cfg(scheduler_cfgs["base_schedulers_cfgs"], optimizer=self.optimizer)
        self.scheduler = self.accelerator.prepare(
            SCHEDULERS.build_from_cfg(
                scheduler_cfgs["warmup_schedulers_cfgs"], 
                base_scheduler=base_scheduler, 
                optimizer=self.optimizer,
                batch_num=self.train_batch_num
        ))

        '''日志模块'''
        self.runner_logger = None
        if self.accelerator.is_main_process:
            self.runner_logger = RunnerLogger(self.mode, self.log_dir, log_interval, eval_interval, self.train_batch_num, self.epoch)
            self.log_dir = self.runner_logger.log_dir

        '''Hook 管理'''
        self._hooks = {}

        # resume
        if resume_path:
            self.start_epoch = train_resume(resume_path, self.model, self.optimizer, self.scheduler, self.runner_logger, self.train_batch_num)         # 打印模型详细信息
        if self.accelerator.is_main_process:
            self.runner_logger.log_model_info(self.model, self.optimizer)

    # Hook 机制 ==========
    def register_hook(self, event: str, func):
        """注册Hook回调函数

        Args:
            event (str): Hook事件名称，如 'before_batch', 'after_epoch' 等
            func (callable): Hook回调函数，接收 runner 实例作为参数
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(func)

    def call_hooks(self, event: str, *args, **kwargs):
        """调用指定事件的所有已注册Hook

        Args:
            event (str): Hook事件名称
            *args: 传递给Hook的位置参数
            **kwargs: 传递给Hook的关键字参数
        """
        for hook in self._hooks.get(event, []):
            hook(*args, **kwargs)



    def fit_batch(self, batch_datas):
        """执行单个batch的前向传播和反向传播

        Args:
            batch_datas: DataLoader传来的数据与标签列表

        Returns:
            dict: 包含各损失项的字典
        """
        self.call_hooks("before_batch", runner=self)

        # 确保 batch_datas 的所有数据已经在 self.device 上(batch_datas的组织形式是list)
        batch_datas = to_device(batch_datas, self.device, non_blocking=True)

        '''计算损失'''
        self.losses = self.model(batch_datas, return_loss=True)
        self.losses["total_loss"] = sum(
            v for v in self.losses.values()
            if torch.is_tensor(v) and v.requires_grad
        )
        if self.grad_accumulate:
            # 启用梯度累加时, 每个batch的loss需要等比例缩小(保证梯度也等比例缩小)
            self.losses["total_loss"] = self.losses["total_loss"] / self.grad_accumulate
        '''反向传播'''
        self.accelerator.backward(self.losses["total_loss"])
        '''梯度更新'''
        # 启用梯度累加时迭代的步数达到累加的步数时才进行梯度更新, 这样等效于增大了bs
        if self.grad_accumulate is None or (self.cur_step+1) % self.grad_accumulate==0:
            # 梯度裁剪 LLM 训练中使用, 保证训练的稳定性
            if self.grad_clip:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # 更新参数
            self.optimizer.step()
            # 将上一次迭代计算的梯度清零 
            self.optimizer.zero_grad(set_to_none=True)

        # 这一步只是为了日志打印时正常显示totalloss, 而不是缩放后的totalloss
        if self.grad_accumulate:
            self.losses["total_loss"] = self.losses["total_loss"] * self.grad_accumulate
            
        self.call_hooks("after_batch", runner=self)



    def fit_epoch(self):
        """执行单个epoch的训练流程

        包含：Hook调用、模型训练、数据迭代、参数更新、学习率调度
        """
        self.call_hooks("before_epoch", runner=self)
        self.model.train()
        # 固定每个epoch的随机性:
        # set_dataloader_epoch(self.train_dataloader, self.cur_epoch, self.seed)
        for step, batch_datas in enumerate(self.train_dataloader):
            self.cur_step = step
            '''一个batch的训练'''
            self.fit_batch(batch_datas)
            # 一个batch结束后更新学习率
            self.scheduler.step(epoch=self.cur_epoch, batch=step) 

        self.call_hooks("after_epoch", runner=self)



    def fit(self):
        """执行完整的训练流程（包含多个epoch的训练和验证）

        遍历所有epoch进行训练，每个epoch结束后执行Hook回调
        """
        self.call_hooks("before_fit", runner=self)

        for epoch in range(self.start_epoch, self.epoch+1):
            self.cur_epoch = epoch
            '''更新 progressive loss 权重 (epoch 级别衰减)(yolo26用到)'''
            model_unwrapped = self.accelerator.unwrap_model(self.model)
            if hasattr(model_unwrapped, 'update_progressive'):
                model_unwrapped.update_progressive(self.cur_epoch)
            '''一个epoch的训练'''
            self.fit_epoch()

        self.call_hooks("after_fit", runner=self)
        self.accelerator.end_training()







