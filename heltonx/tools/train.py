# coding=utf-8
"""
HeltonX 训练框架 - PyTorch 原生 DDP 实现
提供完整的训练、验证和推理流程支持
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

from heltonx.optimization import *
from heltonx.utils.utils import seed_everything, worker_init_fn, get_args, dynamic_import_class, set_dataloader_epoch, to_device
from heltonx.utils.ckpts_utils import train_resume
from heltonx.utils.log_utils import *
from heltonx.utils.hooks import NecessaryHook
from heltonx.utils.register import MODELS, DATASETS, OPTIMIZERS, SCHEDULERS, EVALPIPELINES



class Trainer:
    """整合训练/验证/推理时的抽象流程"""

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
            mode (str): 训练模式，可选值：'train'（单卡训练）、'train_ddp'（多卡DDP训练）
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
        self.seed = seed
        self.grad_accumulate = grad_accumulate
        self.grad_clip = grad_clip
        # 设置全局种子
        seed_everything(self.seed)

        '''确定 CPU/单卡/DDP 训练策略'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.mode=='train_ddp':
            self.local_rank = int(os.environ["LOCAL_RANK"]) 
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group('nccl')

        '''导入网络'''
        self.model = MODELS.build_from_cfg(model_cfgs)
        # self.model = torch.compile(self.model)
        if self.mode=='train_ddp':
            # 多卡时同步BN
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda(self.local_rank)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], find_unused_parameters=False)
        else:
            self.model.to(self.device)

        '''导入数据集'''
        self.train_dataset = DATASETS.build_from_cfg(dataset_cfgs["train_dataset_cfg"])
        self.valid_dataset = DATASETS.build_from_cfg(dataset_cfgs["valid_dataset_cfg"]) if dataset_cfgs["valid_dataset_cfg"] else None
        # DDP训练时需要sampler且shuffle=False
        train_sampler = None
        if self.mode == 'train_ddp':
            train_sampler = DistributedSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            batch_size=dataset_cfgs["train_bs"],
            num_workers=dataset_cfgs["num_workers"],
            shuffle=False if self.mode == 'train_ddp' else dataset_cfgs["train_shuffle"],
            collate_fn=self.train_dataset.dataset_collate,
            worker_init_fn=partial(worker_init_fn, seed=self.seed),
            pin_memory=True # CPU → GPU 数据拷贝速度加速
        )
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
        self.optimizer = OPTIMIZERS.build_from_cfg(optimizer_cfgs, params=self.model.parameters())
        # 学习率衰减策略(+warmup)
        scheduler_cfgs["base_schedulers_cfgs"]["step_size"] *= self.train_batch_num
        scheduler_cfgs["warmup_schedulers_cfgs"]["warmup_epochs"] *= self.train_batch_num
        base_scheduler = SCHEDULERS.build_from_cfg(scheduler_cfgs["base_schedulers_cfgs"], optimizer=self.optimizer)
        self.scheduler = SCHEDULERS.build_from_cfg(
            scheduler_cfgs["warmup_schedulers_cfgs"], 
            base_scheduler=base_scheduler, 
            optimizer=self.optimizer,
            batch_num=self.train_batch_num
            )

        '''日志模块'''
        self.runner_logger = None
        self.runner_logger = RunnerLogger(self.mode, self.log_dir, log_interval, eval_interval, self.train_batch_num)
        self.log_dir = self.runner_logger.log_dir

        '''Hook 管理'''
        self._hooks = {}

        # resume
        if resume_path:
            self.start_epoch = train_resume(resume_path, self.model, self.optimizer, self.scheduler, self.runner_logger, self.train_batch_num)
         # 打印模型详细信息
        if self.mode == 'train' or (self.mode == 'train_ddp' and dist.get_rank() == 0):
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
        self.losses['total_loss'].backward()
        '''梯度更新'''
        # 启用梯度累加时迭代的步数达到累加的步数时才进行梯度更新, 这样等效于增大了bs
        # 注意：cur_step 从 0 开始，第一步 (cur_step+1) % grad_accumulate == 1 % grad_accumulate
        # 当 grad_accumulate=1 时每步都更新；否则每累计 grad_accumulate 步更新一次
        should_update = (self.grad_accumulate is None or 
                         self.grad_accumulate == 1 or 
                         (self.cur_step + 1) % self.grad_accumulate == 0 or
                         self.cur_step + 1 == self.train_batch_num)  # 最后一个 batch 必须更新
        if should_update:
            # 梯度裁剪 LLM 训练中使用, 保证训练的稳定性
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
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
        set_dataloader_epoch(self.train_dataloader, self.cur_epoch, self.seed)
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
            '''一个epoch的训练'''
            self.fit_epoch()

        self.call_hooks("after_fit", runner=self)
