# coding=utf-8
import os
import json
import torch
import random
import numpy as np

from pretrain.utils.metrics import *
from pretrain.datasets.preprocess import Transforms
# 多卡并行训练:
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from heltonx.utils.register import EVALPIPELINES


@EVALPIPELINES.register
class GenerationEvalPipeline():
    '''一个epoch的评估(基于验证集)
    '''
    def __call__(self, runner, model=None):
        # 直接从传入的类中获取参数(避免每个任务的特殊化):
        if model==None:
            model = runner.model
        epoch = runner.cur_epoch
        log_dir = runner.log_dir

        # 图像均值 标准差
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([[0.229, 0.224, 0.225]]) 

        model.eval()
        # 图像生成
        samples = model(bs=64, return_loss=False)

        # 可视化
        generate_images = samples
        B, C, H, W = generate_images.shape
        print(generate_images.shape)
        fig, axes = plt.subplots(8, 8, figsize=(10, 10))  # Create an 8x8 grid of subplots
        for i, ax in enumerate(axes.flat):
            gen_img_norm = generate_images[i].reshape(C, H, W).transpose((1,2,0))
            # figtest = reverse_transform(torch.from_numpy(generate_image))
            gen_img = gen_img_norm * std + mean
            ax.imshow(gen_img) 
            ax.axis("off")  

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"epoch_{epoch}.png"), dpi=300)

        # TODO:
        # 评估结果以字典形式返回(统一格式, key的前缀一定有'val_')
        evaluations = dict(
            ssim=epoch
        )
        # 后续保存best_ckpt以val_flag_metric为参考
        flag_metric_name = "ssim"
        return evaluations, flag_metric_name
    






@EVALPIPELINES.register
class ConditionGenerationEvalPipeline():
    '''一个epoch的评估(基于验证集)
    '''
    def __call__(self, runner, model=None):
        bs = 64
        # 直接从传入的类中获取参数(避免每个任务的特殊化):
        if model==None:
            model = runner.model
        epoch = runner.cur_epoch
        log_dir = runner.log_dir

        '''永远读取第2条数据的caption'''
        prompts = runner.train_dataset.samples[1]['conversations'][1]['content']
        tokenizer = runner.train_dataset.tokenizer
        max_len = runner.train_dataset.max_length
        # 进行tokenize + 截断
        input_ids = tokenizer(prompts).input_ids
        input_ids = input_ids[:max_len]
        # padding补齐
        input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
        prompts_tokens = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).repeat(bs, 1)

        # 图像均值 标准差
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([[0.229, 0.224, 0.225]]) 

        model.eval()
        # 图像生成
        samples = model(bs=bs, c_tokens=prompts_tokens, return_loss=False)

        # 可视化
        generate_images = samples
        B, C, H, W = generate_images.shape
        print(generate_images.shape)
        fig, axes = plt.subplots(8, 8, figsize=(10, 10))  # Create an 8x8 grid of subplots
        for i, ax in enumerate(axes.flat):
            gen_img_norm = generate_images[i].reshape(C, H, W).transpose((1,2,0))
            # figtest = reverse_transform(torch.from_numpy(generate_image))
            gen_img = gen_img_norm * std + mean
            ax.imshow(gen_img) 
            ax.axis("off")  

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"epoch_{epoch}.png"), dpi=300)

        # TODO:
        # 评估结果以字典形式返回(统一格式, key的前缀一定有'val_')
        evaluations = dict(
            ssim=epoch
        )
        # 后续保存best_ckpt以val_flag_metric为参考
        flag_metric_name = "ssim"
        return evaluations, flag_metric_name