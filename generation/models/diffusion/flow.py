import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# 假设这些是你原有的 import
from generation.utils.utils import *
from generation.models.unet.blocks import *
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
from heltonx.utils.register import MODELS


@MODELS.register
class Flow(nn.Module):

    def __init__(self,
                 denoise_model,
                 img_size,
                 batch_size, 
                 load_ckpt=None, 
                 loss_type='huber',
                 sampling_steps=50 # 采样步数 
                 ): 
        super(Flow, self).__init__()
        self.loss_type = loss_type
        self.denoise_model = denoise_model
        self.channel = self.denoise_model.input_dim 
        self.img_size = img_size
        self.bs = batch_size
        self.sampling_steps = sampling_steps

        # Flow Matching 非常简洁，不再需要 get_init_params 生成 alphas 和 betas

        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)



    def compute_loss(self, x_start, loss_type="l1"):
        """
        Flow Matching 的前向与 Loss 计算
        设定: t=0 是纯噪声，t=1 是真实图像 (x_start)
        """
        bs = x_start.shape[0]
        device = x_start.device
        
        # 1. 采样纯高斯噪声 x_0
        noise = torch.randn_like(x_start)
        # 2. 均匀采样连续的时间步 t ~ U[0, 1]
        t = torch.rand((bs,), device=device)
        # 扩展 t 的维度以匹配图像形状 [B, 1, 1, 1]
        t_expand = t.view(-1, 1, 1, 1)
        # 3. 线性插值构建 x_t: x_t = t * x_start + (1 - t) * noise
        x_t = t_expand * x_start + (1.0 - t_expand) * noise
        # 4. 计算目标速度场 (Target Vector Field): v = x_start - noise
        target_v = x_start - noise
        # 5. 模型预测速度场
        # 如果 denoise_model 内部的 TimeEmbedding 依然是按照 0-1000 设计的，
        # 你需要在这里将 0~1 的 t 放大到 0~1000 传给模型： t * 1000.0
        predicted_v = self.denoise_model(x_t, t * 1000.0) 
        
        if loss_type == 'l1':
            loss = F.l1_loss(predicted_v, target_v)
        elif loss_type == 'l2':
            loss = F.mse_loss(predicted_v, target_v)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_v, target_v)
        else:
            raise NotImplementedError()

        return loss



    @torch.no_grad()
    def euler_sample_loop(self, shape, steps=None):
        """
        使用 Euler (欧拉) 方法进行 ODE 求解采样
        """
        device = next(self.denoise_model.parameters()).device
        steps = steps or self.sampling_steps
        b = shape[0]
        
        # 从 t=0 (纯噪声) 开始
        img = torch.randn(shape, device=device)
        imgs = []
        # 计算时间步长 dt
        dt = 1.0 / steps
        # 从 t=0 逐步推演到 t=1
        for i in tqdm(range(steps), desc='Flow Matching Euler sampling', total=steps):
            # 当前的连续时间 t
            t_val = i / steps
            t = torch.full((b,), t_val, device=device)
            # 模型预测当前的速度场 (同样注意 t * 1000.0 的量级适配)
            v_pred = self.denoise_model(img, t * 1000.0)
            # Euler 步进: x_{t+dt} = x_t + v * dt
            img = img + v_pred * dt
            imgs.append(img)
            
        return imgs



    @torch.no_grad()
    def sample(self, bs=None):
        bs = bs if bs else self.bs
        shape = (bs, self.channel, self.img_size[0], self.img_size[1])
        
        # 直接使用 Euler ODE solver 采样，相当于 Diffusion 中的 DDIM 确定性采样
        denoise_img_series = self.euler_sample_loop(shape=shape)
        return denoise_img_series



    def forward(self, batch_data=None, return_loss=True, bs=None):
        """前向 + 计算损失"""
        if return_loss:
            y = batch_data[0]
            # 不再需要外部采样 t，compute_loss 内部会处理连续的 t
            loss = self.compute_loss(x_start=y, loss_type=self.loss_type)
            losses = dict(gen_loss=loss)
            return losses
        else:
            # 返回采样的最后一步图像
            return self.sample(bs=bs)[-1]