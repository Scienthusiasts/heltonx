import torch
from torch import nn
import torch.nn.functional as F

from generation.utils.utils import *
from generation.models.blocks import *
from generation.utils.var_schedule import *
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
from heltonx.utils.register import MODELS



@MODELS.register
class DDPM(nn.Module):
    def __init__(self,
                 denoise_model,
                 img_size,
                 batch_size, 
                 load_ckpt=None, 
                 schedule_name="linear_beta_schedule",
                 loss_type='huber',
                 timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02):
        super(DDPM, self).__init__()
        self.loss_type = loss_type
        self.denoise_model = denoise_model
        self.img_size = img_size
        self.bs = batch_size
        self.timesteps = timesteps

        # variance schedule
        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)
        self.betas = variance_schedule_func(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # optionally load pretrained weights
        if load_ckpt:
            self = load_state_dict_with_prefix(self, load_ckpt)

    # --------------------------------------------------------------------------------
    def q_sample(self, x_start, t, noise=None):
        """前向扩散 q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # --------------------------------------------------------------------------------
    def compute_loss(self, x_start, t, noise=None, loss_type="l1"):
        """训练阶段损失函数"""
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # === add self_cond ===
        x_self_cond = None
        if getattr(self.denoise_model, "self_condition", False):
            # 50% 概率启用 self-conditioning（论文中的做法）
            if torch.rand(1).item() < 0.5:
                x_self_cond = torch.zeros_like(x_start)
            else:
                with torch.no_grad():
                    # 先估计一次噪声作为条件输入
                    x_self_cond = self.denoise_model(x_noisy, t).detach()

        # === add self_cond ===
        # 传入 x_self_cond，如果模型未启用则忽略
        predicted_noise = self.denoise_model(x_noisy, t, x_self_cond)

        # loss计算
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x, t, t_index, x_self_cond=None):
        """反向采样 p(x_{t-1} | x_t)"""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # === add self_cond ===
        predicted_noise = self.denoise_model(x, t, x_self_cond)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def p_sample_loop(self, shape):
        """完整采样过程"""
        device = next(self.denoise_model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []

        # === add self_cond ===
        x_self_cond = torch.zeros_like(img) if getattr(self.denoise_model, "self_condition", False) else None

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i, x_self_cond)

            # 如果模型启用了 self-conditioning，则将当前预测的噪声作为下一步输入
            if getattr(self.denoise_model, "self_condition", False):
                x_self_cond = self.denoise_model(img, torch.full((b,), i, device=device, dtype=torch.long)).detach()

            imgs.append(img.cpu().numpy())
        return imgs

    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, bs=None, channels=3):
        """外部调用采样"""
        bs = bs if bs else self.bs
        denoise_img_series = self.p_sample_loop(shape=(bs, channels, self.img_size[0], self.img_size[1]))
        return denoise_img_series

    # --------------------------------------------------------------------------------
    def forward(self, batch_data=None, return_loss=True, bs=None):
        """前向推理 + 计算损失"""
        if return_loss:
            y = batch_data[0]
            bs = y.shape[0]
            t = torch.randint(0, self.timesteps, (bs,), device=y.device).long()
            loss = self.compute_loss(x_start=y, t=t, loss_type=self.loss_type)
            return dict(gen_loss=loss)
        else:
            return self.sample(bs=bs)
