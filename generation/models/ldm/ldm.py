import torch
from torch import nn
import torch.nn.functional as F
from generation.utils.utils import *
from generation.models.unet.blocks import *
from generation.utils.var_schedule import *
from generation.models.diffusion import DDPM, Flow
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
from heltonx.utils.register import MODELS



@MODELS.register
class LDM(DDPM):

    def __init__(self,
                 vae,
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
        # 潜空间扩散需要VAE将潜空间特征复原到图像空间
        self.vae = vae.eval()
        # latent space 的维度和 vae的下采样率
        self.channel = self.vae.latent_dim
        self.down_scale = self.vae.down_scale
        # 扩散在潜空间进行, 因此图像大小是潜空间特征的大小
        img_size[0] //= self.vae.down_scale
        img_size[1] //= self.vae.down_scale

        self.loss_type = loss_type
        self.denoise_model = denoise_model
        self.img_size = img_size
        self.bs = batch_size

        # 生成训练或推理会用到的参数
        self.get_init_params(schedule_name, timesteps, beta_start, beta_end)
        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)



    def forward(self, batch_data=None, return_loss=True, bs=None):
        """前向+计算损失 (LDM只修改这里)
        """
        if return_loss:
            y = batch_data[0]
            bs = y.shape[0]
            with torch.no_grad():
                # 调用 VAE 的 forward 提取潜特征 
                z_0 = self.vae(x=y, return_loss=True)
            # 随机生成加噪时间步，模拟不同加噪程度的图像
            t = torch.randint(0, self.timesteps, (bs,), device=y.device).long()
            # 计算损失时也是在潜空间计算
            loss = self.compute_loss(x_start=z_0, t=t, loss_type=self.loss_type)
            # 字典形式
            losses = dict(
                gen_loss = loss
            )
            return losses
        else:
            # 在潜空间中进行采样，获取去噪完成的特征序列
            latent_series = self.sample(bs=bs)
            # 提取最后一步去噪完成的潜特征, [-1]是只取最后一个time_step 
            final_latent = latent_series[-1]  
            # 潜空间 -> 图像空间
            gen_img = self.vae.decode(final_latent).float().cpu().numpy()
            return gen_img













@MODELS.register
class LFM(Flow):

    def __init__(self,
                 vae,
                 denoise_model,
                 img_size,
                 batch_size,
                 sampling_steps,
                 load_ckpt=None, 
                 loss_type='huber',
                ):
        super(Flow, self).__init__()
        # 潜空间扩散需要VAE将潜空间特征复原到图像空间
        self.vae = vae.eval()
        # latent space 的维度和 vae的下采样率
        self.channel = self.vae.latent_dim
        self.down_scale = self.vae.down_scale
        # 扩散在潜空间进行, 因此图像大小是潜空间特征的大小
        img_size[0] //= self.vae.down_scale
        img_size[1] //= self.vae.down_scale

        self.loss_type = loss_type
        self.denoise_model = denoise_model
        self.img_size = img_size
        self.bs = batch_size
        self.sampling_steps = sampling_steps

        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)


    def forward(self, batch_data=None, return_loss=True, bs=None):
        """前向+计算损失 (LDM只修改这里)
        """
        if return_loss:
            y = batch_data[0]
            bs = y.shape[0]
            with torch.no_grad():
                # 调用 VAE 的 forward 提取潜特征 
                z_0 = self.vae(x=y, return_loss=True)
            # 不再需要外部采样 t，compute_loss 内部会处理连续的 t
            loss = self.compute_loss(x_start=z_0, loss_type=self.loss_type)
            # 字典形式
            losses = dict(
                gen_loss = loss
            )
            return losses
        else:
            # 在潜空间中进行采样，获取去噪完成的特征序列
            latent_series = self.sample(bs=bs)
            # 提取最后一步去噪完成的潜特征, [-1]是只取最后一个time_step 
            final_latent = latent_series[-1]  
            # 潜空间 -> 图像空间
            gen_img = self.vae.decode(final_latent).float().cpu().numpy()
            return gen_img





