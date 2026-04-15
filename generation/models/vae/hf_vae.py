import torch
import torch.nn as nn
from heltonx.utils.register import MODELS
from diffusers import AutoencoderKL


@MODELS.register
class HFVAE(nn.Module):
    """
    使用预训练的 VAE 作为图像特征编码器。
    将 RGB 图像压缩为保留了空间信息的高频 C 通道隐空间矩阵。
    """
    def __init__(self, weight_dir, latent_dim, down_scale):
        """初始化
            Args:
                weight_dir: VAE 的权重路径 (可以是本地文件夹路径，也可以是 huggingface 仓库名)
        """
        super().__init__()
        self.down_scale = down_scale
        self.latent_dim = latent_dim
        # 1. 核心：直接使用 diffusers 的 API，彻底免去手写网络结构的烦恼
        self.vae = AutoencoderKL.from_pretrained(weight_dir)
        # 2. 如果仅将它作为前置特征提取器（类似 CLIP），通常需要冻结权重
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

            
    def forward(self, x=None, return_loss=True, bs=None):
        """前向，调用 VAE 提取 C 通道隐空间特征
            Args:
                x: 输入图像 [B, 3, H, W], 像素值通常需要归一化到 [-1, 1]
                return_loss: 这里代表的是是否开启训练模型，而不是计算损失
                             return_loss=True:  返回latent表征, 此时x不为None
                             return_loss=False: 采样, 随机生成, 此时x为None
                             
            Returns:
                z: 潜在特征矩阵 [B, C, H/8, W/8]
        """
        if return_loss:
            # 1. 提取潜在分布 (Posterior)
            posterior = self.vae.encode(x).latent_dist
            # 2. 获取特征
            # 作为特征提取器供下游网络(如检测器)使用时，推荐使用 .mode() 取均值以获得确定性特征；
            # 如果是用于扩散模型加噪训练，则使用 .sample() 增加随机性。
            z = posterior.mode() 
            # 3. 极其重要：缩放因子 
            # 必须乘以官方配置好的缩放因子，让特征分布方差接近 1，否则下游网络极难收敛
            z = z * self.vae.config.scaling_factor
            return z
        else:
            with torch.no_grad():
                return self.sample(bs)
    

    def decode(self, z):
        """(附加功能) 如果需要将特征还原回图像查看"""
        # 解码前必须除以缩放因子还原回去
        z = z / self.vae.config.scaling_factor
        return self.vae.decode(z).sample


    def reconstruct(self, img):
        """前向（图像重建），用于验证 VAE 对细节的保真度
            Args:
                img: 输入图像 [B, 3, H, W]
        """
        with torch.no_grad():
            # 1. 编码得到后验分布
            posterior = self.vae.encode(img).latent_dist
            # 2. 推理时直接使用均值 mu，丢弃方差噪声以获得确定性结果
            z = posterior.mode()
            print(z.shape)
            # 3. 直接解码
            recons = self.vae.decode(z).sample
            return recons


    def sample(self, bs, img_size=(256, 256)):
        """从纯高斯噪声中采样并解码(如果是扩散模型用的vae, 这里解码出来基本无意义, 依然是噪声)
            Args:
                bs: 采样数量 (Batch Size)
                img_size: 目标生成的图像尺寸，用于推导潜在特征图的 H 和 W
        """
        device = next(self.vae.encoder.parameters()).device

        # 计算下采样n倍后的潜在空间尺寸
        h_latent, w_latent = img_size[0] // self.down_scale, img_size[1] // self.down_scale
        # 生成d维度的标准正态分布随机噪声 N(0, 1)
        z = torch.randn(bs, self.latent_dim, h_latent, w_latent).to(device)
        with torch.no_grad():
            # 极其关键：还原缩放比例. 生成的纯噪声方差为 1，而在真实 VAE 解码前，必须除以缩放因子还原
            z_unscaled = z / self.vae.config.scaling_factor
            # 解码
            recons = self.vae.decode(z_unscaled).sample
            return recons.float().cpu().numpy()




















# ========== Debug 用例 ==========
if __name__ == '__main__':
    cfg = dict(
        type='HFVAE',
        weight_dir='ckpts/hugging_face/vae-kl-f8-d16',
        latent_dim = 16,
        down_scale = 8,
    )
    
    # 由 Registry 构建模型
    vision_encoder = MODELS.build_from_cfg(cfg)
    
    # 假设输入一张 1024x1024 的高分辨率遥感切片 (Batch=1)
    img = torch.randn(1, 3, 256, 256)
    
    # 前向提取特征
    output_features = vision_encoder(img)
    
    print(f"📦 输入图像尺寸: {img.shape}")
    print(f"🧩 提取的 VAE 特征尺寸: {output_features.shape}") 
    # 预期输出: [1, 16, 128, 128] (空间缩小8倍，通道变为16)