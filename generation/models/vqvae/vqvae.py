import torch
import torch.nn as nn
from functools import partial
from heltonx.utils.register import MODELS
from heltonx.utils.utils import init_weights
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
from generation.models.blocks import *




class Encoder(nn.Module):
    def __init__(self, input_dim, layer_dims):
        """非常朴素的encoder
            Args:
                input_dim:  输入通道数, 一般为3
                layer_dims: 每一层通道数, 例 [64, 128, 256, 512] 每经过一层，特征图尺寸 / 2
            Returns:
                x: [B, C, H, W]
        """
        super().__init__()
        modules = []
        # 编码器结构
        in_channels = input_dim
        for out_channels in layer_dims:
            modules.append(
                nn.Sequential(
                    # k=3, s=2, p=1 实现2倍下采样 (/2)
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        self.encode_layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.encode_layers(x)




class Decoder(nn.Module):
    def __init__(self, output_dim, layer_dims):
        """非常朴素的decoder, 使用转置卷积进行上采样
            Args:
                output_dim:  输出通道数, 一般为3
                layer_dims: 每一层通道数, 例 [64, 128, 256, 512] 每经过一层，特征图尺寸 / 2
            Returns:
                x:  [B, 3, H, W]
        """
        super().__init__()
        modules = []
        # layer_dims 翻转: [512, 256, 128, 64]
        dims = list(reversed(layer_dims))
        # 构建解码层 (除了最后一层)
        for i in range(len(dims) - 1):
            modules.append(
                nn.Sequential(
                    # k=3, s=2, p=1, op=1 实现上采样 (*2)
                    nn.ConvTranspose2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(dims[i+1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(dims[i+1], dims[i+1], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(dims[i+1]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.decode_layers = nn.Sequential(*modules)
        # 最后一层恢复到图像尺寸和通道数
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], output_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.decode_layers(x)
        x = self.final_layer(x)
        return x





class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, encoder_loss_w, codebook_loss_w):
        """向量量化模块 (Vector Quantizer)
        Args:
            num_embeddings: 码本大小 (K)，即有多少个离散的特征向量
            embedding_dim:  每个特征向量的维度 (D)
            encoder_loss_w: beta系数，用于平衡Encoder输出逼近Codebook的程度
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.encoder_loss_w = encoder_loss_w
        self.codebook_loss_w = codebook_loss_w
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
    
        # 初始化改成标准差为 1/dim 或直接正态分布
        self.embedding.weight.data.uniform_(-1.0 / embedding_dim, 1.0 / embedding_dim)


    def forward(self, inputs):
        """
        """
        B, C, H, W = inputs.shape
        
        '''量化'''
        # 计算输入向量与码本中所有向量的距离 (x - e)^2 
        distances = torch.sum((self.embedding.weight.data.reshape(1, self.num_embeddings, C, 1, 1) - inputs.reshape(B, 1, C, H, W))**2, 2)
        # 找到距离最近的 Embedding 索引
        encoding_indices = torch.argmin(distances, dim=1)
        # 根据索引取回对应的 Embedding 向量
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2)

        '''计算 VQ Loss'''
        losses = self.losses(inputs, quantized)
        # 直通估计 (Straight Through Estimator) 前向传播本质还是quantized, 但这样反向传播的梯度就能直接传给 inputs
        # [B, C, H, W]
        quantized = (inputs + (quantized - inputs).detach())
        return quantized, losses, encoding_indices
    

    def losses(self, ze, zq):
        """
            Args:
                ze: 重建图像
                zq: 原始图像
        """
        # 这里不直接使用F.mse_loss(zq, ze)是因为希望模型和codebook学习速度应该不一样快
        # e_latent_loss: 让 Encoder 输出靠近码本向量, 这一步优化encoder
        e_latent_loss = F.mse_loss(zq.detach(), ze, reduction='mean') * self.encoder_loss_w
        # q_latent_loss: 让码本向量靠近 Encoder 输出, 这一步优化码本
        q_latent_loss = F.mse_loss(zq, ze.detach(), reduction='mean') * self.codebook_loss_w
        losses = {'q_latent_loss':q_latent_loss, 'e_latent_loss':e_latent_loss}
        return losses






@MODELS.register
class VQVAE(nn.Module):
    def __init__(self, input_dim, layer_dims, num_embeddings, embedding_dim, img_size=None, encoder_loss_w=1, codebook_loss_w=0.25, load_ckpt=None):
        """VQ-VAE 
        Args:
            input_dim:      输入通道数
            layer_dims:     Encoder/Decoder 层通道配置
            num_embeddings: 码本容量 (K)
            embedding_dim:  潜在向量维度 (D)
            img_size:       (可选) 用于计算尺寸，VQVAE通常不需要像VAE那样展平
            commitment_cost: Commitment loss 权重
        """
        super().__init__()
        # 计算 Flatten 维度, 每一层 layer_dims 对应一次 stride=2 的下采样
        downsample_factor = 2 ** len(layer_dims)
        self.feat_h = img_size[0] // downsample_factor
        self.feat_w = img_size[1] // downsample_factor
        # 1. Encoder
        self.encoder = Encoder(input_dim, layer_dims)
        # 2. Pre-Quantization Convolution 将 Encoder 的输出通道数映射到 Embedding 维度
        # 在量化前加入BN, 强制拉回特征分布, 这能保证进入VQ的特征方差是稳定的, 不会无限变大
        self.proj_conv = nn.Conv2d(layer_dims[-1], embedding_dim, kernel_size=1)
        # 3. Vector Quantizer
        self.vq_module = VectorQuantizer(num_embeddings, embedding_dim, encoder_loss_w, codebook_loss_w)
        # 4. Post-Quantization Convolution
        # 将 Embedding 维度映射回 Decoder 需要的输入通道数
        self.post_quant_conv = nn.Conv2d(embedding_dim, layer_dims[-1], kernel_size=1)
        # 5. Decoder
        self.decoder = Decoder(input_dim, layer_dims)
        self.init_weights()
        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)


    def forward(self, batch_datas=None, return_loss=True, bs=None):
        """量化逻辑: 图片 -> 编码器 -> HxW个向量 -> 每个位置独立查表 -> HxW个索引组成的图
            而不是: 图片 -> 编码器 -> 1个向量 -> 查表 -> 1张图
            Args:
                batch_datas: [B, C, H, W]
        """
        if return_loss:
            x = batch_datas[0] if isinstance(batch_datas, (list, tuple)) else batch_datas
            # Encoder: [B, C, H, W] -> [B, last_channels, H', W']
            z = self.encoder(x)
            # Pre-Quantization: [B, embedding_dim, H', W']
            z = self.proj_conv(z)
            # Vector Quantization
            # z_q: 量化后的特征 [B, embedding_dim, H', W']
            z_q, vq_loss, _ = self.vq_module(z)
            # Post-Quantization: 准备进入 Decoder
            z_decoder_input = self.post_quant_conv(z_q)
            # Decoder
            recons = self.decoder(z_decoder_input)
            return self.losses(recons, x, vq_loss)
        else:
            # 推理模式，用于生成
            with torch.no_grad():
                return self.sample(bs)


    def losses(self, recons, input_img, vq_loss):
        """
            Args:
                recons:    重建图像
                input_img: 原始图像
                vq_loss:   来自 VectorQuantizer 的损失
        """
        # Reconstruction Loss
        recons_loss = F.smooth_l1_loss(recons, input_img, reduction='mean')
        # Total Loss = Recon Loss + VQ Loss
        losses = {'recon_loss': recons_loss}
        losses.update(vq_loss) 
        return losses


    def sample(self, bs):
        """
            注意：标准的 VQ-VAE 不能像 VAE 那样直接采样高斯噪声
            VQ-VAE 需要一个先验模型（如 PixelCNN 或 Transformer）来预测 Codebook 索引。
            这里仅演示随机采样 Codebook 索引并解码（生成的内容将是无意义的噪声拼贴）。
        """
        device = next(self.encoder.parameters()).device
        # 1. 随机采样索引 [B, H, W]
        indices = torch.randint(high=self.vq_module.num_embeddings, size=(bs, self.feat_h, self.feat_w)).to(device)
        # 2. 查表得到向量 [B, H, W, D] -> [B, D, H, W]
        z_q = self.vq_module.embedding(indices).permute(0, 3, 1, 2).contiguous()
        # 3. 解码
        z_decoder_input = self.post_quant_conv(z_q)
        samples = self.decoder(z_decoder_input)
        return samples.detach().float().cpu().numpy()


    def init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # 注意：不初始化 nn.Embedding，保留其在 __init__ 中的均匀分布初始化







