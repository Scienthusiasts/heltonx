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
                x:
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
                x: 
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















@MODELS.register
class CVAE(nn.Module):

    def __init__(self, c_proj_model:nn.Module, input_dim, layer_dims, latent_dim, condition_emb_dim, vocab_size, img_size, kld_weight, z_drop_prob=0.5, z_drop_ratio=0.5, load_ckpt=None):
        """
        Args:
            self: 说明
            c_proj_model: 说明
            c_proj_model: nn.Module
            input_dim: 说明
            layer_dims: 说明
            latent_dim: 说明
            condition_emb_dim: 说明
            vocab_size: 说明
            img_size: 说明
            kld_weight: 说明
            z_drop_prob: 说明
            z_drop_ratio: 说明
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.img_size = img_size
        
        # 1. Encoder (输入维度 = 图片维度，不加 Label)
        self.encoder = Encoder(input_dim, layer_dims)

        # 2.latent space projection
        downsample_factor = 2 ** len(layer_dims)
        self.feat_h = img_size[0] // downsample_factor
        self.feat_w = img_size[1] // downsample_factor
        self.last_channels = layer_dims[-1]
        self.flat_dim = self.last_channels * self.feat_h * self.feat_w
        
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim, latent_dim)

        # condition Embedding (将离散的文本tokens映射成连续的embeddings)
        self.c_embedding = nn.Embedding(vocab_size, condition_emb_dim)
        # c_proj的输入为[bs, seq_len, emb_dim], 输出为[bs, emb_dim] (用于整合文本condition特征)
        self.c_proj = c_proj_model 
        # 保存掩码策略参数
        self.z_drop_prob = z_drop_prob  
        self.z_drop_ratio = z_drop_ratio  

        # 3. Decoder (输入维度 = Latent + Label)
        decoder_in_dim = latent_dim + condition_emb_dim
        self.fc_decoder = nn.Linear(decoder_in_dim, self.flat_dim)
        self.decoder = Decoder(input_dim, layer_dims) # 注意参数顺序
        
        self.init_weights()
        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)


    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, batch_datas=None, return_loss=True, bs=None, c_tokens=None):
        if return_loss:
            x, c_tokens = batch_datas[0], batch_datas[1]
            bs = x.shape[0]
            
            '''Encoder: 只看图, 不看条件'''
            # 直接输入 x
            flat_feat = self.encoder(x).view(bs, -1)
            mu = self.fc_mu(flat_feat)
            log_var = self.fc_var(flat_feat)
            # 重参数采样, 保证可导性
            z = self.reparameterize(mu, log_var)

            '''处理条件文本特征'''
            # [bs, seq_len] -> [bs, seq_len, emb_dim]
            c_emb_seq = self.c_embedding(c_tokens)
            # [bs, seq_len, emb_dim] -> [bs, emb_dim]
            c_emb = self.c_proj(c_emb_seq)

            '''添加下面的策略防止模型只根据z重建输入而不依赖condition, 或者过于依赖condition导致过拟合或多样性下降'''
            # 仅在训练且主开关打开时执行 (利用 z_drop_prob > 0 作为开启此功能的开关)
            if self.z_drop_prob > 0:
                # 1. 生成 [0, 1] 均匀分布的随机数
                # shape: [bs, 1]
                rand_probs = torch.rand(bs, 1, device=z.device)
                # 2. 定义保留比例 (1 - b%)
                keep_prob = 1.0 - self.z_drop_ratio
                # --- 分支 A: 随机数 < 1/3 -> Mask Z (b%) ---
                mask_z_indices = rand_probs < (1.0 / 3.0)
                if mask_z_indices.any():
                    # 生成针对 z 的特征掩码 (伯努利分布)
                    z_mask = torch.bernoulli(torch.full_like(z, keep_prob))
                    # 仅修改选中的样本，其他样本保持原样
                    z = torch.where(mask_z_indices, z * z_mask, z)
                # --- 分支 B: 1/3 <= 随机数 < 2/3 -> Mask Condition (b%) ---
                mask_c_indices = (rand_probs >= (1.0 / 3.0)) & (rand_probs < (2.0 / 3.0))
                if mask_c_indices.any():
                    # 生成针对 c_emb 的特征掩码
                    c_mask = torch.bernoulli(torch.full_like(c_emb, keep_prob))
                    # 仅修改选中的样本
                    c_emb = torch.where(mask_c_indices, c_emb * c_mask, c_emb)
                # --- 分支 C: 随机数 >= 2/3 -> 不做任何操作 (隐式包含) ---

            '''Decoder'''
            # 拼接 z 和 聚合后的 c_emb
            z_cond = torch.cat([z, c_emb], dim=1) 
            z_proj = self.fc_decoder(z_cond).view(bs, self.last_channels, self.feat_h, self.feat_w)
            recons = self.decoder(z_proj)
            return self.losses(recons, x, mu, log_var)
        else:
            with torch.no_grad():
                return self.sample(bs, c_tokens)


    def losses(self, recons, input_img, mu, log_var):
        """计算损失
            Args:
                recons:    生成图片的数量 (Batch Size)
                input_img: 当前设备 (cpu/cuda)
                mu:        latent vector的均值
                log_var:   latent vector的方差    
            Returns:
                samples: 生成的图片 [num_samples, output_dim, H, W]
        """
        # 再次 clamp 确保计算 Loss 安全
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        # 1. Reconstruction Loss (约束重建图像与输入图像一致)
        recons_loss = F.smooth_l1_loss(recons, input_img, reduction='mean') 
        # 2. KLD Loss (约束latent vector所在的分布空间尽量符合标准正态分布)
        # 公式: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        # loss以字典组织
        return {'Recon_Loss': recons_loss, 'KLD': kld_loss * self.kld_weight}


    def sample(self, bs, c_tokens):
        """
            Args:
                bs:       batch size
                c_tokens: 条件文本tokens [bs, seq_len]
        """
        device = next(self.encoder.parameters()).device
        # 1. 采样 latent vector z
        z = torch.randn(bs, self.latent_dim).to(device)
        # 2. 处理文本条件
        c_emb_seq = self.c_embedding(c_tokens.to(device))
        c_emb = self.c_proj(c_emb_seq)
        # 4. 拼接 Condition
        z_cond = torch.cat([z, c_emb], dim=1)
        z_proj = self.fc_decoder(z_cond)
        z_reshaped = z_proj.view(bs, self.last_channels, self.feat_h, self.feat_w)
        samples = self.decoder(z_reshaped)
        return samples.detach().float().cpu().numpy()


    def init_weights(self):
        """权重初始化方法
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # 卷积层 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 💥 全连接层 Xavier 初始化 (防止梯度爆炸)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
