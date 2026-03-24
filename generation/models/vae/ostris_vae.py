import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from heltonx.utils.register import MODELS
from heltonx.utils.utils import init_weights
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix




class VGGPerceptualLoss(nn.Module):
    """预训练 VGG 感知损失 (Perceptual Loss / LPIPS)
    """
    def __init__(self, resize=True):
        super().__init__()
        # 加载预训练的 VGG16 特征提取层
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # 提取第 4, 9, 16, 23, 30 层 (通常对应 relu1_2, relu2_2, relu3_3, relu4_3, relu5_3)
        self.blocks = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23], vgg[23:30]
        ])
        for p in self.parameters():
            p.requires_grad = False # 冻结 VGG 权重
        
        self.resize = resize
        # ImageNet 归一化参数
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # 将输入规范化到 [0, 1] 然后进行 ImageNet 标准化 (假设原图在 [-1, 1])
        input = (input + 1) / 2
        target = (target + 1) / 2
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x, y = input, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            # 计算不同深度的特征图之间的 L1 距离
            loss += F.l1_loss(x, y)
        return loss







class NLayerDiscriminator(nn.Module):
    """标准的 PatchGAN 判别器，用于评估图像局部的真伪"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult), # 判别器常保留 BN
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)





class ResnetBlock(nn.Module):
    """为 VAE 引入的残差块，使用 GroupNorm 替代 BatchNorm"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 如果输入输出通道不一致，使用 1x1 卷积调整
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)




class Encoder(nn.Module):
    def __init__(self, input_dim, layer_dims):
        """支持空间特征保留的 f8 编码器
            Args:
                input_dim:  输入通道数, 一般为3
                layer_dims: 每一层通道数, 例 [128, 256, 512] -> 对应 3 次下采样 (f8)
        """
        super().__init__()
        modules = []
        in_channels = input_dim
        
        # 初始特征提取
        self.conv_in = nn.Conv2d(in_channels, layer_dims[0], kernel_size=3, padding=1)
        in_channels = layer_dims[0]

        # f8 编码器结构 (空间维度缩小 8 倍)
        for out_channels in layer_dims:
            modules.append(
                nn.Sequential(
                    ResnetBlock(in_channels, out_channels),
                    # 使用 stride=2 进行下采样，替代池化
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                )
            )
            in_channels = out_channels
            
        self.encode_layers = nn.Sequential(*modules)
        # Mid Block 进一步提取高级语义
        self.mid_block = ResnetBlock(in_channels, in_channels)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.encode_layers(x)
        x = self.mid_block(x)
        return x




class Decoder(nn.Module):
    def __init__(self, output_dim, layer_dims):
        """恢复 f8 空间分辨率的解码器"""
        super().__init__()
        modules = []
        # layer_dims 翻转: e.g., [512, 256, 128]
        dims = list(reversed(layer_dims))
        
        # 接收投影后的特征
        in_channels = dims[0]
        self.mid_block = ResnetBlock(in_channels, in_channels)

        # 构建解码层，逐步上采样 (*8)
        for i in range(len(dims)):
            out_channels = dims[i+1] if i < len(dims) - 1 else dims[-1]
            modules.append(
                nn.Sequential(
                    ResnetBlock(in_channels, out_channels),
                    # 扩散模型常用最近邻插值+卷积来进行上采样，减少棋盘效应
                    nn.Upsample(scale_factor=2.0, mode='nearest'),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                )
            )
            in_channels = out_channels
            
        self.decode_layers = nn.Sequential(*modules)
        # 最后一层恢复到 RGB 通道数
        self.norm_out = nn.GroupNorm(32, in_channels)
        self.conv_out = nn.Conv2d(in_channels, output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.mid_block(x)
        x = self.decode_layers(x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x








@MODELS.register
class VAE_KL_f8_d16(nn.Module):
    def __init__(self, input_dim=3, layer_dims=[128, 256, 512], latent_dim=16, kld_weight=1e-5):
        """KL-f8-d16 VAE 架构
            Args:
                input_dim:  输入通道数 (3)
                layer_dims: 推荐 [128, 256, 512] 以实现 f8 (3次下采样)
                latent_dim: 潜在矩阵的深度，这里固定为 16 (d16)
                kld_weight: KLD 损失权重，在高质量重建任务中通常极小 (e.g., 1e-5)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.last_channels = layer_dims[-1]
        
        # 1. Encoder
        self.encoder = Encoder(input_dim, layer_dims)
        # 2. Latent Projection (完全抛弃 Linear，使用 1x1 卷积)
        # 将 encoder 输出通道映射到 2 * latent_dim (16均值 + 16对数方差 = 32)
        self.quant_conv = nn.Conv2d(self.last_channels, 2 * latent_dim, kernel_size=1)
        # 3. 将采样的 d16 潜在特征映射回 Decoder 的输入通道
        self.post_quant_conv = nn.Conv2d(latent_dim, self.last_channels, kernel_size=1)
        # 4. Decoder
        self.decoder = Decoder(input_dim, layer_dims)
        
        # 损失函数模块
        self.perceptual_loss_fn = VGGPerceptualLoss().eval() # 冻结模式
        # self.discriminator = NLayerDiscriminator(input_nc=input_dim)

        self.init_weights()


    def reparameterize(self, mu, log_var):
        # 限制 log_var 在合理区间
        log_var = torch.clamp(log_var, min=-30.0, max=20.0)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, batch_datas, return_loss=True):
        x = batch_datas[0] if isinstance(batch_datas, (list, tuple)) else batch_datas
        
        '''1. VAE 编码与重参数化'''
        # 编码器: [B, 3, H, W] -> [B, last_channels, H/8, W/8]
        enc_feat = self.encoder(x)
        # 生成 32 通道的分布参数，并在通道维度切分为 μ 和 log_var
        moments = self.quant_conv(enc_feat)
        mu, log_var = torch.chunk(moments, 2, dim=1)
        # 重参数采样: 得到 [B, 16, H/8, W/8] 的空间潜在矩阵 (d16)
        z = self.reparameterize(mu, log_var)

        if not return_loss:
            return z, mu

        '''2. VAE 解码重建'''
        z_proj = self.post_quant_conv(z)
        recons = self.decoder(z_proj)

        '''3. 统一计算所有损失'''
        return self.losses(recons, x, mu, log_var)


    def losses(self, recons, input_img, mu, log_var):

        '''计算 VAE (Generator) 的损失'''
        # 1. L1 重建损失
        recons_loss = F.l1_loss(recons, input_img, reduction='mean')
        # 2. LPIPS 感知损失
        p_loss = self.perceptual_loss_fn(recons, input_img)
        # 3. KLD 散度损失
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=[1, 2, 3]))
        # 4. GAN Generator Loss
        # VAE 希望判别器把生成的假图判断为真 
        # logits_fake_for_G = self.discriminator(recons)
        # g_loss = -torch.mean(logits_fake_for_G)
        # VAE 总损失
        loss_vae = recons_loss + \
                   self.perceptual_weight * p_loss + \
                   self.kld_weight * kld_loss
                #    self.gan_weight * g_loss

        '''计算 Discriminator (判别器) 的损失'''
        # # 1. 判别器看真图 (希望得分 > 1)
        # logits_real = self.discriminator(input_img.detach())
        # d_loss_real = torch.mean(F.relu(1. - logits_real))
        # # 2. 判别器看假图 (希望得分 < -1)
        # # ⚠️ 极其关键：必须对 recons 使用 detach()，切断梯度回传给 VAE！
        # logits_fake_for_D = self.discriminator(recons.detach())
        # d_loss_fake = torch.mean(F.relu(1. + logits_fake_for_D))
        # # 判别器总损失
        # loss_d = 0.5 * (d_loss_real + d_loss_fake)

        return {
            # 'loss_d': loss_d,
            'L1_Loss': recons_loss,
            'Perceptual_Loss': self.perceptual_weight * p_loss,
            'KLD_Loss': self.kld_weight * kld_loss,
            # 'GAN_G_Loss': g_loss.detach(),
            # 'D_Real_Loss': d_loss_real.detach(),
            # 'D_Fake_Loss': d_loss_fake.detach()
        }


    def reconstruct(self, img):
        with torch.no_grad():
            enc_feat = self.encoder(img)
            moments = self.quant_conv(enc_feat)
            mu, _ = torch.chunk(moments, 2, dim=1)
            # 推理时通常直接使用均值 mu，丢弃方差噪声以获得确定性结果
            z_proj = self.post_quant_conv(mu)
            recons = self.decoder(z_proj)
            return recons


    def sample(self, num_samples, img_size=(512, 512), device='cpu'):
        """从纯高斯噪声中采样并解码 (需要注意维度是下采样 8 倍后的)"""
        h_latent, w_latent = img_size[0] // 8, img_size[1] // 8
        # 生成 d16 维度的纯随机噪声
        z = torch.randn(num_samples, self.latent_dim, h_latent, w_latent).to(device)
        with torch.no_grad():
            z_proj = self.post_quant_conv(z)
            recons = self.decoder(z_proj)
            return recons


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)









# ========== Debug 用例 ==========
if __name__ == '__main__':
    # 模拟高分辨率图像输入 (例如 1024x1024 裁切)
    img_size = (256, 256) 
    
    # 实例化 VAE 模型
    model = VAE_KL_f8_d16(
        input_dim=3,
        layer_dims=[128, 256, 512], # f8: 512 -> 256 -> 128 -> 64
        latent_dim=16               # d16 核心配置
    )

    B, C, H, W = 2, 3, img_size[0], img_size[1]
    x = torch.randn(B, C, H, W)
    print(f"📦 输入图像: {x.shape}")

    # ========== 前向传播提取特征 (Eval 模式提取 z) ==========
    model.eval()
    with torch.no_grad():
        z, mu = model(x, return_loss=False)
        print(f"🧩 提取到的潜在特征矩阵 (z): {z.shape}") 
        # 预期输出: [2, 16, 64, 64] -> 这就是保留了空间特征的 d16 矩阵！

    # ========== 前向传播计算 Loss (Train 模式) ==========
    model.train()
    outputs = model(x, return_loss=True)
    print(f"\n🚀 训练模式计算 Loss:")
    for k, v in outputs.items():
        print(f"  ➤ {k}: {v.item():.6f}")

    # ========== 推理/重构测试 ==========
    model.eval()
    with torch.no_grad():
        recons = model.reconstruct(x)
        print(f"\n📸 重构输出: shape={recons.shape}")