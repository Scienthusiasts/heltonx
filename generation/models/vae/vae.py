import torch
import torch.nn as nn
from functools import partial
from heltonx.utils.register import MODELS
from heltonx.utils.utils import init_weights
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
import torch.nn.functional as F




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
                    # 比编码器多一些参数
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
class VAE(nn.Module):
    def __init__(self, input_dim, layer_dims, latent_dim, img_size, kld_weight, load_ckpt=None):
        """variation auto encoder
            Args:
                input_dim:  输入通道数, 一般为3
                layer_dims: 每一层通道数, 例 [64, 128, 256, 512] 每经过一层，特征图尺寸 / 2
                latent_dim: 潜在向量的维度 (e.g., 128)
                img_size:   输入图片尺寸 (H, W)，用于计算 Flatten 大小
                kld_weight: kld损失的权重(一般很小)
            Returns:

        """
        super().__init__()
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        # 1. Encoder
        self.encoder = Encoder(input_dim, layer_dims)

        # 计算 Flatten 维度, 每一层 layer_dims 对应一次 stride=2 的下采样
        downsample_factor = 2 ** len(layer_dims)
        self.feat_h = img_size[0] // downsample_factor
        self.feat_w = img_size[1] // downsample_factor
        self.last_channels = layer_dims[-1]
        
        # 计算encoder输出特征拉平后的总维度
        self.flat_dim = self.last_channels * self.feat_h * self.feat_w
        # 2. Latent Projection, 将encoder特征变换到潜在特征的维度
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim, latent_dim)
        # 3. 将latent vector映射到decoder输入的维度
        self.fc_decoder = nn.Linear(latent_dim, self.flat_dim)
        # 4. Decoder
        self.decoder = Decoder(input_dim, layer_dims)
        # 权重初始化
        self.init_weights()
        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)

    def reparameterize(self, mu, log_var):
        # 显式截断 log_var，防止 exp() 爆炸
        # 限制 log_var 在 [-10, 10] 之间 
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, batch_datas=None, return_loss=True, bs=None):
        """前向/损失
            Args:
                batch_datas:  输入图像
                return_loss:  是否训练模式
                bs:           采样的batch size
        """
        if return_loss:
            x = batch_datas[0] if isinstance(batch_datas, (list, tuple)) else batch_datas
            bs = x.shape[0]
            '''编码器 [B, C, H, W] -> [B, flat_dim]'''
            flat_feat = self.encoder(x).view(bs, -1)
            
            '''生成latent vector'''
            mu = self.fc_mu(flat_feat)
            log_var = self.fc_var(flat_feat)
            # 重参数采样(根据encoder特征的μ和σ采样高斯噪声)
            z = self.reparameterize(mu, log_var)
            
            '''解码器'''
            # [bs, latent] -> [bs, flat_dim] -> [bs, C, H, W]
            z_proj = self.fc_decoder(z).view(bs, self.last_channels, self.feat_h, self.feat_w)
            # [bs, C, H, W]-> [bs, 3, img_H, img_W]
            recons = self.decoder(z_proj)
            return self.losses(recons, x, mu, log_var)
        else:
            with torch.no_grad():
                return self.sample(bs)



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


    def sample(self, bs):
        """随机采样标准高斯噪声生成图片
            Args:
                num_samples:    生成图片的数量 (Batch Size)
                current_device: 当前设备 (cpu/cuda)
            Returns:
                samples: 生成的图片 [num_samples, output_dim, H, W]
        """
        device = next(self.encoder.parameters()).device
        # 1.从标准正态分布采样 z: [num_samples, latent_dim]
        z = torch.randn(bs, self.latent_dim).to(device)
        # 2.通过全连接层映射回 Flatten 维度
        z_proj = self.fc_decoder(z)
        # 3.形状变换: [B, flat_dim] -> [B, channels, h, w]
        z_reshaped = z_proj.view(bs, self.last_channels, self.feat_h, self.feat_w)
        # 4. 解码生成图片
        samples = self.decoder(z_reshaped)
        return samples.detach().float().cpu().numpy()


    # def sample(self, bs):
    #     """
    #     使用 4 个随机 z 作为角点，在 latent space 上进行 mxm 网格插值采样

    #     Args:
    #         bs: batch size, 要求 bs = m * m
    #     Returns:
    #         samples: [bs, C, H, W]
    #     """
    #     device = next(self.encoder.parameters()).device
    #     # ---------- 1. 检查 bs 是否为完全平方 ----------
    #     m = int(math.sqrt(bs))
    #     assert m * m == bs, "bs 必须是某个整数 m 的平方 (bs = m * m)"
    #     # ---------- 2. 采样 4 个角点 latent ----------
    #     z_tl = torch.randn(self.latent_dim, device=device)  # top-left
    #     z_tr = torch.randn(self.latent_dim, device=device)  # top-right
    #     z_bl = torch.randn(self.latent_dim, device=device)  # bottom-left
    #     z_br = torch.randn(self.latent_dim, device=device)  # bottom-right
    #     # ---------- 3. 构造 mxm 网格插值 ----------
    #     z_list = []
    #     for i in range(m):          # 行 (vertical)
    #         v = i / (m - 1) if m > 1 else 0.0
    #         for j in range(m):      # 列 (horizontal)
    #             u = j / (m - 1) if m > 1 else 0.0
    #             z_ij = (
    #                 (1 - u) * (1 - v) * z_tl +
    #                 u * (1 - v) * z_tr +
    #                 (1 - u) * v * z_bl +
    #                 u * v * z_br
    #             )
    #             z_list.append(z_ij)
    #     # [bs, latent_dim]
    #     z = torch.stack(z_list, dim=0)
    #     # ---------- 4. 解码 ----------
    #     z_proj = self.fc_decoder(z)
    #     z_reshaped = z_proj.view(bs, self.last_channels, self.feat_h, self.feat_w)
    #     samples = self.decoder(z_reshaped)

    #     return samples.detach().float().cpu().numpy()



    def reconstruct(self, img, bs=1):
        """前向/损失
            Args:
                img: 输入图像 [B, C, H, W]
                bs:  采样的batch size
        """
        with torch.no_grad():
            x = img
            '''编码器 [B, C, H, W] -> [B, flat_dim]'''
            flat_feat = self.encoder(x).view(bs, -1)
            
            '''生成latent vector'''
            mu = self.fc_mu(flat_feat)
            # log_var = self.fc_var(flat_feat)
            # 重参数采样(根据encoder特征的μ和σ采样高斯噪声)
            # z = self.reparameterize(mu, log_var)
            z = mu

            '''解码器'''
            # [bs, latent] -> [bs, flat_dim] -> [bs, C, H, W]
            z_proj = self.fc_decoder(z).view(bs, self.last_channels, self.feat_h, self.feat_w)
            # [bs, C, H, W]-> [bs, 3, img_H, img_W]
            recons = self.decoder(z_proj)
            return recons




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















# ========== Debug 用例 ==========
if __name__ == '__main__':
    # 配置测试参数
    init_dim = 64
    img_size = (128, 128)
    latent_dim = 256
    
    # 实例化 VAE 模型
    model = VAE(
        input_dim=3,
        layer_dims=[init_dim, init_dim*2, init_dim*4], # [64, 128, 256] -> 下采样 2 次 (128->64->32)
        latent_dim=latent_dim,
        img_size=img_size,
        resnet_block_groups=4
    )

    # ========== Step 1. 统计模型参数量 ==========
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧩 模型参数统计 (AutoencoderKL):")
    print(f"  ➤ 总参数量：{total_params:,}")
    print(f"  ➤ 可训练参数量：{trainable_params:,}")
    print(f"  ➤ 参数占用显存约：{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 打印一些关键维度信息以供检查
    print(f"  ➤ Flatten前特征图尺寸: {model.feat_h} x {model.feat_w}")
    print(f"  ➤ Flatten维度: {model.flat_dim} -> Latent维度: {model.latent_dim}")

    # ========== Step 2. 构造输入 ==========
    B, C, H, W = 2, 3, img_size[0], img_size[1]
    x = torch.randn(B, C, H, W)
    print(f"\n📦 构造输入数据: shape={x.shape}")

    # ========== Step 3. 前向传播 (Train模式) ==========
    # VAE 训练时返回的是 Loss 字典
    outputs = model(x, return_loss=True)
    print(f"\n🚀 前向传播 (Return Loss=True):")
    for k, v in outputs.items():
        print(f"  ➤ {k}: {v.item():.4f}")

    # ========== Step 4. 反向传播测试 ==========
    loss = outputs['loss']
    loss.backward()
    print("✅ 反向传播成功！梯度已计算。")

    # ========== Step 5. 推理/重构测试 (Eval模式) ==========
    model.eval()
    with torch.no_grad():
        # 测试重构 (Return Loss=False)
        recons = model(x, return_loss=False)
        print(f"\n📸 重构输出 (Return Loss=False): shape={recons.shape}")
        assert recons.shape == x.shape, "重构图片尺寸与输入不一致！"
        
        # 测试随机采样
        samples = model.sample(num_samples=4, current_device=x.device)
        print(f"🎲 随机采样 (Sample): shape={samples.shape}")
        assert samples.shape == (4, C, H, W), "采样图片尺寸不正确！"

    print("\n🎉 所有测试通过！")