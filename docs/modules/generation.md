# 生成模型模块 (Generation)

生成模型模块位于 `generation/` 目录，实现了多种生成范式，包括扩散模型、Flow Matching 和自编码器。

## 1. 目录结构

```
generation/
├── models/
│   ├── diffusion/         # 扩散模型
│   │   ├── ddpm.py         # DDPM 基础实现
│   │   ├── ddim.py         # DDIM 加速采样
│   │   └── sde.py          # SDE 变体
│   ├── flow_matching/     # Flow Matching
│   │   ├── rect_flow.py    # Rectified Flow
│   │   └── linear_ot.py    # 线性最优传输
│   ├── latent/            # 潜空间扩散
│   │   ├── ldm.py          # Latent Diffusion Model
│   │   ├── mask_ldm.py     # Mask Git 风格
│   │   └── class_ldm.py    # 类别条件 LDM
│   ├── transformer/        # Diffusion Transformer
│   │   ├── dit.py          # DiT 主模型
│   │   └── dit_crossattn.py # 交叉注意力 DiT
│   ├── autoencoder/        # 自编码器
│   │   ├── vae.py          # VAE
│   │   ├── vqvae.py        # VQ-VAE
│   │   └── cvae.py         # 条件 VAE
│   ├── modules/            # 通用模块
│   │   ├── unet.py         # U-Net
│   │   ├── attention.py    # 注意力机制
│   │   └── timestep.py     # 时间步嵌入
│   └── losses/             # 损失函数
│       └── diffusion_loss.py
├── schedulers/             # 噪声调度器
│   ├── ddpm_scheduler.py   # DDPM 调度
│   └── flow_matching_scheduler.py
├── train.py                # 训练脚本
└── README.md
```

## 2. 扩散模型 (Diffusion Models)

### 2.1 DDPM (Denoising Diffusion Probabilistic Models)

DDPM 通过逐步加噪和去噪实现数据生成。

#### 数学原理

**前向过程 (q)**：逐步向数据添加高斯噪声
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
```

**逆向过程 (p)**：学习去噪
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

#### DDPM 实现

```python
@MODELS.register
class DDPM(nn.Module):
    def __init__(self, denoise_model, img_size, beta_start=0.0001, beta_end=0.02,
                 num_timesteps=1000, beta_schedule='linear'):
        super().__init__()
        self.denoise_model = denoise_model
        self.img_size = img_size
        self.num_timesteps = num_timesteps
        
        # 噪声调度
        betas = self._get_beta_schedule(beta_schedule, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - alphas_cumprod))
    
    def q_sample(self, x_start, t, noise=None):
        """前向加噪过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t
    
    def p_mean_variance(self, x_t, t, condition=None):
        """预测均值和方差"""
        model_output = self.denoise_model(x_t, t, condition)
        
        # 简化的均值预测
        pred_noise = model_output
        x_0_pred = (x_t - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * pred_noise) / \
                   self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        
        # 计算逆向过程均值
        model_mean = self._extract(self.sqrt_alphas_cumprod_prev, t, x_t.shape) * x_0_pred + \
                    self._extract(self.sqrt_one_minus_alphas_cumprod_prev, t, x_t.shape) * pred_noise
        
        return model_mean
    
    @torch.no_grad()
    def p_sample(self, x_t, t, condition=None):
        """单步去噪采样"""
        model_mean = self.p_mean_variance(x_t, t, condition)
        if t > 0:
            noise = torch.randn_like(x_t)
            x_prev = model_mean + self._extract(self.sqrt_betas, t, x_t.shape) * noise
        else:
            x_prev = model_mean
        return x_prev
    
    @torch.no_grad()
    def sample(self, batch_size=None, condition=None, return_intermediates=False):
        """完整采样"""
        bs = batch_size or 1
        shape = (bs, *self.img_size)
        x_t = torch.randn(shape, device=next(self.parameters()).device)
        
        intermediates = [] if return_intermediates else None
        
        for t in reversed(range(self.num_timesteps)):
            x_t = self.p_sample(x_t, t, condition)
            if return_intermediates:
                intermediates.append(x_t)
        
        return x_t if intermediates is None else (x_t, intermediates)
```

### 2.2 DDIM (Denoising Diffusion Implicit Models)

DDIM 提供更快的确定性采样。

```python
@MODELS.register
class DDIM(DDPM):
    def __init__(self, *args, ddim_num_steps=50, ddim_eta=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ddim_num_steps = ddim_num_steps
        self.ddim_eta = ddim_eta
    
    @torch.no_grad()
    def ddim_p_sample(self, x_t, t, t_prev, eta=0.0, condition=None):
        """DDIM 单步采样"""
        pred_noise = self.denoise_model(x_t, t, condition)
        x_0_pred = (x_t - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * pred_noise) / \
                   self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        
        # 确定性路径
        alpha_t = self.ddim_alpha(t)
        alpha_t_prev = self.ddim_alpha(t_prev)
        
        predicted_x0 = x_0_pred.clamp(-1, 1)
        direction_pointing_to_xt = torch.sqrt(1 - alpha_t_prev) * pred_noise
        
        x_prev = torch.sqrt(alpha_t_prev) * predicted_x0 + direction_pointing_to_xt
        
        return x_prev
    
    @torch.no_grad()
    def ddim_sample(self, batch_size=None, condition=None):
        """DDIM 完整采样"""
        bs = batch_size or 1
        step_ratio = self.num_timesteps // self.ddim_num_steps
        
        times = list(range(0, self.num_timesteps, step_ratio))
        times_next = times[1:] + [0]
        
        x_t = torch.randn((bs, *self.img_size), device=next(self.parameters()).device)
        
        for t, t_next in zip(reversed(times), reversed(times_next)):
            x_t = self.ddim_p_sample(x_t, t, t_next, self.ddim_eta, condition)
        
        return x_t
```

## 3. Flow Matching

### 3.1 Rectified Flow

通过常微分方程定义从噪声到数据的路径。

```python
@MODELS.register
class RectifiedFlow(nn.Module):
    def __init__(self, model, img_size, num_timesteps=1000):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.num_timesteps = num_timesteps
    
    def interpolate(self, x0, x1, t):
        """线性插值: x_t = (1-t) * x0 + t * x1"""
        return (1 - t) * x0 + t * x1
    
    def velocity(self, x_t, t, condition=None):
        """学习速度场: v = x1 - x0"""
        return self.model(x_t, t, condition)
    
    def forward(self, x0, t, condition=None):
        """训练: 学习从 x0 到 x1 的速度"""
        x1 = torch.randn_like(x0)  # 目标（噪声）
        t = t.view(-1, 1, 1, 1).expand_as(x0)
        
        x_t = self.interpolate(x0, x1, t)
        v_target = x1 - x0
        
        v_pred = self.velocity(x_t, t.squeeze(), condition)
        
        return v_pred, v_target
    
    @torch.no_grad()
    def sample(self, batch_size=None, condition=None):
        """ODE 求解采样"""
        bs = batch_size or 1
        x_t = torch.randn((bs, *self.img_size), device=next(self.parameters()).device)
        
        dt = 1.0 / self.num_timesteps
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((bs,), t / self.num_timesteps, 
                                  device=x_t.device)
            
            with torch.no_grad():
                v = self.velocity(x_t, t_tensor, condition)
            
            x_t = x_t - dt * v
        
        return x_t
```

## 4. 潜空间扩散模型 (Latent Diffusion)

### 4.1 LDM (Latent Diffusion Model)

在潜空间中进行扩散，结合自编码器压缩图像。

```python
@MODELS.register
class LatentDiffusionModel(nn.Module):
    def __init__(self, autoencoder, diffusion_model, img_size, latent_scale_factor=8):
        super().__init__()
        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model
        self.latent_scale_factor = latent_scale_factor
        
        latent_dim = img_size[0] // latent_scale_factor
        self.latent_shape = (latent_dim, latent_dim, autoencoder.latent_channels)
    
    def encode(self, x):
        """编码到潜空间"""
        return self.autoencoder.encode(x)
    
    def decode(self, z):
        """从潜空间解码"""
        return self.autoencoder.decode(z)
    
    def forward(self, x, t, condition=None):
        """训练: 在潜空间进行扩散"""
        z = self.encode(x)
        predicted_noise, noise = self.diffusion_model(z, t, condition)
        return predicted_noise, noise
    
    @torch.no_grad()
    def sample(self, batch_size=None, condition=None):
        """采样: 先扩散再解码"""
        z = self.diffusion_model.sample(batch_size, condition)
        x = self.decode(z)
        return x
```

## 5. Diffusion Transformer (DiT)

### 5.1 DiT 主模型

使用 Transformer 替代 U-Net 作为去噪模型。

```python
@MODELS.register
class DiT(nn.Module):
    def __init__(self, img_size, patch_size=2, in_channels=4, hidden_size=1152,
                 num_heads=16, num_layers=28, mlp_ratio=4.0, class_free=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        self.x_embedder = nn.Linear(in_channels * patch_size * patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, in_channels * patch_size * patch_size)
        
        if class_free:
            self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
    
    def forward(self, x, t, y=None):
        x = patchify(x, self.patch_size)
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        x = x + t
        
        if y is not None and hasattr(self, 'y_embedder'):
            y = self.y_embedder(y)
            x = x + y
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm_final(x)
        x = self.proj_out(x)
        x = self.unpatchify(x)
        
        return x
```

## 6. 自编码器 (Autoencoders)

### 6.1 VAE (Variational Autoencoder)

```python
@MODELS.register
class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim=4):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        
        self.fc_mu = nn.Linear(encoder.out_channels, latent_dim)
        self.fc_logvar = nn.Linear(encoder.out_channels, latent_dim)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss(self, x):
        x_recon, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss
```

### 6.2 VQ-VAE (Vector Quantized VAE)

```python
@MODELS.register
class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim=256, num_embeddings=8192):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def encode(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        min_encoding_idx = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_idx).view(z.shape)
        
        return z_q, min_encoding_idx
    
    def decode(self, z_q):
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)
    
    def forward(self, x):
        z, encoding_idx = self.encode(x)
        z_q_st = z + (self.embedding(encoding_idx) - z).detach()
        x_recon = self.decode(z_q_st)
        
        commitment_loss = F.mse_grad(z.detach(), z_q)
        codebook_loss = F.mse_loss(z, z_q.detach())
        
        return x_recon, codebook_loss, commitment_loss
```

## 7. 条件生成与 CFG

### 7.1 Classifier-Free Guidance

```python
def classifier_free_guidance(model_output, guidance_scale=7.5, 
                             conditional_output=None, unconditional_output=None):
    """
    CFG 实现
    """
    if guidance_scale == 0 or conditional_output is None:
        return model_output
    
    guided_output = unconditional_output + guidance_scale * (conditional_output - unconditional_output)
    return guided_output
```

## 8. 训练配置示例

```yaml
model:
  type: LatentDiffusionModel
  autoencoder:
    type: VAE
    latent_dim: 4
    latent_scale_factor: 8
  diffusion_model:
    type: DDPM
    denoise_model:
      type: UNet
      in_channels: 4
      base_channels: 128
    num_timesteps: 1000
  img_size: [64, 64]

scheduler:
  type: DDIM
  ddim_num_steps: 50
  ddim_eta: 0.0

cfg:
  guidance_scale: 7.5
  unconditional_prob: 0.1
```

## 9. 扩展方式

### 新增扩散模型

```python
@MODELS.register
class NewDiffusionModel(nn.Module):
    def __init__(self, denoise_model, ...):
        super().__init__()
        self.denoise_model = denoise_model
    
    def forward(self, x, t, condition=None):
        # 定义前向/损失计算
        ...
    
    @torch.no_grad()
    def sample(self, batch_size, condition=None):
        # 定义采样策略
        ...
```

### 新增自编码器

```python
@MODELS.register
class NewAutoEncoder(nn.Module):
    def encode(self, x):
        return z
    
    def decode(self, z):
        return x_recon
    
    def forward(self, x):
        return self.decode(self.encode(x))
```
