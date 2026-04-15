import torch
from torch import nn
import torch.nn.functional as F
from generation.utils.utils import *
from generation.models.unet.blocks import *
from generation.utils.var_schedule import *
from generation.models.diffusion import DDPM
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
from heltonx.utils.register import MODELS



@MODELS.register
class ClassLDM(DDPM):

    def __init__(self,
                 vae,
                 denoise_model,
                 img_size,
                 batch_size,
                 num_classes,          # ===== 新增：数据集的总类别数，用于 DiT 的空条件 token =====
                 load_ckpt=None, 
                 schedule_name="linear_beta_schedule",
                 loss_type='huber',
                 timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 cfg_drop_prob=0.15,   # 训练时丢弃条件的概率 (CFG Dropout)
                 cfg_scale=4.0         # 推理时条件引导的强度 (CFG Scale)
                 ):
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
        self.num_classes = num_classes # 记录总类别数
        
        # 记录 CFG 参数
        self.cfg_drop_prob = cfg_drop_prob
        if cfg_scale != None:
            self.cfg_scale = cfg_scale

        # 生成训练或推理会用到的参数
        self.get_init_params(schedule_name, timesteps, beta_start, beta_end)
        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)



    def forward(self, batch_data=None, return_loss=True, bs=None):
        """前向+计算损失 / 条件生成推理
        """
        if return_loss:
            # 训练阶段
            y_img = batch_data[0]     # 原图 [B, 3, H, W]
            labels = batch_data[1]    # 类别标签 [B,]
            bs = y_img.shape[0]
            
            with torch.no_grad():
                # 调用 VAE 的 forward 提取图像的潜特征 
                z_0 = self.vae(x=y_img, return_loss=True) 

            # ===== 核心：CFG 条件 Dropout =====
            # 随机以 cfg_drop_prob 概率将 labels 替换为 num_classes (即无条件 token)
            if self.cfg_drop_prob > 0:
                drop_mask = torch.rand(bs, device=y_img.device) < self.cfg_drop_prob
                labels = labels.clone()
                labels[drop_mask] = self.num_classes

            # 随机生成加噪时间步
            t = torch.randint(0, self.timesteps, (bs,), device=y_img.device).long()
            
            # 计算损失 (将 labels 作为条件传入)
            loss = self.compute_loss(x_start=z_0, t=t, labels=labels)
            losses = dict(gen_loss=loss)
            return losses
            
        else:
            # 推理阶段：条件生成必须要传入 batch_data 来获取 Labels
            if batch_data is None:
                raise ValueError("推理阶段必须传入 batch_data 以提供类别 Labels 条件！")
            
            labels = batch_data[1]
            bs = labels.shape[0]
            
            # 在潜空间中进行条件采样
            latent_series = self.sample(labels=labels, bs=bs)
            
            # 提取最后一步去噪完成的潜特征
            final_latent = latent_series[-1]  
            
            # 潜空间 -> 图像空间
            gen_img = self.vae.decode(final_latent).float().cpu().numpy()
            return gen_img



    def compute_loss(self, x_start, t, labels, noise=None):
        """重写计算损失函数，直接将 labels 传给网络
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 1. 仅对图像潜特征 z_0 进行加噪
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 2. 预测噪声 (无需通道拼接，直接通过参数 y 传递类别条件)
        predicted_noise = self.denoise_model(x_noisy, t, y=labels)
        
        # 3. 计算误差
        if self.loss_type == 'huber':
            loss = F.huber_loss(noise, predicted_noise, delta=0.1)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        else:
            loss = F.mse_loss(noise, predicted_noise)
            
        return loss



    def _predict_noise(self, x, t, labels):
        """
        内部辅助方法：统一处理条件引导和 CFG 逻辑
        """
        if hasattr(self, 'cfg_scale'):
            # Batch 维度翻倍
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t, t], dim=0)
            
            # 构建有条件 labels 和无条件 labels (id = num_classes)
            labels_uncond = torch.full_like(labels, self.num_classes)
            labels_double = torch.cat([labels, labels_uncond], dim=0)
            
            # 传给 DiT 网络进行预测
            pred_double = self.denoise_model(x_double, t_double, y=labels_double)
            
            # 拆解并应用 CFG 外推公式
            pred_cond, pred_uncond = pred_double.chunk(2, dim=0)
            w_t = self.cfg_scale
            predicted_noise = pred_uncond + w_t * (pred_cond - pred_uncond)
        else:
            # 不使用 CFG 的直接预测
            predicted_noise = self.denoise_model(x, t, y=labels)
            
        return predicted_noise



    @torch.no_grad()
    def p_sample(self, x, t, labels, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # 替换原有的预测调用，传入 labels
        predicted_noise = self._predict_noise(x, t, labels)
        
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise



    @torch.no_grad()
    def p_sample_loop(self, shape, labels):
        device = next(self.denoise_model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []
        for t in tqdm(reversed(range(0, self.timesteps)), desc='DDPM sampling loop', total=self.timesteps):
            batch_t = torch.full((b,), t, device=device, dtype=torch.long)
            # 传入 labels
            img = self.p_sample(img, batch_t, labels, t)
            imgs.append(img.cpu().numpy())
        return imgs



    @torch.no_grad()
    def ddim_p_sample(self, x_t, t, t_prev, labels, eta=0.0):
        # 传入 labels 获取预测噪声
        eps = self._predict_noise(x_t, t, labels)

        alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
        if t_prev[0] < 0:
            alpha_prev = torch.ones_like(alpha_t)
        else:
            alpha_prev = extract(self.alphas_cumprod, t_prev, x_t.shape)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

        pred_x0 = (x_t - sqrt_one_minus_alpha_t * eps) / (sqrt_alpha_t + 1e-8)

        eps_ratio = (alpha_t / (alpha_prev + 1e-8)).clamp(max=1.0)
        sigma = eta * torch.sqrt(((1.0 - alpha_prev) / (1.0 - alpha_t + 1e-8)) * (1.0 - eps_ratio))

        coef = (1.0 - alpha_prev - sigma ** 2).clamp(min=0.0)
        dir_term = torch.sqrt(coef) * eps

        noise = torch.randn_like(x_t) if eta > 0 else 0.0
        x_prev = sqrt_alpha_prev * pred_x0 + dir_term + sigma * noise
        return x_prev



    @torch.no_grad()
    def ddim_p_sample_loop(self, shape, labels, ddim_steps=None, eta=0.0):
        device = next(self.denoise_model.parameters()).device
        b = shape[0]
        T = self.timesteps
        img = torch.randn(shape, device=device)
        
        if ddim_steps is None:
            ddim_steps = T
        else:
            ddim_steps = min(T, ddim_steps)
        times = np.linspace(0, T - 1, ddim_steps, dtype=int).tolist()

        imgs = []
        for i in tqdm(reversed(range(0, len(times))), desc='DDIM sampling loop', total=len(times)):
            t_cur = times[i]
            t_prev = times[i - 1] if i > 0 else -1
            batch_t = torch.full((b,), t_cur, device=device, dtype=torch.long)
            batch_t_prev = torch.full((b,), t_prev, device=device, dtype=torch.long)
            # 传入 labels
            img = self.ddim_p_sample(img, batch_t, batch_t_prev, labels, eta=eta)
            imgs.append(img)

        return imgs



    @torch.no_grad()
    def sample(self, labels, bs=None):
        bs = bs if bs else self.bs
        shape = (bs, self.channel, self.img_size[0], self.img_size[1])
        
        # DDPM 采样 (根据需要取消注释):
        # denoise_img_series = self.p_sample_loop(shape=shape, labels=labels)
        
        # DDIM 采样:
        denoise_img_series = self.ddim_p_sample_loop(shape=shape, labels=labels, ddim_steps=50, eta=0.0)
        return denoise_img_series