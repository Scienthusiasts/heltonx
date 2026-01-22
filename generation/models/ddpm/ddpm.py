import torch
from torch import nn
import torch.nn.functional as F
from generation.utils.utils import *
from generation.models.ddpm.blocks import *
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
        # 方差生成
        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)
        self.timesteps = timesteps
        # self.betas是从beta_start到beta_end的数列(用于控制噪声与图像的比例)
        self.betas = variance_schedule_func(timesteps)
        # define alphas
        self.alphas = 1. - self.betas
        # self.alphas_cumprod[i]是self.alphas[0]到self.alphas[i]的累乘
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 去掉self.alphas_cumprod[-1], 并在最左侧填充一个1.
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 这里用的不是简化后的方差而是算出来的
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)


    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程(原图->加噪图(根据t加不同程度的噪声))
            Args:
                x_start: 原始图像
                t:       时间步
                noise:   pure 高斯噪声
            Returns:
                noise_x: 原始图像与pure高斯噪声混合后的图像(根据t决定噪声占比, t越大噪声越多)
        """
        # forward diffusion (using the nice property)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        noise_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noise_x


    def compute_loss(self, x_start, t, noise=None, loss_type="l1"):
        # 生成pure高斯噪声
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 预测 x_start -> x_noisy过程中所加的原始pure高斯噪声(不包含加权系数)
        predicted_noise = self.denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.denoise_model.parameters()).device

        b = shape[0]
        # 从pure高斯噪声开始逆扩散
        img = torch.randn(shape, device=device)
        imgs = []
        # 需经过self.timesteps步去噪
        for t in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            # 一步去噪
            batch_t = torch.full((b,), t, device=device, dtype=torch.long)
            img = self.p_sample(img, batch_t, t)
            imgs.append(img.cpu().numpy())
        return imgs

















    @torch.no_grad()
    def ddim_p_sample(self, x_t, t, t_prev, eta=0.0):
        """
        DDIM single step from x_t to x_{t_prev}.

        Args:
            x_t: current noisy image tensor at timestep t (shape [B,C,H,W])
            t: tensor of current timesteps (shape [B], dtype long)
            t_prev: tensor of previous (next smaller) timesteps (shape [B], dtype long)
            eta: float, controls stochasticity (0.0 = deterministic DDIM)
        Returns:
            x_prev: tensor for timestep t_prev
        """
        # predicted noise eps_theta(x_t, t)
        eps = self.denoise_model(x_t, t)

        # alpha_cumprod at t and t_prev
        alpha_t = extract(self.alphas_cumprod, t, x_t.shape)            # \bar{alpha}_t
        alpha_prev = extract(self.alphas_cumprod, t_prev, x_t.shape)    # \bar{alpha}_{t_prev}

        # sqrt terms
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

        # predict x0 from x_t and eps
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * eps) / (sqrt_alpha_t + 1e-8)

        # compute sigma following DDIM paper:
        # sigma = eta * sqrt( (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t/alpha_prev) )
        # note: guard against division by zero if alpha_prev == 0
        # both alpha_t and alpha_prev are tensors
        eps_ratio = (alpha_t / (alpha_prev + 1e-8)).clamp(max=1.0)  # alpha_t / alpha_prev
        sigma = eta * torch.sqrt(((1.0 - alpha_prev) / (1.0 - alpha_t + 1e-8)) * (1.0 - eps_ratio))

        # the "direction pointing to x_t" term
        # sqrt(1 - alpha_prev - sigma^2) * eps
        # ensure numerical stability inside sqrt
        coef = (1.0 - alpha_prev - sigma ** 2).clamp(min=0.0)
        dir_term = torch.sqrt(coef) * eps

        # noise (only when eta > 0)
        noise = torch.randn_like(x_t) if eta > 0 else 0.0

        # final composition
        x_prev = sqrt_alpha_prev * pred_x0 + dir_term + sigma * noise
        return x_prev


    @torch.no_grad()
    def ddim_p_sample_loop(self, shape, ddim_steps=None, eta=0.0):
        """
        DDIM sampling loop.

        Args:
            shape: tuple, (B, C, H, W)
            ddim_steps: int or None. If int < self.timesteps, use a subsampled timestep sequence with that many steps.
                        If None, uses self.timesteps (full).
            eta: float, DDIM eta parameter controlling stochasticity.
        Returns:
            list of numpy arrays: intermediate images (CPU numpy) at each sampling step (in time order).
        """
        device = next(self.denoise_model.parameters()).device
        b = shape[0]
        T = self.timesteps

        # initial noise x_T
        # 1.采用完全不同的噪声
        img = torch.randn(shape, device=device)
        # 2.采用完全相同的噪声
        # img = torch.randn(shape[1:], device=device)
        # img = img.unsqueeze(0).repeat(shape[0], 1, 1, 1)
        # build timestep sequence (ascending): [0, ..., T-1]
        if ddim_steps is None:
            ddim_steps = T
        else:
            ddim_steps = min(T, ddim_steps)
        # create ddim_steps linearly spaced timesteps including 0 and T-1
        times = np.linspace(0, T - 1, ddim_steps, dtype=int).tolist()

        # we'll iterate from largest t down to 0
        imgs = []
        # times is ascending: [0,...,T-1], we want to iterate reverse
        for i in tqdm(reversed(range(0, len(times))), desc='sampling loop time step', total=len(times)):
            t_cur = times[i]
            # next (previous in time) timestep: if idx>0 use times[idx-1], else 0
            t_prev = times[i - 1] if i > 0 else 0

            batch_t = torch.full((b,), t_cur, device=device, dtype=torch.long)
            batch_t_prev = torch.full((b,), t_prev, device=device, dtype=torch.long)

            img = self.ddim_p_sample(img, batch_t, batch_t_prev, eta=eta)
            imgs.append(img.cpu().numpy())

        return imgs


















    @torch.no_grad()
    def sample(self, bs=None, channels=3):
        bs = bs if bs else self.bs
        # DDPM 采样:
        # denoise_img_series = self.p_sample_loop(shape=(bs, channels, self.img_size[0], self.img_size[1]))
        # DDIM 采样:
        # eta=0 完全确定性 DDIM(通常用于快速、稳定采样) / eta>0 引入部分随机性, 最后会退回到 DDPM 的随机性(eta=1 时近似随机)
        denoise_img_series = self.ddim_p_sample_loop(shape=(bs, channels, self.img_size[0], self.img_size[1]), ddim_steps=50, eta=0.0)
        return denoise_img_series




    def forward(self, batch_data=None, return_loss=True, bs=None):
        """前向+计算损失
        """
        if return_loss:
            y = batch_data[0]
            bs = y.shape[0]
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, self.timesteps, (bs,), device=y.device).long()
            loss = self.compute_loss(x_start=y, t=t, loss_type=self.loss_type)
            # 字典形式
            losses = dict(
                gen_loss = loss
            )
            return losses
        else:
            # [-1]是只取最后一个time_step
            return self.sample(bs=bs)[-1]










