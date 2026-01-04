import torch
import torch.nn as nn
import torch.nn.functional as F
from heltonx.utils.register import MODELS






@MODELS.register
class VQVAE_PixelCNN(nn.Module):
    """训练与推理包装器 PixelCNN用于生成离散编码, vqvae负责将离散
    """
    def __init__(self, vqvae_model:nn.Module, pixelcnn_model:nn.Module):
        """
        Args:
            vqvae_model: 训练好的 VQ-VAE 实例
            num_embeddings: 码本大小 (需要和 VQ-VAE 一致)
        """
        super().__init__()
        self.vqvae = vqvae_model
        # 冻结 VQ-VAE 参数，不参与更新
        for param in self.vqvae.parameters():
            param.requires_grad = False
        self.vqvae.eval() 
        self.pixelcnn = pixelcnn_model
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, batch_datas=None, return_loss=True, bs=None, sample_shape=(8,8)):
        """
        Args:
            batch_datas: [B, 3, H, W] 原始图片
            sample_shape: (H_feat, W_feat) 特征图的空间尺寸，生成时需要指定
        """
        if return_loss:
            x = batch_datas[0] if isinstance(batch_datas, (list, tuple)) else batch_datas
            
            with torch.no_grad():
                # 1. 获取 VQ-VAE 的编码索引 (GT)
                # VQ-VAE encoder -> pre_quant -> indices
                z = self.vqvae.encoder(x)
                z = self.vqvae.proj_conv(z)
                # indices: [B*H*W, 1]
                _, _, indices = self.vqvae.vq_module(z)
                
                # Reshape indices to [B, H, W]
                b, _, h, w = z.shape
                indices = indices.view(b, h, w)
            
            # 2. PixelCNN 前向传播
            # 输入 indices, 预测 logits
            logits = self.pixelcnn(indices) # [B, num_embeddings, H, W]
            # 3. 计算 Cross Entropy Loss
            loss = self.criterion(logits, indices)
            return {'pixelcnn_loss': loss}
        else:
            # 推理模式
            with torch.no_grad():
                return self.sample(bs, sample_shape)


    def sample(self, bs, shape):
        """
        自回归生成 (Autoregressive Sampling)
        过程很慢，因为要一个一个像素生成 (Row by Row, Col by Col)
        Args:
            shape: (h, w) 特征图尺寸
        """
        device = next(self.pixelcnn.parameters()).device
        h, w = shape
        
        # 1. 初始化空白画布 (全0或随机都可以，反正会被覆盖，只要shape对就行)
        # 注意：PixelCNN 会看着前面的点预测当前点，所以必须填入真实生成的点
        indices = torch.zeros((bs, h, w), dtype=torch.long).to(device)
        
        # 2. 双重循环逐点生成
        for row in range(h):
            for col in range(w):
                # 传入当前的 indices 画布
                # PixelCNN 会并行计算所有位置的 logits，但我们只取当前 (row, col) 的结果
                logits = self.pixelcnn(indices) # [B, K, H, W]
                # 获取当前位置的概率分布
                probs = F.softmax(logits[:, :, row, col], dim=-1) # [B, K]
                # 采样 (Multinomial sampling)
                # 也可以用 argmax 贪婪采样，但多样性会变差
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # [B]
                # 填入画布
                indices[:, row, col] = next_token
        
        # 3. 循环结束，indices 填满，送入 VQ-VAE Decoder
        # indices [B, H, W] -> [B, D, H, W]
        z_q = self.vqvae.vq_module.embedding(indices).permute(0, 3, 1, 2).contiguous()
        z_decoder_input = self.vqvae.post_quant_conv(z_q)
        # 解码生成最终图像
        fake_imgs = self.vqvae.decoder(z_decoder_input)
        return fake_imgs.detach().float().cpu().numpy()
