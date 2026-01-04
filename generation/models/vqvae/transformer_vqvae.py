import torch
import torch.nn as nn
import torch.nn.functional as F
from heltonx.utils.register import MODELS






@MODELS.register
class VQVAE_Transformer(nn.Module):
    """训练与推理包装器 PixelCNN用于生成离散编码, vqvae负责将离散
    """
    def __init__(self, vqvae_model:nn.Module, transformer_model:nn.Module):
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
        self.transformer = transformer_model
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
            logits = self.transformer(indices) # [B, num_embeddings, H, W]
            # 3. 计算 Cross Entropy Loss
            loss = self.criterion(logits, indices)
            return {'transformer_loss': loss}
        else:
            # 推理模式
            with torch.no_grad():
                return self.sample(bs, sample_shape)


    def sample(self, bs, shape):
        """Transformer Autoregressive Sampling
        Args:
            shape: (h, w) 特征图尺寸
        """
        device = next(self.vqvae.parameters()).device
        h, w = shape
        seq_len = h * w
        
        # 1. 初始输入仅为 SOS Token
        # [B, 1, Dim]
        sos_token = self.transformer.sos_token.expand(bs, -1, -1)
        
        # Current input sequence indices (starts empty)
        # 我们这里不直接存 indices，而是维护 input embeddings 比较方便? 
        # 不，还是维护 indices 比较直观，因为需要 lookup embedding
        
        # Start with just SOS embedding logic handled inside loop or explicit
        # 为了复用 forward 中的逻辑，我们一点点构建 indices 序列是不行的，
        # 因为 forward 是一次性处理并移位。
        # 我们需要写一个专门的 step inference。
        
        generated_indices = torch.zeros((bs, 0), dtype=torch.long, device=device)
        
        # 缓存 (KV Cache) 可以加速，但为了代码简单，这里演示这一种 "笨办法"：
        # 每次都把生成的全序列扔进去，只取最后一个 Logit。
        
        with torch.no_grad():
            for i in range(seq_len):
                # 1. 准备当前输入: [B, current_len]
                # 第一次循环 generated_indices 为空
                
                # 我们需要手动模拟 forward 的前半部分
                if i == 0:
                    x_input = sos_token # [B, 1, Dim]
                    pos = self.transformer.pos_emb[:, :1, :]
                else:
                    # Look up embedding for known indices
                    # [B, i] -> [B, i, Dim]
                    tok_emb = self.transformer.tok_emb(generated_indices)
                    x_input = torch.cat([sos_token, tok_emb], dim=1) # [B, i+1, Dim]
                    pos = self.transformer.pos_emb[:, :i+1, :]
                
                # Add Pos Emb
                x_input = x_input + pos
                
                # Mask (虽然这里不是必须的，因为我们没有未来的token，但为了保持分布一致最好加上)
                # 其实 inference 时不需要 causal mask，因为我们只给了过去的数据
                
                # Transformer Pass
                out = self.transformer.transformer(x_input) # [B, i+1, Dim]
                
                # 取最后一个时间步的输出，预测下一个词
                last_token_feat = out[:, -1, :] # [B, Dim]
                last_token_feat = self.transformer.ln_f(last_token_feat)
                logits = self.transformer.head(last_token_feat) # [B, Num_Embeddings]
                
                # 采样
                probs = F.softmax(logits, dim=-1)
                # Top-k sampling or simple multinomial(生成图片多样性的核心)
                next_token = torch.multinomial(probs, num_samples=1) # [B, 1]
                
                # Append result
                generated_indices = torch.cat([generated_indices, next_token], dim=1)
        
        # End loop
        # generated_indices: [B, Seq_Len] -> Reshape to [B, H, W]
        indices = generated_indices.view(bs, h, w)
        
        # VQ-VAE Decode
        z_q = self.vqvae.vq_module.embedding(indices).permute(0, 3, 1, 2).contiguous()
        z_decoder_input = self.vqvae.post_quant_conv(z_q)
        fake_imgs = self.vqvae.decoder(z_decoder_input)
        
        return fake_imgs.detach().float().cpu().numpy()