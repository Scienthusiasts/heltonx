import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from heltonx.utils.register import MODELS



@MODELS.register
class ImageTransformer(nn.Module):
    def __init__(self, num_embeddings, dim=256, n_layers=8, n_head=8, max_seq_len=1024, dropout=0.1):
        """基于 Transformer 的vqvae自回归先验模型 (ImageGPT style)
            Args:
                num_embeddings: Codebook 大小 (Vocabulary Size)
                dim: Transformer 的隐藏层维度 (d_model)
                n_layers: Transformer Block 层数
                n_head: 多头注意力头数 (dim 必须能被 n_head 整除)
                max_seq_len: 最大序列长度 (H * W)。例如 8x8=64, 16x16=256, 32x32=1024
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 1. Token Embedding
        self.tok_emb = nn.Embedding(num_embeddings, dim)
        # 2. Positional Embedding (Learnable)
        # 长度为 max_seq_len + 1 (因为我们要加一个 SOS token)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len + 1, dim))
        # 3. SOS Token (Start of Sequence) - 作为一个可学习的向量
        self.sos_token = nn.Parameter(torch.zeros(1, 1, dim))
        # 4. Transformer Decoder
        # 我们使用 TransformerEncoder 结构，且加上 Causal Mask，这在 GPT 中很常见
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=n_head, 
            dim_feedforward=dim * 4, 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-Norm 结构，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 5. Layer Norm & Output Head
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_embeddings, bias=False)
        # 初始化权重
        self.init_weights()



    def forward(self, x):
        """
        Args:
            x: [B, H, W] (Indices)
        Returns:
            logits: [B, num_embeddings, H, W]
        """
        b, h, w = x.shape
        seq_len = h * w
        
        # 1. Flatten Input: [B, H, W] -> [B, Seq_Len]
        x_flat = x.view(b, -1)
        # 2. Embedding Look up: [B, Seq_Len, Dim]
        token_embeddings = self.tok_emb(x_flat)
        # 3. Prepend SOS Token
        # 为了预测第一个像素 x[0]，我们需要输入 SOS
        # Input 序列变为: [SOS, x[0], x[1], ..., x[N-1]]
        # 对应的 Target 是: [x[0], x[1], x[2], ..., x[N]]
        sos_tokens = self.sos_token.expand(b, -1, -1) # [B, 1, Dim]
        x_input = torch.cat([sos_tokens, token_embeddings], dim=1) # [B, Seq_Len + 1, Dim]
        # 4. Add Positional Embedding
        # 取前 seq_len + 1 个位置编码
        x_input = x_input + self.pos_emb[:, :seq_len + 1, :]
        # 5. Generate Causal Mask (上三角 Mask)
        # 确保位置 i 只能看到 0...i，不能看到 i+1
        # mask shape: [Seq_Len+1, Seq_Len+1]
        mask = torch.triu(torch.full((seq_len + 1, seq_len + 1), float('-inf'), device=x.device), diagonal=1)
        # 6. Transformer Forward
        out = self.transformer(x_input, mask=mask) # [B, Seq_Len + 1, Dim]
        # 7. Post-process
        out = self.ln_f(out)
        logits = self.head(out) # [B, Seq_Len + 1, Num_Embeddings]
        # 8. Align Output
        # 我们现在的 logits 是对输入 [SOS, x[0]...x[N-1]] 的预测 [x[0]...x[N]]
        # logits[N] (对应x[N-1]) 预测 下一张图的开始(无意义), 所以我们去掉最后一个输出，保留前 N 个
        logits = logits[:, :-1, :] # [B, Seq_Len, Num_Embeddings]
        # 9. Reshape back to image format for Loss calculation
        # [B, H*W, K] -> [B, K, H*W] -> [B, K, H, W]
        logits = logits.permute(0, 2, 1).view(b, self.num_embeddings, h, w)
        return logits



    def init_weights(self):
        # 这里的初始化对于 Transformer 收敛非常重要
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.sos_token, std=0.02)
        # Apply special init to Linear and Embedding
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)