import torch
import torch.nn as nn
from heltonx.utils.register import MODELS
from heltonx.utils.utils import init_weights
from generation.models.blocks import *




@MODELS.register
class LightBERT(nn.Module):
    def __init__(self, emb_dim, n_layers, heads=4, max_len=512, dropout=0.1):
        """轻量级Transformer Encoder用于聚合文本特征
            Args:
                emb_dim: embedding维度
                n_layers: transformer层数
                heads: 多头注意力的头数
                max_len: 支持的最大序列长度
        """
        super().__init__()
        self.emb_dim = emb_dim
        # 1. Learnable Classification Token (类似于BERT的[CLS]) [1, 1, emb_dim]
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # 2. 可学习位置编码 [1, max_len + 1, emb_dim] (+1 是为了留给 cls_token)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, emb_dim))
        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=heads, 
            dim_feedforward=emb_dim * 4, 
            dropout=dropout,
            # batch_first=True 使得输入维度为 [bs, seq_len, emb_dim]
            batch_first=True,
            # norm_first=True (Pre-LN) 通常训练更稳定
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 初始化参数
        self._init_weights()


    def forward(self, x):
        """
            Args:
                x: 文本embeddings [bs, seq_len, emb_dim]
            Returns:
                pooled_output: [bs, emb_dim]
        """
        bs, seq_len, _ = x.shape
        # 1. 扩展 cls_token 到 batch 大小: [bs, 1, emb_dim]
        cls_tokens = self.cls_token.repeat(bs, 1, 1)
        # 2. cls_token拼接到输入序列前面: [bs, seq_len + 1, emb_dim]
        x = torch.cat((cls_tokens, x), dim=1)
        # 3. 加上位置编码 (截取当前序列长度对应的位置编码)
        # pos_embedding[:, :(seq_len + 1)] 广播加到 x 上
        x = x + self.pos_embedding[:, :(seq_len + 1)]
        # 4. Transformer Forward (无因果掩码，双向注意力)
        x = self.transformer(x)
        # 5. 只取cls_token作为整体特征 [bs, seq_len + 1, emb_dim] -> [bs, emb_dim]
        return x[:, 0]


    def _init_weights(self):
        # 对token和pos embedding进行正态分布初始化
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        # Transformer内部通常有自己的初始化，但我们可以额外处理Linear层
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)