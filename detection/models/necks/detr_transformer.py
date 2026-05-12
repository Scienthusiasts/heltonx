import torch
import torch.nn as nn
import copy
from heltonx.utils.register import MODELS
from heltonx.utils.utils import init_weights
from detection.utils.detr_utils import PositionEmbeddingSine2D


class DETREncoderLayer(nn.Module):
    """DETR Encoder 层

    与标准 TransformerEncoderLayer 的区别：位置编码只加到 Q 和 K 上，V 不加。
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src, pos=None, src_key_padding_mask=None):
        """前向传播

        Args:
            src:   [B, H*W, d_model] 输入特征
            pos:   [B, H*W, d_model] 位置编码（只加到 Q/K）
            src_key_padding_mask: [B, H*W] padding mask
        """
        # Q = K = src + pos, V = src（位置编码只加到 Q/K）
        q = k = src + pos if pos is not None else src
        src2, _ = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout3(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DETRDecoderLayer(nn.Module):
    """DETR Decoder 层

    与标准 TransformerDecoderLayer 的区别：
    - 自注意力中 query_pos 只加到 Q/K，V 不加
    - 交叉注意力中 query_pos 加到 query，pos 加到 key，value 不加
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt, memory, pos=None, query_pos=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """前向传播

        Args:
            tgt:   [B, num_queries, d_model] decoder 输入（内容）
            memory: [B, H*W, d_model] encoder 输出
            pos:   [B, H*W, d_model] encoder 位置编码（加到 cross-attn 的 key）
            query_pos: [B, num_queries, d_model] query 位置编码（加到 self-attn 和 cross-attn 的 query）
        """
        # 自注意力: Q = K = tgt + query_pos, V = tgt
        q = k = tgt + query_pos if query_pos is not None else tgt
        tgt2, _ = self.self_attn(q, k, value=tgt, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 交叉注意力: Q = tgt + query_pos, K = memory + pos, V = memory
        q = tgt + query_pos if query_pos is not None else tgt
        k = memory + pos if pos is not None else memory
        tgt2, _ = self.multihead_attn(q, k, value=memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout4(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DETRDecoder(nn.Module):
    """DETR Decoder

    收集每层中间输出并经过 LayerNorm（用于 Auxiliary Loss）。
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, pos=None, query_pos=None, memory_key_padding_mask=None):
        """前向传播

        Returns:
            hs_all: [num_decoder_layers, B, num_queries, d_model] 每层经 LayerNorm 的输出
        """
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos,
                           memory_key_padding_mask=memory_key_padding_mask)
            if self.norm is not None:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            # 最终输出也经过 norm（与中间层一致）
            intermediate.pop()
            intermediate.append(self.norm(output))
        return torch.stack(intermediate)


@MODELS.register
class DETRTransformer(nn.Module):
    """DETR Transformer 模块 (占 fpn 槽位)

    包含: 投影层 + 2D 正弦位置编码 + Transformer Encoder + Transformer Decoder + Object Queries

    与官方实现一致：
    - 位置编码只加到 Q/K，V 不加
    - Decoder 输入 tgt 为零，query_embed 作为位置编码
    - 每层 decoder 输出经过 LayerNorm（用于 Auxiliary Loss）

    Args:
        in_channels (int):        backbone 输出通道数 (如 ResNet50 为 2048)
        hidden_dim (int):         Transformer 隐层维度
        num_heads (int):          多头注意力头数
        num_encoder_layers (int): Encoder 层数
        num_decoder_layers (int): Decoder 层数
        num_queries (int):        Object Query 数量
        dim_feedforward (int):    FFN 中间层维度
        dropout (float):          Dropout 率
    """

    def __init__(self, in_channels=2048, hidden_dim=256, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 num_queries=100, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # 1x1 投影: backbone 通道 -> hidden_dim
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # 2D 正弦位置编码
        self.position_encoding = PositionEmbeddingSine2D(num_feats=hidden_dim // 2)

        # Transformer Encoder（自定义层，pos 只加到 Q/K）
        encoder_layer = DETREncoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)])

        # Transformer Decoder（自定义层，query_pos 分离）
        decoder_layer = DETRDecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = DETRDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # Object Queries (可学习，仅作为位置编码)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 权重初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 投影层用更小的初始化
        for m in [self.input_proj]:
            init_weights(m, 'normal', 0, 0.01)

    def forward(self, features):
        """前向传播

        Args:
            features (List[Tensor]): backbone 输出特征列表，取最后一层 [B, C, H, W]

        Returns:
            hs_all:   [num_decoder_layers, B, num_queries, hidden_dim] 所有 Decoder 层输出（经 LayerNorm）
            init_ref: [B, num_queries, 4] 参考点 (初始为 0.5)
        """
        # 取 backbone 最后一层特征
        if isinstance(features, (list, tuple)):
            src = features[-1]  # [B, C, H, W]
        else:
            src = features

        bs, _, h, w = src.shape

        # 投影到 hidden_dim
        src = self.input_proj(src)  # [B, hidden_dim, H, W]

        # 位置编码
        pos = self.position_encoding(src)  # [B, hidden_dim, H, W]

        # 展平空间维度: [B, hidden_dim, H, W] -> [B, H*W, hidden_dim]
        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_flat = pos.flatten(2).permute(0, 2, 1)

        # Encoder: pos 通过参数传入，只加到 Q/K
        memory = src_flat
        for layer in self.encoder:
            memory = layer(memory, pos=pos_flat, src_key_padding_mask=None)

        # Object Queries (仅作为位置编码)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # [B, num_queries, hidden_dim]

        # 参考点 (图像中心点为参考, Deformable DETR中使用，标准DETR不使用)
        init_ref = torch.zeros(bs, self.num_queries, 4, device=src.device, dtype=src.dtype)
        init_ref[:, :, :2] = 0.5  # 中心点

        # Decoder: tgt 为零（内容从零开始学习），
        tgt = torch.zeros_like(query_embed)
        hs_all = self.decoder(tgt, memory, pos=pos_flat, query_pos=query_embed,
                              memory_key_padding_mask=None)

        return hs_all, init_ref
