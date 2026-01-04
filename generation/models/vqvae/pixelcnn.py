import torch
import torch.nn as nn
import torch.nn.functional as F
from heltonx.utils.register import MODELS



class MaskedConv2d(nn.Conv2d):
    """Masked Convolution (核心组件)
    """
    def __init__(self, mask_type, *args, **kwargs):
        """
        Args:
            mask_type: 'A' (只能看前面，不能看自己) 或 'B' (能看前面，也能看自己)
            Mask A 用于第一层，Mask B 用于后续层
        """
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kH, kW = self.weight.size()
        # 定义中心点
        cC, cR = kH // 2, kW // 2
        # 1. 将中心行下方的所有行置 0
        self.mask[:, :, cC+1:, :] = 0.0
        # 2. 将中心行右侧的所有列置 0
        self.mask[:, :, cC, cR+1:] = 0.0
        # 3. 处理中心点 (Mask A 不能看自己，Mask B 可以)
        if mask_type == 'A':
            self.mask[:, :, cC, cR] = 0.0

    def forward(self, x):
        # 在卷积前将权重乘以 mask
        self.weight.data *= self.mask
        return super().forward(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            MaskedConv2d('B', dim, dim // 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            MaskedConv2d('B', dim // 2, dim, 1)
        )
    def forward(self, x):
        return x + self.block(x)





@MODELS.register
class PixelCNN(nn.Module):
    """PixelCNN 模型主体
    """
    def __init__(self, num_embeddings, dim=128, n_layers=10):
        super().__init__()
        self.dim = dim
        self.num_embeddings = num_embeddings
        # 1. Embedding 层: 将离散索引变成向量
        self.embedding = nn.Embedding(num_embeddings, dim)
        # 2. 第一层使用 Mask A (不能看见自己)
        self.layers = nn.ModuleList([
            MaskedConv2d('A', dim, dim, kernel_size=7, padding=3)
        ])
        # 3. 后续堆叠 ResBlock (使用 Mask B)
        for _ in range(n_layers):
            self.layers.append(ResBlock(dim))
        # 4. 输出层: 映射回 num_embeddings 大小，用于分类
        self.out_conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            MaskedConv2d('B', dim, dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            MaskedConv2d('B', dim, num_embeddings, 1) 
        )

    def forward(self, x):
        # x: [B, H, W] (Indices)
        x = self.embedding(x).permute(0, 3, 1, 2) # -> [B, dim, H, W]
        for layer in self.layers:
            x = layer(x)
        # [B, num_embeddings, H, W]
        return self.out_conv(x)


    def init_weights(self):
        # 简单的初始化逻辑，PixelCNN通常对初始化不敏感，这里略过或使用默认
        pass

