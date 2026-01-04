import torch
import torch.nn as nn
import torch.nn.functional as F
from heltonx.utils.register import MODELS

class GatedActivation(nn.Module):
    """门控激活单元: y = tanh(a) * sigmoid(b)"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: [B, 2*C, H, W]
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)

class MaskedConv2d(nn.Conv2d):
    """
    统一的 Masked Convolution，支持 Vertical 和 Horizontal 两种模式
    """
    def __init__(self, mask_type, location, *args, **kwargs):
        """
        Args:
            mask_type: 'A' (不包含中心) 或 'B' (包含中心)
            location: 'vertical' (垂直流) 或 'horizontal' (水平流)
        """
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kH, kW = self.weight.size()
        cC, cR = kH // 2, kW // 2

        self.mask.fill_(1.0)

        if location == 'vertical':
            # 垂直流：中心行及以下全部 Mask 掉 (保证只看上面)
            # 无论是 Mask A 还是 B，垂直流都不能看当前行，因为它要喂给水平流
            # 如果垂直流看了当前行，水平流一旦融合，就等于泄露了
            self.mask[:, :, cC:, :] = 0.0
        
        elif location == 'horizontal':
            # 水平流：只看当前行 (1xK 卷积)
            # 这里的输入假设已经是 1xK 或者通过 mask 变成了 1xK 效果
            # 我们只 Mask 右侧
            self.mask[:, :, :, cR+1:] = 0.0
            
            if mask_type == 'A':
                # Mask A: 不能看自己 (中心点置0)
                self.mask[:, :, :, cR] = 0.0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class GatedBlock(nn.Module):
    def __init__(self, in_dim, dim, kernel_size=3, mask_type='B', residual=True):
        super().__init__()
        self.residual = residual
        self.mask_type = mask_type
        
        # 1. 垂直流: NxN 卷积，Mask 掉中心及下方
        # padding 保证输出尺寸不变
        self.v_conv = MaskedConv2d(mask_type='A', location='vertical', 
                                   in_channels=in_dim, out_channels=2*dim, 
                                   kernel_size=kernel_size, padding=kernel_size//2)
        
        # 2. 垂直 -> 水平 投影 (1x1)
        self.v_to_h_proj = nn.Conv2d(2*dim, 2*dim, 1)
        
        # 3. 水平流: 1xN 卷积
        # 注意：水平流卷积核的高度必须是 1，否则会再次引入垂直维度的泄露
        self.h_conv = MaskedConv2d(mask_type=mask_type, location='horizontal', 
                                   in_channels=in_dim, out_channels=2*dim, 
                                   kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        
        self.h_out_proj = nn.Conv2d(dim, dim, 1)
        self.gate = GatedActivation()

    def forward(self, v_input, h_input):
        # --- Vertical Stream ---
        # 垂直流卷积 (已经 Mask 掉了当前行及下方)
        v_feat = self.v_conv(v_input)
        v_out = self.gate(v_feat)
        
        # --- Horizontal Stream ---
        # 水平流卷积 (1xK, Mask 掉了右侧)
        h_feat = self.h_conv(h_input)
        
        # 融合: 水平流 + 垂直流信息
        # 垂直流提供了当前行之上的 Context
        v_feat_for_h = self.v_to_h_proj(v_feat) # 使用 v_feat (未激活) 还是 v_out (激活后)? 原文通常是未激活的相加
        
        # Gated Activation
        h_out = self.gate(h_feat + v_feat_for_h)
        h_out = self.h_out_proj(h_out)
        
        # Residual Connection (仅在 Horizontal 上，且仅 Mask B)
        if self.residual and self.mask_type == 'B':
            h_out = h_out + h_input
            
        # 垂直流通常不加 Residual，或者有专门的结构，这里简化处理只返回 v_out 作为下一层输入
        # 注意: 严格的 GatedPixelCNN 中垂直流也有 Residual (v_out + v_input)，但这要求通道数匹配
        # 这里如果输入输出 dim 一致，可以加
        if self.residual and self.mask_type == 'B' and v_input.shape == v_out.shape:
             v_out = v_out + v_input
             
        return v_out, h_out

@MODELS.register
class GatedPixelCNN(nn.Module):
    def __init__(self, num_embeddings, dim=128, n_layers=10, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(num_embeddings, dim)
        
        # Layer 1: Mask A (Blind)
        self.first_block = GatedBlock(dim, dim, kernel_size, mask_type='A', residual=False)
        
        # Layers 2..N: Mask B (See self history)
        self.layers = nn.ModuleList([
            GatedBlock(dim, dim, kernel_size, mask_type='B', residual=True)
            for _ in range(n_layers)
        ])
        
        self.out_conv = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, 1024, 1),
            nn.ReLU(True),
            nn.Conv2d(1024, num_embeddings, 1)
        )
        
        self.init_weights()

    def forward(self, x):
        # x: [B, H, W] Indices
        x = self.embedding(x).permute(0, 3, 1, 2)
        
        v_stack = x
        h_stack = x
        
        v_stack, h_stack = self.first_block(v_stack, h_stack)
        
        for layer in self.layers:
            v_stack, h_stack = layer(v_stack, h_stack)
            
        return self.out_conv(h_stack)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight) # 使用 Kaiming 初始化
                if m.bias is not None: nn.init.zeros_(m.bias)