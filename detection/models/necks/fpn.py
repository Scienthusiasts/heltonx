import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch.distributed as dist
from heltonx.utils.utils import init_weights
# 注册机制
from heltonx.utils.register import MODELS





@MODELS.register
class FPN(nn.Module):
    def __init__(self, in_channels, out_channel=256, num_extra_levels=2):
        """通用 FPN 模块(只对跨尺度特征融合, 无非线性变换)
            Args:
                in_channels (List[int]): 输入每层特征的通道数, 例如 [256, 512, 1024, 2048]
                out_channel (int):       每层输出通道数
                num_extra_levels (int):  额外下采样层数量, 例如 2 -> P6, P7
                init_fn (Callable):      权重初始化函数，可选
        """
        super().__init__()
        self.num_levels = len(in_channels)
        self.out_channels = out_channel
        self.num_extra_levels = num_extra_levels

        # 1x1卷积将各层投影到相同通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channel, kernel_size=1)
            for c in in_channels
        ])
        # 3x3卷积用于平滑融合后的特征
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
            for _ in in_channels
        ])
        # 额外的下采样层 (例如 P6, P7)
        self.extra_convs = nn.ModuleList([
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1)
            for _ in range(num_extra_levels)
        ])
        # 权重初始化
        for m in self.modules():
            init_weights(m, 'normal', 0, 0.01)


    def _upsample_add(self, higher, lower):
        """上采样 + 相加
        """
        _, _, H, W = lower.shape
        # bilinear 可能会出现对齐误差(半像素偏移)
        return F.interpolate(higher, size=(H, W), mode='nearest') + lower


    def forward(self, features):
        """
        Args:
            features (List[Tensor]): backbone输出的多层特征图，低->高，例如 [C3, C4, C5]
        Returns:
            List[Tensor]: FPN输出特征图 [P3, P4, P5, (P6, P7...)]
        """
        assert len(features) == self.num_levels, \
            f"Expect {self.num_levels} input features, got {len(features)}"

        # lateral conv 投影(将各层投影到相同通道数)
        lateral_feats = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # 从最高层向下融合
        results = [None] * self.num_levels
        results[-1] = lateral_feats[-1]
        for i in range(self.num_levels - 2, -1, -1):
            results[i] = self._upsample_add(results[i + 1], lateral_feats[i])

        # 平滑卷积
        results = [conv(f) for conv, f in zip(self.output_convs, results)]

        # 添加额外层
        last = results[-1]
        for conv in self.extra_convs:
            last = conv(last)
            results.append(last)

        return results








# for test only
if __name__ == '__main__':
    cfg = dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048], 
        out_channel=256, 
        num_extra_levels=1
    )
    fpn = MODELS.build_from_cfg(cfg)
    print(fpn)
    x = [
        torch.randn(1, 256, 160, 160),
        torch.randn(1, 512, 80, 80),
        torch.randn(1, 1024, 40, 40),
        torch.randn(1, 2048, 20, 20)
        ]
    outs = fpn(x)
    for i, o in enumerate(outs):
        print(f"P{i+3}: {o.shape}")