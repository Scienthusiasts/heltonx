import torch
import torch.nn as nn
import torch.nn.functional as F

from heltonx.utils.utils import init_weights
from heltonx.utils.register import MODELS
from detection.models.yolo_blocks import C2f




class ConvBlock(nn.Module):
    """Conv + BN + ReLU 基本卷积块"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)






# @MODELS.register
# class C2fPAFPN(nn.Module):
#     """Path Aggregation Feature Pyramid Network (PAFPN)
#        参考: PANet, CVPR 2018
#     """
#     def __init__(self, in_channels, out_channel=256, num_extra_levels=2, C2f_n=1):
#         """
#         Args:
#             in_channels (List[int]): backbone 各层输出通道，例如 [256, 512, 1024, 2048]
#             out_channel (int): 每层输出通道数
#             num_extra_levels (int): 额外下采样层数量（如 P6, P7）
#             C2f_n
#         """
#         super().__init__()
#         self.num_levels = len(in_channels)
#         self.out_channels = out_channel
#         self.num_extra_levels = num_extra_levels

#         # 1. Top-down: lateral 1x1 conv 对齐通道
#         self.lateral_convs = nn.ModuleList([
#             nn.Conv2d(c, out_channel, kernel_size=1)
#             for c in in_channels
#         ])
#         # 2. Top-down: 3x3 conv 平滑（含 BN+ReLU）
#         # ✅ 这里替换成C2f
#         self.fpn_convs = nn.ModuleList([
#             C2f(out_channel, out_channel, n=C2f_n, shortcut=True)
#             for _ in in_channels
#         ])
#         # 3. Bottom-up: 增强路径 (PAFPN 关键)
#         #   对相邻层的下采样后融合
#         self.pan_convs = nn.ModuleList([
#             ConvBlock(out_channel, out_channel, k=3, s=2, p=1)
#             for _ in range(self.num_levels - 1)
#         ])
#         # 4. Bottom-up: 平滑卷积（再融合一次）
#         # ✅ 这里替换成C2f
#         self.output_convs = nn.ModuleList([
#             C2f(out_channel, out_channel, n=C2f_n, shortcut=True)
#             for _ in range(self.num_levels)
#         ])
#         # 5. 额外下采样层 (P6, P7)
#         self.extra_convs = nn.ModuleList([
#             ConvBlock(out_channel, out_channel, k=3, s=2, p=1)
#             for _ in range(num_extra_levels)
#         ])
#         # 初始化
#         for m in self.modules():
#             init_weights(m, 'normal', 0, 0.01)


#     def _upsample_add(self, higher, lower):
#         """上采样 + 相加"""
#         _, _, H, W = lower.shape
#         return F.interpolate(higher, size=(H, W), mode='bilinear') + lower


#     def forward(self, features):
#         """
#         Args:
#             features (List[Tensor]): backbone 输出的多层特征图 [C3, C4, C5, ...]
#         Returns:
#             List[Tensor]: 多尺度输出 [P3, P4, P5, P6, P7...]
#         """
#         assert len(features) == self.num_levels, \
#             f"Expect {self.num_levels} input features, got {len(features)}"

#         # Step 1: lateral conv 投影
#         laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
#         # Step 2: top-down 路径融合
#         for i in range(self.num_levels - 1, 0, -1):
#             laterals[i - 1] = self._upsample_add(laterals[i], laterals[i - 1])
#         # Step 3: top-down 平滑卷积
#         fpn_outs = [conv(f) for conv, f in zip(self.fpn_convs, laterals)]
#         # Step 4: bottom-up 路径增强
#         pan_outs = [fpn_outs[0]]
#         for i in range(1, self.num_levels):
#             down = self.pan_convs[i - 1](pan_outs[-1])
#             fused = down + fpn_outs[i]
#             pan_outs.append(fused)
#         # Step 5: bottom-up 平滑卷积
#         outs = [conv(f) for conv, f in zip(self.output_convs, pan_outs)]
#         # Step 6: 添加额外层 (P6, P7)
#         last = outs[-1]
#         for conv in self.extra_convs:
#             last = conv(last)
#             outs.append(last)

#         return outs





@MODELS.register
class C2fPAFPN(nn.Module):
    """YOLOv8-style Path Aggregation PAFPN

    Args:
        in_channels (List[int]): 各层输入通道数，如 [512, 1024, 2048]
        out_channel (int): 输出通道数，默认 256
        num_extra_levels (int): 额外下采样层数量，默认 2 (P6, P7)
        C2f_n (int): C2f 模块中的 Bottleneck 数量，默认 1
    """
    def __init__(self, in_channels, out_channel=256, num_extra_levels=2, C2f_n=1):
        super().__init__()
        self.num_levels = len(in_channels)
        self.out_channels = out_channel
        self.num_extra_levels = num_extra_levels

        # 1. lateral 1x1 conv
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channel, kernel_size=1)
            for c in in_channels
        ])
        # 2. 上采样阶段 C2f（融合后使用）
        self.top_down_C2f = nn.ModuleList([
            C2f(out_channel * 2, out_channel, n=C2f_n, shortcut=False)
            for _ in range(self.num_levels - 1)
        ])
        # 3. 下采样卷积
        self.downsample_convs = nn.ModuleList([
            ConvBlock(out_channel, out_channel, k=3, s=2, p=1)
            for _ in range(self.num_levels - 1)
        ])
        # 4. 下采样阶段 C2f（融合后使用）
        self.bottom_up_C2f = nn.ModuleList([
            C2f(out_channel * 2, out_channel, n=C2f_n, shortcut=False)
            for _ in range(self.num_levels - 1)
        ])
        # 5. 额外层 (P6, P7)
        self.extra_convs = nn.ModuleList([
            ConvBlock(out_channel, out_channel, k=3, s=2, p=1)
            for _ in range(num_extra_levels)
        ])
        # 初始化
        for m in self.modules():
            init_weights(m, 'normal', 0, 0.01)


    def _upsample_cat(self, higher, lower):
        """上采样 + 拼接"""
        _, _, H, W = lower.shape
        # 不用add而用concat
        up = F.interpolate(higher, size=(H, W), mode='bilinear')
        return torch.cat((up, lower), dim=1)


    def forward(self, features):
        """
        Args:
            features (List[Tensor]): [C3, C4, C5, ...]
        Returns:
            [P3, P4, P5, P6, P7...]
        """
        assert len(features) == self.num_levels
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        # Top-down
        td_results = [laterals[-1]]  # start from top (C5)
        for i in range(self.num_levels - 2, -1, -1):
            fusion = self._upsample_cat(td_results[0], laterals[i])
            out = self.top_down_C2f[i](fusion)
            td_results.insert(0, out)  # prepend (keep order low->high)
        # Bottom-up
        bu_results = [td_results[0]]
        for i in range(1, self.num_levels):
            down = self.downsample_convs[i - 1](bu_results[-1])
            fusion = torch.cat((down, td_results[i]), dim=1)
            out = self.bottom_up_C2f[i - 1](fusion)
            bu_results.append(out)
        # Extra layers
        last = bu_results[-1]
        for conv in self.extra_convs:
            last = conv(last)
            bu_results.append(last)

        return bu_results





# for test only
if __name__ == '__main__':
    cfg = dict(
        type='PAFPN',
        in_channels=[512, 1024, 2048],
        out_channel=256,
        num_extra_levels=2
    )
    fpn = MODELS.build_from_cfg(cfg)
    print(fpn)

    x = [
        torch.randn(4, 512, 80, 80),
        torch.randn(4, 1024, 40, 40),
        torch.randn(4, 2048, 20, 20)
    ]

    outs = fpn(x)
    for i, o in enumerate(outs):
        print(f"P{i+3}: {o.shape}")
