import torch
import torch.nn as nn
import numpy as np
import math
from heltonx.utils.register import MODELS


@MODELS.register
class AnchorGenerator(nn.Module):
    """标准 Anchor 生成器 (参考 MMDetection 实现)

    为 FPN 的每一层特征图生成对应尺度的 anchors。
    每层 anchor 尺寸 = base_size * scales * ratios

    Args:
        strides (List[int]): 每层特征图相对于原图的下采样率, e.g. [4, 8, 16, 32, 64]
        ratios (List[float]): 宽高比, e.g. [0.5, 1.0, 2.0]
        scales (List[float]): 尺度系数, e.g. [8]
        base_sizes (List[int], optional): 每层的基础尺寸。默认等于 strides
    """

    def __init__(self, strides, ratios=[0.5, 1.0, 2.0], scales=[8], base_sizes=None):
        super().__init__()
        self.strides = strides
        self.ratios = torch.tensor(ratios, dtype=torch.float32)
        self.scales = torch.tensor(scales, dtype=torch.float32)
        self.num_base_anchors = len(ratios) * len(scales)

        if base_sizes is None:
            self.base_sizes = strides
        else:
            self.base_sizes = base_sizes

        # 预计算每层的 base anchors (zero-centered)
        self.base_anchors = self._generate_base_anchors()

    def _generate_base_anchors(self):
        """生成零中心的基础 anchors

        Returns:
            List[Tensor]: 每层的基础 anchors, 每个 Tensor 形状为 [num_base_anchors, 4] (xyxy)
        """
        base_anchors = []
        for base_size in self.base_sizes:
            # 参考面积
            area = base_size * base_size
            # 面积按 scales^2 缩放
            areas = area * (self.scales ** 2)
            # 计算每个 ratio 下的 w 和 h
            # w = sqrt(area / ratio), h = sqrt(area * ratio)
            # 等价于: w = sqrt(areas) * sqrt(1/ratios), h = sqrt(areas) * sqrt(ratios)
            # 广播: areas [num_scales], ratios [num_ratios]
            # 我们需要 [num_scales, num_ratios]
            ws = (areas[:, None] / self.ratios[None, :]).sqrt()  # [num_scales, num_ratios]
            hs = (areas[:, None] * self.ratios[None, :]).sqrt()

            # 转换为 zero-centered xyxy
            # x1 = -w/2, y1 = -h/2, x2 = w/2, y2 = h/2
            x1 = -ws / 2
            y1 = -hs / 2
            x2 = ws / 2
            y2 = hs / 2

            # reshape 为 [num_scales * num_ratios, 4]
            base_anchor = torch.stack([x1.flatten(), y1.flatten(), x2.flatten(), y2.flatten()], dim=-1)
            base_anchors.append(base_anchor)
        return base_anchors

    def single_level_grid_anchors(self, base_anchor, featmap_size, stride, device):
        """在单一层特征图上生成网格 anchors

        Args:
            base_anchor (Tensor): [num_base_anchors, 4], zero-centered
            featmap_size (Tuple[int, int]): (h, w)
            stride (int): 下采样率
            device: torch device

        Returns:
            Tensor: [h * w * num_base_anchors, 4], 原图坐标系下的 xyxy anchors
        """
        feat_h, feat_w = featmap_size
        num_base_anchors = base_anchor.shape[0]

        # 网格偏移: 每个 grid 的中心点坐标
        shifts_x = torch.arange(0, feat_w, device=device, dtype=torch.float32) * stride + stride / 2
        shifts_y = torch.arange(0, feat_h, device=device, dtype=torch.float32) * stride + stride / 2

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        # shifts: [feat_h * feat_w, 4] = [dx, dy, dx, dy]
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1)

        # base_anchor: [num_base_anchors, 4]
        # shifts: [feat_h*feat_w, 4]
        # 结果: [feat_h*feat_w, num_base_anchors, 4]
        anchors = base_anchor[None, :, :] + shifts[:, None, :]
        anchors = anchors.reshape(-1, 4)

        return anchors

    def grid_anchors(self, featmap_sizes, device='cuda'):
        """为所有 FPN 层生成 anchors

        Args:
            featmap_sizes (List[Tuple[int, int]]): 每层特征图尺寸 [(h1,w1), (h2,w2), ...]
            device: torch device

        Returns:
            List[Tensor]: 每层的 anchors, 每个 [h*w*num_base_anchors, 4] (xyxy)
        """
        assert len(featmap_sizes) == len(self.strides), \
            f"featmap_sizes ({len(featmap_sizes)}) 与 strides ({len(self.strides)}) 数量不匹配"

        anchors = []
        for i, featmap_size in enumerate(featmap_sizes):
            anchor = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_size,
                self.strides[i],
                device
            )
            anchors.append(anchor)
        return anchors

    def num_base_anchors_per_level(self):
        """返回每层的基础 anchor 数量"""
        return self.num_base_anchors


if __name__ == '__main__':
    # 验证
    anchor_gen = AnchorGenerator(
        strides=[4, 8, 16, 32, 64],
        ratios=[0.5, 1.0, 2.0],
        scales=[8]
    )
    featmap_sizes = [(160, 160), (80, 80), (40, 40), (20, 20), (10, 10)]
    anchors = anchor_gen.grid_anchors(featmap_sizes, device='cpu')
    for i, anchor in enumerate(anchors):
        print(f"P{i+2}: stride={anchor_gen.strides[i]}, featmap={featmap_sizes[i]}, anchors={anchor.shape}")
