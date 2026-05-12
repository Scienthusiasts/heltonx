import torch
import torch.nn as nn
import math
from torchvision.ops import roi_align as tv_roi_align
from heltonx.utils.register import MODELS


def bbox2roi(bbox_list):
    """将 bbox list 转换为 RoI 格式

    Args:
        bbox_list (List[Tensor]): 每张图的 bbox，每个 Tensor 形状为 [num_bboxes, 4] (xyxy)

    Returns:
        Tensor: [total_num_bboxes, 5]，格式为 [batch_ind, x1, y1, x2, y2]
    """
    rois = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.numel() == 0:
            continue
        img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
        rois.append(torch.cat([img_inds, bboxes], dim=-1))
    if len(rois) == 0:
        return bbox_list[0].new_zeros((0, 5))
    return torch.cat(rois, dim=0)


def roi2bbox(roi_tensor):
    """将 RoI 格式转回 bbox list

    Args:
        roi_tensor (Tensor): [total_num_bboxes, 5]，格式为 [batch_ind, x1, y1, x2, y2]

    Returns:
        List[Tensor]: 按 batch_ind 分组的 bbox 列表
    """
    if roi_tensor.numel() == 0:
        return []
    batch_inds = roi_tensor[:, 0].long()
    max_batch = batch_inds.max().item()
    results = []
    for i in range(max_batch + 1):
        mask = batch_inds == i
        results.append(roi_tensor[mask, 1:])
    return results


def map_rois_to_fpn_levels(rois, num_levels=5, finest_scale=56):
    """根据 RoI 的尺度将其分配到对应的 FPN 层级

    参考 MMDetection 实现，使用 sqrt(w*h) 作为尺度度量：
        k = floor( log2( sqrt(w*h) / finest_scale ) ) + k0
    其中 k0 对应最精细层级（P2 -> level 0）

    Args:
        rois (Tensor): [N, 5] 或 [N, 4]，格式为 (batch_ind, x1, y1, x2, y2) 或 xyxy
        num_levels (int): FPN 层数，默认 5 (P2~P6)
        finest_scale (int): 参考尺度，默认 56

    Returns:
        Tensor: [N] 每个 roi 对应的层级索引 (0 ~ num_levels-1)
    """
    if rois.size(-1) == 5:
        boxes = rois[:, 1:]
    else:
        boxes = rois

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    scales = torch.sqrt(w * h)

    # target_lvls = floor( log2(scales / finest_scale) ) + 2
    # P2 对应 level 0, P3->1, ...
    # MMDetection 公式: target_lvls = floor(log2(sqrt(w*h)/224) + 4)
    # 等价于: floor(log2(sqrt(w*h)/56)) + 2
    target_lvls = torch.floor(torch.log2(scales / finest_scale + 1e-6)) + 2
    target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
    return target_lvls


def roi_align(featmaps, rois, output_size=(7, 7), spatial_scale=1.0,
              sampling_ratio=-1, aligned=True):
    """对单张特征图执行 RoI Align

    Args:
        featmaps (Tensor): [C, H, W] 或 [bs, C, H, W]
        rois (Tensor): [N, 5] 格式为 [batch_ind, x1, y1, x2, y2]
        output_size (Tuple[int, int]): 输出空间尺寸，默认 (7, 7)
        spatial_scale (float): 特征图相对于原图的下采样率
        sampling_ratio (int): 每维采样点数，-1 表示自适应
        aligned (bool): 是否使用 aligned RoI Align

    Returns:
        Tensor: [N, C, output_size[0], output_size[1]]
    """
    return tv_roi_align(
        featmaps, rois, output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned
    )


def multilevel_roi_align(featmaps, rois, output_size=(7, 7),
                         featmap_strides=[4, 8, 16, 32, 64],
                         sampling_ratio=-1, aligned=True,
                         finest_scale=56):
    """多层级 RoI Align (FPN 版本)

    根据每个 RoI 的尺度将其分配到对应的 FPN 层级，然后分别执行 RoI Align。

    Args:
        featmaps (List[Tensor]): 每层特征图，例如 [P2, P3, P4, P5, P6]
        rois (Tensor): [N, 5] 格式为 [batch_ind, x1, y1, x2, y2]
        output_size (Tuple[int, int]): 输出空间尺寸，默认 (7, 7)
        featmap_strides (List[int]): 每层的下采样率
        sampling_ratio (int): 每维采样点数
        aligned (bool): 是否使用 aligned RoI Align
        finest_scale (int): 层级分配参考尺度

    Returns:
        Tensor: [N, C, output_size[0], output_size[1]]
    """
    num_levels = len(featmaps)
    target_lvls = map_rois_to_fpn_levels(rois, num_levels=num_levels,
                                         finest_scale=finest_scale)

    # 为每个层级收集对应的 rois
    roi_results = []
    roi_indices = []
    for lvl in range(num_levels):
        lvl_mask = target_lvls == lvl
        if not lvl_mask.any():
            continue
        lvl_rois = rois[lvl_mask]
        lvl_feat = featmaps[lvl]
        spatial_scale = 1.0 / featmap_strides[lvl]
        lvl_results = tv_roi_align(
            lvl_feat, lvl_rois, output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned
        )
        roi_results.append(lvl_results)
        roi_indices.append(lvl_mask.nonzero(as_tuple=False).squeeze(1))

    if len(roi_results) == 0:
        # 没有 rois，返回空张量
        C = featmaps[0].size(1)
        return rois.new_zeros((0, C, output_size[0], output_size[1]))

    # 按原始顺序拼接结果
    all_results = torch.cat(roi_results, dim=0)
    all_indices = torch.cat(roi_indices, dim=0)
    # 恢复原始顺序
    sorted_indices = torch.argsort(all_indices)
    return all_results[sorted_indices]


@MODELS.register
class RoIAlign(nn.Module):
    """RoI Align 模块封装 (支持单层级和多层级)

    Args:
        output_size (Tuple[int, int]): 输出空间尺寸
        spatial_scale (float): 特征图相对于原图的下采样率
        sampling_ratio (int): 每维采样点数
        aligned (bool): 是否使用 aligned RoI Align
        use_multilevel (bool): 是否使用多层级 (FPN)
        featmap_strides (List[int]): FPN 各层下采样率
        finest_scale (int): 层级分配参考尺度
    """

    def __init__(self, output_size=(7, 7), spatial_scale=1.0 / 16.0,
                 sampling_ratio=-1, aligned=True,
                 use_multilevel=False, featmap_strides=[4, 8, 16, 32, 64],
                 finest_scale=56):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        self.use_multilevel = use_multilevel
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale

    def forward(self, features, rois):
        """
        Args:
            features (Tensor or List[Tensor]): 单层级特征图或多层级特征图列表
            rois (Tensor): [N, 5] 格式为 [batch_ind, x1, y1, x2, y2]

        Returns:
            Tensor: [N, C, output_size[0], output_size[1]]
        """
        if self.use_multilevel:
            return multilevel_roi_align(
                features, rois,
                output_size=self.output_size,
                featmap_strides=self.featmap_strides,
                sampling_ratio=self.sampling_ratio,
                aligned=self.aligned,
                finest_scale=self.finest_scale
            )
        else:
            return tv_roi_align(
                features, rois, self.output_size,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                aligned=self.aligned
            )


if __name__ == '__main__':
    # 验证 multilevel_roi_align
    featmaps = [
        torch.randn(2, 256, 160, 160),  # P2, stride=4
        torch.randn(2, 256, 80, 80),    # P3, stride=8
        torch.randn(2, 256, 40, 40),    # P4, stride=16
        torch.randn(2, 256, 20, 20),    # P5, stride=32
        torch.randn(2, 256, 10, 10),    # P6, stride=64
    ]
    rois = torch.tensor([
        [0, 10, 10, 50, 50],    # 小目标 -> P2
        [0, 200, 200, 400, 400], # 大目标 -> P4/P5
        [1, 100, 100, 300, 300], # 大目标 -> P4/P5
    ], dtype=torch.float32)

    out = multilevel_roi_align(featmaps, rois, output_size=(7, 7),
                               featmap_strides=[4, 8, 16, 32, 64])
    print("multilevel_roi_align output shape:", out.shape)

    # 验证 RoIAlign 模块
    roi_align_module = RoIAlign(
        output_size=(7, 7),
        use_multilevel=True,
        featmap_strides=[4, 8, 16, 32, 64]
    )
    out2 = roi_align_module(featmaps, rois)
    print("RoIAlign module output shape:", out2.shape)
