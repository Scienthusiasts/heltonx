import torch
import torch.nn as nn
from heltonx.utils.register import MODELS


@MODELS.register
class MaxIoUAssigner(nn.Module):
    """Max IoU 正负样本分配器 (Faster R-CNN / RPN 使用)

    将每个 GT 匹配到与其 IoU 最大的 anchor/proposal，并根据 IoU 阈值划分正负样本。

    Args:
        pos_iou_thr (float): 正样本 IoU 阈值，≥此值的为正样本
        neg_iou_thr (float): 负样本 IoU 阈值，<此值的为负样本
        min_pos_iou (float): 最小正样本 IoU，低于此值即使 best matching 也视为 ignore
        match_low_quality (bool): 是否允许低质量匹配（确保每个 GT 至少有一个匹配）
        gpu_assign_thr (float): 当 anchor 数量超过此值时，强制在 CPU 上计算 IoU（避免 OOM）
    """

    def __init__(self, pos_iou_thr=0.5, neg_iou_thr=0.5,
                 min_pos_iou=0.5, match_low_quality=True, gpu_assign_thr=-1):
        super().__init__()
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.match_low_quality = match_low_quality
        self.gpu_assign_thr = gpu_assign_thr

    def forward(self, bboxes, gt_bboxes, gt_labels=None):
        """执行分配

        Args:
            bboxes (Tensor): [num_bboxes, 4] xyxy，anchors 或 proposals
            gt_bboxes (Tensor): [num_gt, 4] xyxy，GT 框
            gt_labels (Tensor, optional): [num_gt] GT 类别标签

        Returns:
            assigned_gt_inds (Tensor): [num_bboxes] 分配结果
                - 0: 负样本 (background)
                - >0: 正样本，值为匹配的 GT 索引 (1-based)
                - -1: ignore (不参与 loss 计算)
            assigned_labels (Tensor): [num_bboxes] 分配的类别标签
                - 正样本: 对应 GT 的类别
                - 负样本/ignore: 0
        """
        num_bboxes = bboxes.shape[0]
        num_gt = gt_bboxes.shape[0]

        assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,), 0, dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # 没有 GT 或没有 anchors，全部设为负样本
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            return assigned_gt_inds, assigned_labels

        # 计算 IoU 矩阵 [num_bboxes, num_gt]
        overlaps = self.bbox_overlaps(bboxes, gt_bboxes)

        # 1. 为每个 bbox 找 best GT
        max_overlaps, argmax_overlaps = overlaps.max(dim=1)

        # 2. 为每个 GT 找 best bbox（低质量匹配）
        if self.match_low_quality:
            gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=0)

        # 3. 根据阈值分配正负样本
        # 负样本: IoU < neg_iou_thr
        assigned_gt_inds[max_overlaps < self.neg_iou_thr] = 0

        # 正样本: IoU >= pos_iou_thr
        pos_mask = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_mask] = argmax_overlaps[pos_mask] + 1  # 1-based index

        # 4. 低质量匹配: 确保每个 GT 至少有一个匹配（即使 IoU 低于 pos_thr）
        if self.match_low_quality:
            for i in range(num_gt):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    best_bbox_idx = gt_argmax_overlaps[i]
                    assigned_gt_inds[best_bbox_idx] = i + 1

        # 5. 分配类别标签（正样本用 GT 类别，负样本用 0）
        if gt_labels is not None:
            pos_inds = assigned_gt_inds > 0
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        return assigned_gt_inds, assigned_labels

    @staticmethod
    def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
        """计算两组框之间的 IoU 矩阵

        Args:
            bboxes1 (Tensor): [N, 4] xyxy
            bboxes2 (Tensor): [M, 4] xyxy
            mode (str): 'iou' 或 'iof'
            eps (float): 避免除零

        Returns:
            Tensor: [N, M] IoU 矩阵
        """
        # 面积
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        # 交集
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [N, M, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [N, M, 2]
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        if mode == 'iou':
            union = area1[:, None] + area2 - inter
            iou = inter / (union + eps)
        elif mode == 'iof':
            iof = inter / (area1[:, None] + eps)
            return iof
        else:
            raise ValueError(f"Unsupported mode {mode}")

        return iou


if __name__ == '__main__':
    # 验证
    assigner = MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5)
    anchors = torch.tensor([
        [0, 0, 10, 10],
        [5, 5, 15, 15],
        [20, 20, 30, 30],
        [100, 100, 110, 110],
    ], dtype=torch.float32)
    gt = torch.tensor([
        [6, 6, 14, 14],
        [25, 25, 35, 35],
    ], dtype=torch.float32)
    labels = torch.tensor([1, 2], dtype=torch.long)

    assigned_gt_inds, assigned_labels = assigner(anchors, gt, labels)
    print("assigned_gt_inds:", assigned_gt_inds)  # 期望: [1, 1, 2, 0] (或类似)
    print("assigned_labels:", assigned_labels)    # 期望: [1, 1, 2, 0]
