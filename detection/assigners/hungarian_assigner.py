import torch
import torch.nn as nn
from heltonx.utils.register import MODELS
from detection.utils.detr_utils import hungarian_matcher


@MODELS.register
class HungarianAssigner(nn.Module):
    """DETR 匈牙利匹配分配器（与官方 DETR 一致）

    基于 DETR 的二分图匹配策略，将 query 与 GT 进行一对一匹配。
    代价矩阵 = cls_cost + L1_cost + GIoU_cost
    所有代价均在归一化空间计算。

    Args:
        cls_cost_weight (float):   分类代价权重
        l1_cost_weight (float):    L1 回归代价权重
        giou_cost_weight (float):  GIoU 回归代价权重
    """

    def __init__(self, cls_cost_weight=1.0, l1_cost_weight=5.0, giou_cost_weight=2.0):
        super(HungarianAssigner, self).__init__()
        self.cls_cost_weight = cls_cost_weight
        self.l1_cost_weight = l1_cost_weight
        self.giou_cost_weight = giou_cost_weight

    def forward(self, cls_preds, box_preds, gt_labels, gt_bboxes):
        """执行匈牙利匹配

        Args:
            cls_preds:  [bs, num_queries, nc+1] 分类 logits (包含背景类)
            box_preds:  [bs, num_queries, 4]  归一化 cxcywh
            gt_labels:  list[Tensor], 每个 [num_gt]
            gt_bboxes:  list[Tensor], 每个 [num_gt, 4] 归一化 cxcywh

        Returns:
            list[tuple]: 每个 batch 的 (query_idx, gt_idx) 匹配对
        """
        return hungarian_matcher(
            cls_preds, box_preds, gt_labels, gt_bboxes,
            cls_cost_weight=self.cls_cost_weight,
            l1_cost_weight=self.l1_cost_weight,
            giou_cost_weight=self.giou_cost_weight,
        )
