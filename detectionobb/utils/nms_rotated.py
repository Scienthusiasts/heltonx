"""旋转框 NMS (可选工具)

YOLO26 使用 NMS-free 推理 (o2o 头 + topk), 通常不需要 NMS.
本模块提供基于 probiou 的旋转框 NMS, 供后处理使用.
"""

import torch
from detectionobb.utils.obb_iou import batch_probiou


def rotated_nms(boxes, scores, iou_thr=0.5):
    """基于 probiou 的旋转框 NMS

    Args:
        boxes (Tensor): [N, 5] xywhr
        scores (Tensor): [N] 置信度
        iou_thr (float): IoU 阈值

    Returns:
        (Tensor): 保留的索引
    """
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.sort(descending=True)[1]
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        iou = batch_probiou(boxes[i:i + 1], boxes[order[1:]]).squeeze(0)
        mask = iou <= iou_thr
        order = order[1:][mask]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_rotated_nms(boxes, scores, idxs, iou_thr=0.5):
    """分类别旋转框 NMS

    Args:
        boxes (Tensor): [N, 5] xywhr
        scores (Tensor): [N] 置信度
        idxs (Tensor): [N] 类别索引
        iou_thr (float): IoU 阈值

    Returns:
        (Tensor): 保留的索引
    """
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    keep = []
    for cls_id in idxs.unique():
        mask = idxs == cls_id
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        cls_keep = rotated_nms(cls_boxes, cls_scores, iou_thr)
        # 映射回原始索引
        orig_idx = mask.nonzero(as_tuple=True)[0][cls_keep]
        keep.append(orig_idx)
    return torch.cat(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)
