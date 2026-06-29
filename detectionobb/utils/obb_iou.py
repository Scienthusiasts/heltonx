import torch
import math
import numpy as np


def _get_covariance_matrix(boxes):
    """从旋转框生成协方差矩阵

    将 OBB 的 (w, h, r) 转换为 2D 高斯分布的协方差矩阵
    Σ = [[a, c], [c, b]]

    Args:
        boxes (Tensor): [N, 5] xywhr 格式

    Returns:
        tuple: (a, b, c) 协方差矩阵分量, 各 [N, 1]
    """
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1, obb2, eps=1e-7):
    """计算旋转框之间的概率 IoU (基于 Bhattacharyya 距离)

    参考: https://arxiv.org/pdf/2106.06072v1.pdf

    Args:
        obb1 (Tensor | ndarray): [N, 5] GT 旋转框 (xywhr)
        obb2 (Tensor | ndarray): [M, 5] 预测旋转框 (xywhr)
        eps (float): 防除零小值

    Returns:
        (Tensor): [N, M] 相似度矩阵
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp(min=eps) * (a2 * b2 - c2.pow(2)).clamp(min=eps)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return torch.nan_to_num(1 - hd, nan=0.0, posinf=1.0, neginf=0.0)


def probiou_loss(pred, target, eps=1e-7):
    """基于 probiou 的成对回归损失

    Args:
        pred (Tensor): [N, 5] 预测框 xywhr
        target (Tensor): [N, 5] 目标框 xywhr
        eps (float): 防除零

    Returns:
        (Tensor): [N] 每对的损失 (1 - probiou)
    """
    iou = batch_probiou(pred, target).diag()
    return 1 - iou


def probiou_matrix(obb1, obb2, eps=1e-7):
    """计算旋转框 IoU 矩阵 (用于 assigner)

    Args:
        obb1 (Tensor): [bs, N, 5] 预测框 xywhr
        obb2 (Tensor): [bs, M, 5] GT 框 xywhr

    Returns:
        (Tensor): [bs, M, N] probiou 矩阵
    """
    bs, N, _ = obb1.shape
    M = obb2.shape[1]

    # 逐 batch 计算
    overlaps = torch.zeros(bs, M, N, device=obb1.device, dtype=obb1.dtype)
    for b in range(bs):
        overlaps[b] = batch_probiou(obb2[b], obb1[b], eps)
    return overlaps
