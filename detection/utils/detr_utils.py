import torch
import math
from scipy.optimize import linear_sum_assignment


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """Sigmoid Focal Loss (DETR 分类损失)

    Args:
        inputs:  [num_queries, num_classes] 预测 logits
        targets: [num_queries, num_classes] one-hot 标签
        alpha:   正负样本平衡因子
        gamma:   调制因子
        reduction: 'mean' / 'sum' / 'none'

    Returns:
        loss: 标量或张量
    """
    prob = inputs.sigmoid()
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def box_cxcywh_to_xyxy(x):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)"""
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """(x1, y1, x2, y2) -> (cx, cy, w, h)"""
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2,
         (x2 - x1), (y2 - y1)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(box1, box2):
    """计算两组框之间的广义 IoU 矩阵

    Args:
        box1: [N, 4] (x1, y1, x2, y2)
        box2: [M, 4] (x1, y1, x2, y2)

    Returns:
        giou: [N, M] 广义 IoU 矩阵
    """
    # 面积
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 交集
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # 并集
    union = area1[:, None] + area2 - inter  # [N, M]
    iou = inter / (union + 1e-7)

    # 最小包围框
    lt_enc = torch.min(box1[:, None, :2], box2[:, :2])
    rb_enc = torch.max(box1[:, None, 2:], box2[:, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]

    giou = iou - (area_enc - union) / (area_enc + 1e-7)
    return giou


def hungarian_matcher(cls_preds, box_preds, gt_labels, gt_bboxes,
                      cls_cost_weight=1.0, l1_cost_weight=5.0, giou_cost_weight=2.0):
    """匈牙利匹配（与官方 DETR 一致：所有代价在归一化空间计算）

    Args:
        cls_preds:      [bs, num_queries, nc+1] 分类 logits (包含背景类)
        box_preds:      [bs, num_queries, 4] 归一化 cxcywh
        gt_labels:      list[Tensor], 每个 [num_gt]
        gt_bboxes:      list[Tensor], 每个 [num_gt, 4] 归一化 cxcywh
        cls_cost_weight:    分类代价权重
        l1_cost_weight:     L1 代价权重
        giou_cost_weight:   GIoU 代价权重

    Returns:
        list[tuple]: 每个 batch 元素的 (query_idx, gt_idx) 匹配对
    """
    bs, num_queries = cls_preds.shape[:2]
    # 对完整的 nc+1 维度做 softmax（与官方一致），然后索引目标类概率
    cls_probs = cls_preds.flatten(0, 1).softmax(-1)  # [bs * num_queries, nc+1]

    # 将 box 转为 xyxy 用于 GIoU 计算
    all_box_xyxy = box_cxcywh_to_xyxy(box_preds.flatten(0, 1))  # [bs * num_queries, 4]

    indices = []
    for b in range(bs):
        # 当前 batch 的 GT
        gt_lbl = gt_labels[b]  # [num_gt]
        gt_bbox = gt_bboxes[b]  # [num_gt, 4]

        num_gt = gt_lbl.shape[0]
        if num_gt == 0:
            indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
            continue

        # 分类代价: [num_queries, num_gt]
        cls_cost = -cls_probs[b * num_queries:(b + 1) * num_queries, gt_lbl]

        # L1 代价: [num_queries, num_gt] (归一化空间，与官方一致)
        bbox_pred = box_preds[b]  # [num_queries, 4]
        l1_cost = torch.cdist(bbox_pred, gt_bbox, p=1)

        # GIoU 代价: [num_queries, num_gt] (归一化空间，与官方 DETR 一致)
        gt_bbox_xyxy = box_cxcywh_to_xyxy(gt_bbox)  # [num_gt, 4]
        giou_cost = -generalized_box_iou(
            all_box_xyxy[b * num_queries:(b + 1) * num_queries],
            gt_bbox_xyxy)

        # 总代价
        cost = cls_cost_weight * cls_cost + l1_cost_weight * l1_cost + giou_cost_weight * giou_cost

        # 匈牙利算法
        cost_cpu = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_cpu)
        indices.append((torch.tensor(row_ind, dtype=torch.long),
                        torch.tensor(col_ind, dtype=torch.long)))

    return indices


class PositionEmbeddingSine2D(torch.nn.Module):
    """2D 正弦位置编码 (DETR 使用)

    输出维度为 num_feats * 2 (满足 hidden_dim)
    """

    def __init__(self, num_feats=128, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x):
        """生成 2D 正弦位置编码

        Args:
            x: 特征图 Tensor [B, C, H, W] (仅用于获取空间尺寸)

        Returns:
            pos: [B, num_feats*2, H, W]
        """
        bs, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        y_embed = torch.arange(1, h + 1, device=device, dtype=dtype).unsqueeze(1).repeat(1, w).unsqueeze(0)
        x_embed = torch.arange(1, w + 1, device=device, dtype=dtype).unsqueeze(0).repeat(h, 1).unsqueeze(0)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_y = y_embed[:, :, :, None] / dim_t  # [B, H, W, num_feats]
        pos_x = x_embed[:, :, :, None] / dim_t

        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)

        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)  # [B, num_feats*2, H, W]
        return pos
