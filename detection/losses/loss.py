import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from heltonx.utils.register import MODELS
from detection.utils.detr_utils import (
    sigmoid_focal_loss, generalized_box_iou,
    box_cxcywh_to_xyxy
)





@MODELS.register
class MSELoss(nn.Module):
    '''L2损失
    '''
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')
        self.reduction = reduction

    def forward(self, pred, target):
        """
        """
        loss = self.loss(pred, target)
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()




@MODELS.register
class BCELoss(nn.Module):
    '''二分类交叉熵损失 sigmoid + bceloss
    '''
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction=reduction
        self.loss = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, pred, target):
        """
        """
        loss = self.loss(pred, target)
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()






@MODELS.register
class FocalLoss(nn.Module):
    '''FocalLoss
    '''
    def __init__(self, reduction='mean', gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.reduction=reduction
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, pred, target):
        """
        """
        loss = self.loss(pred, target)
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()







@MODELS.register
class QFocalLoss(nn.Module):
    '''QFocalLoss
    '''
    def __init__(self, reduction='mean', gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.reduction=reduction
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, pred, target):
        """
        """
        loss = self.loss(pred, target)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        modulating_factor = torch.abs(target - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()










@MODELS.register
class L1Loss(nn.Module):
    '''L1 损失
    '''
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')
        self.reduction = reduction

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'none':
            return loss
        if self.reduction == 'sum':
            return loss.sum()


@MODELS.register
class CrossEntropyLoss(nn.Module):
    '''交叉熵损失 (用于 DETR 分类)
    '''
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.reduction = reduction

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'none':
            return loss
        if self.reduction == 'sum':
            return loss.sum()


@MODELS.register
class IoULoss(nn.Module):
    '''L2损失
    '''
    def __init__(self, iou_type, xywh=False, reduction='mean'):
        super(IoULoss, self).__init__()
        self.reduction = reduction
        self.iou_type = iou_type
        self.xywh = xywh
        self.eps = 1e-7


    def forward(self, pred, target):
        """
        """
        iou = self.bbox_iou_pairwise(pred, target)
        # iou = self.yolov8_bbox_iou_pairwise(pred, target).squeeze(-1)
        loss = 1. - iou
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()
        
    def bbox_iou_pairwise(self, box1, box2):
        """计算 box1 和 box2 的 IoU (对应位置一对一计算)
            Args:
                box1: [total_anchor_num, 4(x, y, w, h / x0, y0, x1, y1)]
                box2: [total_anchor_num, 4(x, y, w, h / x0, y0, x1, y1)]
            Returns:
                iou:  [bs, anchor_num, h, w]
        """
        if self.xywh:  # (cx, cy, w, h) → (x1, y1, x2, y2)
            x1, y1, w1, h1 = box1.unbind(-1)
            x2, y2, w2, h2 = box2.unbind(-1)
            b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
            b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
            b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
            b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
        else:  # (x1, y1, x2, y2)
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

        # 相交区域
        inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
        inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        inter = inter_w * inter_h
        # 各自面积
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        # 并集面积
        union = area1 + area2 - inter + self.eps
        # IoU
        iou = inter / union

        # 处理 GIoU / DIoU / CIoU
        if self.iou_type in ["giou", "diou", "ciou"]:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 包围盒宽度
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 包围盒高度

            if self.iou_type in ["diou", "ciou"]:
                c2 = cw ** 2 + ch ** 2 + self.eps
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2) / 4
                if self.iou_type == "ciou":
                    v = (4 / math.pi**2) * (torch.atan((b2_x2 - b2_x1) / (b2_y2 - b2_y1 + self.eps)) -
                                            torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1 + self.eps)))**2
                    with torch.no_grad():
                        alpha = v / (v - iou + 1 + self.eps)
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                return iou - rho2 / c2  # DIoU
            # GIoU
            c_area = cw * ch + self.eps
            return iou - (c_area - union) / c_area
        # IoU
        return iou
    



    def yolov8_bbox_iou_pairwise(self, box1, box2, xywh=True, eps=1e-7):
        """来自YOLOv8源码
        Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4). from yolo8 Ultralytics

        Args:
            box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
            box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
            xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                                (x1, y1, x2, y2) format. Defaults to True.
            GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
            DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
            CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
        """

        # Get the coordinates of bounding boxes
        if xywh:  # transform from xywh to xyxy
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        ).clamp_(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + eps

        # IoU
        iou = inter / union
        if self.iou_type in ["giou", "diou", "ciou"]:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
            if self.iou_type in ["diou", "ciou"]:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw**2 + ch**2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
                if self.iou_type == "ciou": # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                return iou - rho2 / c2  # DIoU
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
        return iou  # IoU


# ============================================================
# DETR Loss 模块
# ============================================================


@MODELS.register
class DETRCrossEntropyLoss(nn.Module):
    """DETR 分类损失：Cross Entropy Loss（与官方 DETR 实现一致）

    使用 softmax 多分类 + reduction='mean'，背景类（索引 nc）权重为 eos_coef。
    官方 DETR 直接使用 F.cross_entropy(pred, target, weight) 默认 mean 归一化。

    Args:
        nc (int):       前景类别数 (不含背景)
        eos_coef (float): 背景类权重系数，默认 0.1
    """

    def __init__(self, nc, eos_coef=0.1):
        super().__init__()
        self.nc = nc
        self.eos_coef = eos_coef
        # 背景类权重较小，降低其对损失的贡献
        weight = torch.ones(nc + 1)
        weight[nc] = eos_coef
        self.register_buffer('weight', weight)

    def forward(self, cls_preds, cls_targets):
        """
        Args:
            cls_preds:   [B, num_queries, nc+1] 分类 logits
            cls_targets: [B, num_queries] 类别索引（nc 表示背景）

        Returns:
            标量分类损失（mean over B*num_queries，与官方 DETR 一致）
        """
        loss = F.cross_entropy(
            cls_preds.reshape(-1, self.nc + 1),
            cls_targets.reshape(-1),
            weight=self.weight,
            reduction='mean'
        )
        return loss


@MODELS.register
class DETRFocalLoss(nn.Module):
    """DETR 分类损失：Sigmoid Focal Loss（备选）

    Args:
        alpha (float): 正负样本平衡因子
        gamma (float): 调制因子
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_preds, cls_targets, num_boxes):
        """
        Args:
            cls_preds:   [B, num_queries, nc+1] 分类 logits
            cls_targets: [B, num_queries, nc+1] one-hot 标签
            num_boxes:   int, GT 框总数（归一化因子）

        Returns:
            标量分类损失
        """
        loss = sigmoid_focal_loss(
            cls_preds, cls_targets,
            alpha=self.alpha, gamma=self.gamma,
            reduction='sum'
        )
        return loss / num_boxes


@MODELS.register
class DETRL1Loss(nn.Module):
    """DETR L1 回归损失（仅对匹配 query 计算，除以 num_boxes）

    与 mmdetection 一致：在归一化 cxcywh 空间计算。
    """

    def __init__(self):
        super().__init__()

    def forward(self, box_preds, box_targets, matched_masks, num_boxes):
        """
        Args:
            box_preds:    [B, num_queries, 4] 归一化 cxcywh
            box_targets:   [B, num_queries, 4] 归一化 cxcywh
            matched_masks: [B, num_queries] bool，True 表示该 query 被匹配
            num_boxes:     int, GT 框总数（归一化因子）

        Returns:
            标量 L1 损失
        """
        pred_matched = box_preds[matched_masks]
        target_matched = box_targets[matched_masks]
        return F.l1_loss(pred_matched, target_matched, reduction='sum') / num_boxes


@MODELS.register
class DETRGiouLoss(nn.Module):
    """DETR GIoU 回归损失（仅对匹配 query 计算，除以 num_boxes）

    与 mmdetection 一致：GIoU 在反归一化后的像素坐标上计算，
    因为 GIoU 对绝对尺度敏感，归一化空间会压缩梯度信号。
    """

    def __init__(self):
        super().__init__()

    def forward(self, box_preds, box_targets, matched_masks, num_boxes, img_h=None, img_w=None):
        """
        Args:
            box_preds:    [B, num_queries, 4] 归一化 cxcywh
            box_targets:   [B, num_queries, 4] 归一化 cxcywh
            matched_masks: [B, num_queries] bool，True 表示该 query 被匹配
            num_boxes:     int, GT 框总数（归一化因子）
            img_h:         图像高度，用于反归一化。不传则用默认值 1（维持旧行为）
            img_w:         图像宽度

        Returns:
            标量 GIoU 损失
        """
        pred_matched = box_preds[matched_masks]
        target_matched = box_targets[matched_masks]
        pred_xyxy = box_cxcywh_to_xyxy(pred_matched)
        target_xyxy = box_cxcywh_to_xyxy(target_matched)

        # 反归一化到像素空间（mmdetection 标准做法）
        if img_h is not None and img_w is not None:
            factor = torch.tensor([img_w, img_h, img_w, img_h],
                                  device=pred_xyxy.device, dtype=pred_xyxy.dtype)
            pred_xyxy = pred_xyxy * factor
            target_xyxy = target_xyxy * factor

        giou = generalized_box_iou(pred_xyxy, target_xyxy)
        giou_diag = torch.diag(giou)
        return (1 - giou_diag).sum() / num_boxes



@MODELS.register
class SmoothL1Loss(nn.Module):
    '''Smooth L1 损失 (Huber Loss)

    用于 Faster R-CNN 的 RPN 和 RoI Head 回归损失。
    对较小误差使用 L2，对较大误差使用 L1，对 outliers 更鲁棒。

    Args:
        beta (float): L1/L2 切换阈值，默认 1.0/9.0
        reduction (str): 'mean' / 'sum' / 'none'
    '''
    def __init__(self, beta=1.0/9.0, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss = nn.SmoothL1Loss(beta=beta, reduction='none')

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss


# ============================================================
# YOLO26 Loss 模块 (对齐官方 Ultralytics)
# ============================================================


def _ciou_pairwise(a, b):
    """CIoU per pair [N, 4] vs [N, 4] -> [N]"""
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a + area_b - inter + 1e-7
    iou = inter / union
    cw = torch.max(a[:, 2], b[:, 2]) - torch.min(a[:, 0], b[:, 0])
    ch = torch.max(a[:, 3], b[:, 3]) - torch.min(a[:, 1], b[:, 1])
    c2 = cw ** 2 + ch ** 2 + 1e-7
    rho2 = ((b[:, 0] + b[:, 2] - a[:, 0] - a[:, 2]) ** 2 +
            (b[:, 1] + b[:, 3] - a[:, 1] - a[:, 3]) ** 2) / 4
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan((b[:, 2] - b[:, 0]) / (b[:, 3] - b[:, 1] + 1e-7)) -
        torch.atan((a[:, 2] - a[:, 0]) / (a[:, 3] - a[:, 1] + 1e-7)), 2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + 1e-7)
    return iou - (rho2 / c2 + v * alpha)


@MODELS.register
class YOLO26ClsLoss(nn.Module):
    """YOLO26 分类损失: BCEWithLogitsLoss + 软标签 / target_scores.sum()

    与官方 Ultralytics 一致:
    - 输入 logits (未经 sigmoid)
    - 目标为 TAL 产生的软标签 (one_hot × norm_align_metric)
    - 归一化: loss.sum() / target_scores.sum()
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, cls_logits, target_scores):
        loss = self.bce(cls_logits.reshape(-1, cls_logits.shape[-1]),
                        target_scores.reshape(-1, target_scores.shape[-1])).sum()
        tss = target_scores.sum().clamp(min=1)
        return loss / tss


@MODELS.register
class YOLO26BoxLoss(nn.Module):
    """YOLO26 回归损失: 带分数加权的 CIoU Loss

    与官方 Ultralytics 一致:
    - 仅对正样本 (fg_mask) 计算
    - 每个正样本的损失用 target_scores 加权
    - 归一化: sum(1-CIoU * weight) / target_scores.sum()
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, target_bboxes, target_scores, fg_mask):
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        fg_pred = pred_boxes[fg_mask]
        fg_gt   = target_bboxes[fg_mask]
        fg_w    = target_scores[fg_mask].max(dim=-1)[0]
        ciou = _ciou_pairwise(fg_pred, fg_gt)
        tss = target_scores.sum().clamp(min=1)
        return ((1 - ciou) * fg_w).sum() / tss


@MODELS.register
class YOLO26Loss(nn.Module):
    """YOLO26 总损失 (o2m + o2o, progressive loss)

    与官方 E2ELoss 一致:
    - o2m 分支用 TAL topk=10 (或更多)
    - o2o 分支用 TAL topk=1
    - progressive weight: o2m 从 0.8 衰减到 0.1
    - loss = o2m * w + o2o * (1-w)

    Args:
        nc (int): 类别数
        img_size (list): [H, W]
        cls_loss (nn.Module): YOLO26ClsLoss 实例
        box_loss (nn.Module): YOLO26BoxLoss 实例
        assigner_o2m (nn.Module): TAL assigner (topk=10)
        assigner_o2o (nn.Module): TAL assigner (topk=1)
        bbox_coder (nn.Module): bbox 编解码
        strides (list): 各层 stride
        box_gain (float): 回归损失权重 (官方 7.5)
        cls_gain (float): 分类损失权重 (官方 0.5)
        o2m_init (float): o2m 初始权重 (官方 0.8)
        final_o2m (float): o2m 最终权重 (官方 0.1)
        total_epochs (int): 总 epoch 数
    """

    def __init__(self, nc, img_size, cls_loss, box_loss,
                 assigner_o2m, assigner_o2o, bbox_coder,
                 strides=None, box_gain=7.5, cls_gain=0.5,
                 o2m_init=0.8, final_o2m=0.1, total_epochs=300):
        super().__init__()
        self.nc = nc
        self.img_size = img_size
        self.cls_loss_fn = cls_loss
        self.box_loss_fn = box_loss
        self.assigner_o2m = assigner_o2m
        self.assigner_o2o = assigner_o2o
        self.bbox_coder = bbox_coder
        self.strides = strides or [8, 16, 32]
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.o2m_init = o2m_init
        self.final_o2m = final_o2m
        self.total_epochs = total_epochs
        self.updates = 0
        self.o2m_weight = o2m_init

    def _compute_branch_loss(self, preds, batch_bboxes, batch_labels, assigner):
        """计算单个分支 (o2m 或 o2o) 的损失"""
        bs_pred = preds[0].shape[0]
        device = preds[0].device
        nc = self.nc

        cls_logits_list, box_preds_list = [], []
        for i, feat in enumerate(preds):
            stride = self.strides[i]
            h, w = feat.shape[2], feat.shape[3]
            cls_logits = feat[:, 4:, :, :].reshape(bs_pred, nc, -1).permute(0, 2, 1)
            cxywh = self.bbox_coder.decode_single(feat, i)
            boxes = self._feat2xyxy(cxywh) * stride
            cls_logits_list.append(cls_logits)
            box_preds_list.append(boxes.reshape(bs_pred, -1, 4))

        cls_cat = torch.cat(cls_logits_list, dim=1)
        box_cat = torch.cat(box_preds_list, dim=1)

        anc_points, _ = self._make_anchors(preds)

        bs = len(batch_bboxes)
        max_gt = max(len(b) for b in batch_bboxes)
        if max_gt == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        gt_lbl_pad = torch.zeros(bs, max_gt, 1, dtype=torch.long, device=device)
        gt_box_pad = torch.zeros(bs, max_gt, 4, device=device)
        mask_gt = torch.zeros(bs, max_gt, 1, device=device)
        for b_i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
            n = len(bboxes)
            if n == 0: continue
            mask_gt[b_i, :n] = 1
            gt_lbl_pad[b_i, :n, 0] = labels.long()
            b = bboxes.clone()
            gt_box_pad[b_i, :n, 0] = b[:, 0]
            gt_box_pad[b_i, :n, 1] = b[:, 1]
            gt_box_pad[b_i, :n, 2] = b[:, 0] + b[:, 2]
            gt_box_pad[b_i, :n, 3] = b[:, 1] + b[:, 3]

        scores_sig = torch.sigmoid(cls_cat)
        target_labels, target_bboxes, target_scores, fg_mask = assigner(
            scores_sig, box_cat, anc_points,
            gt_lbl_pad, gt_box_pad, mask_gt, nc)

        cls_l = self.cls_loss_fn(cls_cat, target_scores)
        box_l = self.box_loss_fn(box_cat, target_bboxes, target_scores, fg_mask)
        return cls_l, box_l

    def forward(self, o2m_preds, o2o_preds, batch_bboxes, batch_labels):
        """计算 o2m + o2o 双头损失

        Args:
            o2m_preds (list[Tensor]): o2m 分支预测 [bs,5+nc,h_i,w_i]
            o2o_preds (list[Tensor]): o2o 分支预测
            batch_bboxes/batch_labels: GT

        Returns:
            dict: box_loss, cls_loss, dfl_loss
        """
        cls_o2m, box_o2m = self._compute_branch_loss(
            o2m_preds, batch_bboxes, batch_labels, self.assigner_o2m)
        cls_o2o, box_o2o = self._compute_branch_loss(
            o2o_preds, batch_bboxes, batch_labels, self.assigner_o2o)

        o2o_weight = 1.0 - self.o2m_weight

        return {
            'box_loss': box_o2m * self.o2m_weight * self.box_gain +
                        box_o2o * o2o_weight * self.box_gain,
            'cls_loss': cls_o2m * self.o2m_weight * self.cls_gain +
                        cls_o2o * o2o_weight * self.cls_gain,
            'dfl_loss': torch.tensor(0.0, device=cls_o2m.device),
        }

    def update_progressive(self, cur_epoch):
        """Epoch 级别更新 progressive loss 权重

        官方 E2ELoss 逻辑: o2m 从 o2m_init 线性衰减到 final_o2m
        cur_epoch 从 1 开始 (1-indexed)
        """
        progress = max(cur_epoch - 1, 0) / max(self.total_epochs - 1, 1)
        progress = min(progress, 1.0)
        self.o2m_weight = self.o2m_init - progress * (self.o2m_init - self.final_o2m)
        self.o2m_weight = max(self.o2m_weight, self.final_o2m)

    # ---- helpers ----

    def _make_anchors(self, feats):
        anc, s = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            stride = self.strides[i]
            gy, gx = torch.meshgrid(
                torch.arange(h, device=feat.device, dtype=torch.float32),
                torch.arange(w, device=feat.device, dtype=torch.float32),
                indexing='ij')
            anc.append(torch.stack([(gx + 0.5) * stride, (gy + 0.5) * stride], dim=-1).reshape(-1, 2))
            s.append(torch.full((h * w, 1), stride, device=feat.device, dtype=torch.float32))
        return torch.cat(anc, dim=0), torch.cat(s, dim=0)

    def _feat2xyxy(self, cxywh):
        cxywh = cxywh.squeeze(1)
        cx, cy, cw, ch = cxywh[..., 0], cxywh[..., 1], cxywh[..., 2], cxywh[..., 3]
        return torch.stack([cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2], dim=-1)

