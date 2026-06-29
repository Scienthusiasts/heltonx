import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from heltonx.utils.register import MODELS
from detectionobb.utils.obb_iou import batch_probiou
from detectionobb.utils.obb_ops import rbox2dist


@MODELS.register
class YOLO26OBBClsLoss(nn.Module):
    """OBB 分类损失: BCEWithLogitsLoss + 软标签

    ★ 与官方 v8OBBLoss 完全一致: 无 pos_weight, reduction='none'
    官方归约方式: .sum() / target_scores.sum()
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, cls_logits, target_scores):
        # ★ clamp 防止 logits 过大导致 BCE loss = INF (sigmoid 饱和)
        cls_logits = torch.nan_to_num(cls_logits, nan=0.0).clamp(-15, 15)
        loss = self.bce(cls_logits.reshape(-1, cls_logits.shape[-1]),
                        target_scores.reshape(-1, target_scores.shape[-1])).sum()
        tss = target_scores.sum().clamp(min=1)
        return loss / tss


@MODELS.register
class YOLO26OBBBoxLoss(nn.Module):
    """OBB 回归损失: 带分数加权的 probiou Loss

    ★ 与官方 v8OBBLoss 一致: 使用 probiou (概率 IoU)
    - 输入为 xywhr (5维) anchor 空间
    - weight = target_scores.sum(-1)[fg_mask]
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, target_bboxes, target_scores, fg_mask):
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        fg_pred = pred_boxes[fg_mask]      # [num_fg, 5] xywhr
        fg_gt = target_bboxes[fg_mask]      # [num_fg, 5] xywhr
        fg_w = target_scores[fg_mask].max(dim=-1)[0]

        # probiou 逐对计算 (不使用 regularize, 与官方一致)
        iou = batch_probiou(fg_pred, fg_gt).diag()
        tss = target_scores.sum().clamp(min=1)
        return ((1 - iou) * fg_w).sum() / tss


@MODELS.register
class YOLO26OBBL1Loss(nn.Module):
    """OBB L1 回归损失 (与官方 YOLO26 reg_max=1 一致, 无 DFL)

    ★★★ 与官方 RotatedBboxLoss (reg_max=1 分支) 完全一致:
    - 将 pred_dist 和 target_ltrb 都乘 stride 并除以 imgsz 归一化
    - 计算 L1 loss: F.l1_loss(pred, target, reduction='none').mean(-1, keepdim=True) * weight

    注意: 官方代码中此损失名为 loss_dfl, 但当 reg_max=1 时实际是 L1 回归损失而非 DFL.
    我们命名为 L1Loss 以避免混淆.

    官方代码:
        target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss = F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction='none').mean(-1, keepdim=True) * weight
        loss = loss.sum() / target_scores_sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_dist, target_bboxes_xywh, anchor_points, target_angle,
                target_scores, fg_mask, imgsz, stride_ts):
        """计算 L1 回归损失

        Args:
            pred_dist (Tensor): [bs, N, 4] 原始 ltrb 距离 (anchor 空间)
            target_bboxes_xywh (Tensor): [bs, N, 4] GT xywh (anchor 空间, grid 坐标)
            anchor_points (Tensor): [N, 2] 锚点 grid 坐标 (anchor 空间)
            target_angle (Tensor): [bs, N, 1] GT 角度 (弧度)
            target_scores (Tensor): [bs, N, nc] 目标分数
            fg_mask (Tensor): [bs, N] 前景掩码
            imgsz (Tensor): [2] 图像尺寸 [W, H] (与官方 imgsz[1]=W, imgsz[0]=H)
            stride_ts (Tensor): [1, N, 1] stride 张量

        Returns:
            Tensor: L1 回归损失
        """
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_dist.device)

        # ★ GT → ltrb: rbox2dist (旋转框到距离, 与官方一致)
        target_ltrb = rbox2dist(anchor_points, target_bboxes_xywh, target_angle, dim=-1)

        # ★ 归一化: 与官方完全一致 (stride * ltrb / imgsz)
        # 官方: imgsz = [H, W], imgsz[1]=W 用于 x(l,r), imgsz[0]=H 用于 y(t,b)
        # 我们: imgsz_ts = [W, H], imgsz_ts[0]=W 用于 x(l,r), imgsz_ts[1]=H 用于 y(t,b)
        target_ltrb = target_ltrb * stride_ts  # [bs, N, 4] * [1, N, 1]
        target_ltrb[..., 0::2] /= imgsz[0]     # l,r (x方向) 除以 W (= imgsz_ts[0])
        target_ltrb[..., 1::2] /= imgsz[1]     # t,b (y方向) 除以 H (= imgsz_ts[1])

        pred_dist_n = pred_dist * stride_ts
        pred_dist_n[..., 0::2] /= imgsz[0]  # l,r (x方向) 除以 W
        pred_dist_n[..., 1::2] /= imgsz[1]  # t,b (y方向) 除以 H

        # ★ L1 loss (与官方一致)
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # [num_fg, 1]
        tss = target_scores.sum().clamp(min=1)

        loss = F.l1_loss(pred_dist_n[fg_mask], target_ltrb[fg_mask],
                         reduction='none').mean(-1, keepdim=True) * weight
        return loss.sum() / tss


@MODELS.register
class YOLO26AngleLoss(nn.Module):
    """OBB 角度损失: sin²(2Δθ) + 宽高比自适应权重

    ★★★ 与官方 v8OBBLoss.calculate_angle_loss 完全一致:
    1. delta_theta_wrapped = delta_theta - round(delta_theta / π) * π
    2. ang_loss = sin²(2 * delta_theta_wrapped)
    3. scale_weight = exp(-(log_ar²) / λ²), λ=3
    4. ang_loss = scale_weight * ang_loss * weight / target_scores_sum

    ★ OBB26 角度为原始 logits (弧度), 无 sigmoid 变换
    - pred_theta 直接为网络输出的弧度值
    - target_theta 为 GT 弧度值
    """
    def __init__(self, lambda_val=3):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum):
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_bboxes.device)

        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        # 宽高比自适应权重
        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar ** 2) / (self.lambda_val ** 2))

        # ★ 与官方 v8OBBLoss.calculate_angle_loss 完全一致:
        # delta_theta_wrapped = delta_theta - round(delta_theta / π) * π
        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight[fg_mask]

        return ang_loss.sum() / target_scores_sum


@MODELS.register
class YOLO26OBBLoss(nn.Module):
    """YOLO26-OBB 总损失 (o2m + o2o, progressive loss)

    ★★★ 与官方 YOLO26 OBB 完全一致的回归方式:
    - BBox: 直接回归 ltrb 距离 (无 sigmoid, 无 DFL), 使用 dist2rbox 解码
    - Angle: 直接输出原始 logits (弧度, 无 sigmoid 变换)
    - 4 项损失: box(probiou) + cls(BCE) + l1(L1回归) + angle(sin²(2Δθ_wrapped))

    与旧版 YOLO26OBBLoss 的区别:
    - bbox 解码: sigmoid → ltrb 直接回归 + dist2rbox
    - angle 解码: sigmoid 变换 → 原始 logits
    - l1_loss: 硬编码 0 → L1 回归损失 (stride/imgsz 归一化)

    Args:
        nc (int): 类别数
        img_size (list): [H, W]
        cls_loss (nn.Module): YOLO26OBBClsLoss 实例
        box_loss (nn.Module): YOLO26OBBBoxLoss 实例
        l1_loss (nn.Module): YOLO26OBBL1Loss 实例 (L1 回归损失)
        angle_loss (nn.Module): YOLO26AngleLoss 实例
        assigner_o2m (nn.Module): OBB TAL assigner (topk=10)
        assigner_o2o (nn.Module): OBB TAL assigner (topk=1)
        bbox_coder (nn.Module): OBB bbox 编解码
        strides (list): 各层 stride
        box_gain (float): 回归损失权重
        cls_gain (float): 分类损失权重
        l1_gain (float): L1 回归损失权重
        angle_gain (float): 角度损失权重
        o2m_init (float): o2m 初始权重
        final_o2m (float): o2m 最终权重
        total_epochs (int): 总 epoch 数
    """

    def __init__(self, nc, img_size, cls_loss, box_loss, l1_loss, angle_loss,
                 assigner_o2m, assigner_o2o, bbox_coder,
                 strides=None, box_gain=7.5, cls_gain=0.5, l1_gain=1.5, angle_gain=1.0,
                 o2m_init=0.8, final_o2m=0.1, total_epochs=300):
        super().__init__()
        self.nc = nc
        self.img_size = img_size
        self.cls_loss_fn = cls_loss
        self.box_loss_fn = box_loss
        self.l1_loss_fn = l1_loss
        self.angle_loss_fn = angle_loss
        self.assigner_o2m = assigner_o2m
        self.assigner_o2o = assigner_o2o
        self.bbox_coder = bbox_coder
        self.strides = strides or [8, 16, 32]
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.l1_gain = l1_gain
        self.angle_gain = angle_gain
        self.o2m_init = o2m_init
        self.final_o2m = final_o2m
        self.total_epochs = total_epochs
        self.updates = 0
        self.o2m_weight = o2m_init

    def _compute_branch_loss(self, preds, batch_bboxes, batch_labels, assigner):
        """计算单个分支 (o2m 或 o2o) 的损失

        ★★★ 与官方 v8OBBLoss.loss() 完全一致的流程:
        1. 提取 ltrb + angle logits (无 sigmoid)
        2. bbox_decode(anchor_points, pred_dist, pred_angle) → xywhr (anchor 空间)
        3. bboxes_for_assigner = pred_bboxes.clone().detach(); *= stride → 像素空间
        4. assigner 在像素空间匹配
        5. target_bboxes[..., :4] /= stride → anchor 空间
        6. box_loss: probiou (anchor 空间)
        7. l1_loss: L1 (stride/imgsz 归一化)
        8. angle_loss: sin²(2Δθ_wrapped) (anchor 空间, 角度不变)

        Args:
            preds: dict {'box': [bs,4,h_i,w_i], 'cls': [bs,nc,h_i,w_i], 'angle': [bs,1,h_i,w_i]}

        Returns:
            cls_l, box_l, l1_l, angle_l
        """
        bs_pred = preds['box'][0].shape[0]
        device = preds['box'][0].device
        nc = self.nc

        # ★ 提取 ltrb 距离 + angle logits + cls logits (从分离的 head 输出)
        pred_dist_list, pred_angle_list, cls_logits_list = [], [], []
        for i in range(len(preds['box'])):
            box_feat = preds['box'][i]      # [bs, 4, h, w]
            cls_feat = preds['cls'][i]      # [bs, nc, h, w]
            angle_feat = preds['angle'][i]  # [bs, 1, h, w]

            box_feat = torch.nan_to_num(box_feat, nan=0.0, posinf=1e4, neginf=-1e4)
            cls_feat = torch.nan_to_num(cls_feat, nan=0.0, posinf=1e4, neginf=-1e4)
            angle_feat = torch.nan_to_num(angle_feat, nan=0.0, posinf=1e4, neginf=-1e4)

            h, w = box_feat.shape[2], box_feat.shape[3]
            N = h * w

            # ltrb: [bs, 4, N] → [bs, N, 4]
            pred_dist = box_feat.reshape(bs_pred, 4, N).permute(0, 2, 1)
            # angle: [bs, 1, N] → [bs, N, 1]
            pred_angle = angle_feat.reshape(bs_pred, 1, N).permute(0, 2, 1)
            # cls: [bs, nc, N] → [bs, N, nc]
            cls_logits = cls_feat.reshape(bs_pred, nc, N).permute(0, 2, 1)

            pred_dist_list.append(pred_dist)
            pred_angle_list.append(pred_angle)
            cls_logits_list.append(cls_logits)

        cls_cat = torch.cat(cls_logits_list, dim=1)        # [bs, N, nc]
        pred_dist_cat = torch.cat(pred_dist_list, dim=1)    # [bs, N, 4] ltrb
        pred_angle_cat = torch.cat(pred_angle_list, dim=1)  # [bs, N, 1] 角度

        # ★ 构建锚点和 stride 张量
        anc_points, stride_tensor = self._make_anchors(preds['box'])

        bs = len(batch_bboxes)
        max_gt = max(len(b) for b in batch_bboxes)
        if max_gt == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero, zero

        # GT padding: xywhr (5维) 像素坐标
        gt_lbl_pad = torch.zeros(bs, max_gt, 1, dtype=torch.long, device=device)
        gt_box_pad = torch.zeros(bs, max_gt, 5, device=device)
        mask_gt = torch.zeros(bs, max_gt, 1, device=device)
        for b_i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
            n = len(bboxes)
            if n == 0:
                continue
            bboxes = bboxes.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            if bboxes.isnan().any() or bboxes.isinf().any():
                valid = ~(bboxes.isnan().any(dim=-1) | bboxes.isinf().any(dim=-1))
                bboxes = bboxes[valid]
                labels = labels[valid]
            if len(bboxes) > 0:
                small_mask = (bboxes[:, 2] < 2) | (bboxes[:, 3] < 2)
                if small_mask.any():
                    bboxes = bboxes[~small_mask]
                    labels = labels[~small_mask]
            n = len(bboxes)
            if n == 0:
                continue
            mask_gt[b_i, :n] = 1
            lbl = labels.long()
            if lbl.max() >= nc:
                lbl = lbl.clamp(0, nc - 1)
            if lbl.min() < 0:
                lbl = lbl.clamp(0, nc - 1)
            gt_lbl_pad[b_i, :n, 0] = lbl
            gt_box_pad[b_i, :n] = bboxes

        # ★★★ 小 GT 扩展 (与官方 v8OBBLoss 一致):
        # w 或 h < stride[0] (=8) 时扩展到 stride_val (=16)
        # 扩展后的 GT 同时用于 assigner 匹配、probiou loss 和 L1 loss target
        wh = gt_box_pad[..., 2:4]
        stride_min = self.assigner_o2m.stride[0]  # threshold = 8
        stride_expand_val = self.assigner_o2m.stride_val  # expand target = 16
        wh.masked_fill_((wh < stride_min) & mask_gt.bool(), stride_expand_val)

        # ★★★ bbox_decode: ltrb + angle → xywhr (anchor 空间)
        # 与官方 v8OBBLoss.bbox_decode 一致 (anc_points 为 grid 坐标)
        pred_bboxes = self.bbox_coder.bbox_decode(
            anc_points, pred_dist_cat, pred_angle_cat
        )  # [bs, N, 5] xywhr anchor 空间

        # ★ assigner 在像素空间操作 (与官方一致: anchor_points * stride)
        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor  # anchor → 像素空间

        # ★ scores_sig 必须 detach (防止计算图泄漏)
        scores_sig = torch.sigmoid(cls_cat).detach()

        # ★ 与官方一致: anchor_points * stride_tensor → 像素空间
        anc_points_px = anc_points * stride_tensor  # grid → 像素空间

        _, target_bboxes, target_scores, fg_mask = assigner(
            scores_sig, bboxes_for_assigner, anc_points_px,
            gt_lbl_pad, gt_box_pad, mask_gt, nc)

        # ★ 与官方一致: target_bboxes /= stride → anchor 空间
        stride_ts = stride_tensor.squeeze(-1).unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        target_bboxes_anchor = target_bboxes.clone()
        target_bboxes_anchor[..., :4] /= stride_ts  # GT 像素 → anchor 空间

        # 图像尺寸张量 (与官方 imgsz 一致)
        imgsz = torch.tensor(self.img_size, device=device, dtype=torch.float32)  # [H, W]
        imgsz_ts = imgsz[[1, 0]]  # [W, H] (与官方 imgsz[1]=W, imgsz[0]=H 一致)

        # ★ 分类损失
        cls_l = self.cls_loss_fn(cls_cat, target_scores)

        # ★ probiou 回归损失 (anchor 空间)
        box_l = self.box_loss_fn(pred_bboxes, target_bboxes_anchor, target_scores, fg_mask)

        # ★ L1 回归损失 (stride/imgsz 归一化)
        # pred_dist_cat: [bs, N, 4] 原始 ltrb (anchor 空间)
        # target_bboxes_anchor[..., :4]: [bs, N, 4] GT xywh (anchor 空间)
        # target_bboxes_anchor[..., 4:5]: [bs, N, 1] GT angle (anchor 空间, 角度不变)
        l1_l = self.l1_loss_fn(
            pred_dist_cat,
            target_bboxes_anchor[..., :4], anc_points,
            target_bboxes_anchor[..., 4:5],
            target_scores, fg_mask, imgsz_ts, stride_ts
        )

        # ★ 角度损失 (anchor 空间, 角度不变)
        weight = target_scores.sum(-1)  # [bs, N]
        tss = target_scores.sum().clamp(min=1)
        angle_l = self.angle_loss_fn(pred_bboxes, target_bboxes_anchor, fg_mask, weight, tss)

        return cls_l, box_l, l1_l, angle_l

    def forward(self, o2m_preds, o2o_preds, batch_bboxes, batch_labels):
        """计算 o2m + o2o 双头损失

        Args:
            o2m_preds: dict {'box': [...], 'cls': [...], 'angle': [...]}
            o2o_preds: dict {'box': [...], 'cls': [...], 'angle': [...]}
            batch_bboxes: GT xywhr (5维) 列表
            batch_labels: GT 类别 列表

        Returns:
            dict: box_loss, cls_loss, l1_loss, angle_loss
        """
        cls_o2m, box_o2m, l1_o2m, ang_o2m = self._compute_branch_loss(
            o2m_preds, batch_bboxes, batch_labels, self.assigner_o2m)
        cls_o2o, box_o2o, l1_o2o, ang_o2o = self._compute_branch_loss(
            o2o_preds, batch_bboxes, batch_labels, self.assigner_o2o)

        o2o_weight = 1.0 - self.o2m_weight
        bs = o2m_preds['box'][0].shape[0]

        return {
            'box_loss': (box_o2m * self.o2m_weight + box_o2o * o2o_weight) * self.box_gain * bs,
            'cls_loss': (cls_o2m * self.o2m_weight + cls_o2o * o2o_weight) * self.cls_gain * bs,
            'l1_loss': (l1_o2m * self.o2m_weight + l1_o2o * o2o_weight) * self.l1_gain * bs,
            'angle_loss': (ang_o2m * self.o2m_weight + ang_o2o * o2o_weight) * self.angle_gain * bs,
        }

    def update_progressive(self, cur_epoch):
        """Epoch 级别更新 progressive loss 权重"""
        progress = max(cur_epoch - 1, 0) / max(self.total_epochs - 1, 1)
        progress = min(progress, 1.0)
        self.o2m_weight = self.o2m_init - progress * (self.o2m_init - self.final_o2m)
        self.o2m_weight = max(self.o2m_weight, self.final_o2m)

    # ---- helpers ----

    def _make_anchors(self, feats):
        """构建锚点 (与官方 make_anchors 一致: grid 坐标 + stride)

        ★★★ 与官方完全一致:
        - anchor_points = grid 坐标 (0.5, 1.5, 2.5, ...) 不乘 stride
        - stride_tensor = 各层 stride 值

        注意: 旧版返回像素空间坐标 (乘了 stride), 新版改为 grid 坐标
        这是因为 dist2rbox/rbox2dist 需要 anchor 空间坐标
        """
        anc, s = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            stride = self.strides[i]
            gy, gx = torch.meshgrid(
                torch.arange(h, device=feat.device, dtype=torch.float32),
                torch.arange(w, device=feat.device, dtype=torch.float32),
                indexing='ij')
            # ★ grid 坐标 (与官方一致: 不乘 stride)
            anc.append(torch.stack([gx + 0.5, gy + 0.5], dim=-1).reshape(-1, 2))
            s.append(torch.full((h * w, 1), stride, device=feat.device, dtype=torch.float32))
        return torch.cat(anc, dim=0), torch.cat(s, dim=0)
