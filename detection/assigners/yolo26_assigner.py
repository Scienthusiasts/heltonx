import torch
import torch.nn as nn
from heltonx.utils.register import MODELS


@MODELS.register
class YOLO26Assigner(nn.Module):
    """YOLO26 TaskAlignedAssigner (对齐官方 Ultralytics 实现)

    分配流程:
    1. select_candidates_in_gts: 锚点必须落入 GT 框内 (小 GT 扩展到 >= stride)
    2. get_box_metrics: align_metric = cls^alpha * CIoU^beta
    3. select_topk_candidates: 每 GT 选 topk 个对齐度最高的锚点
    4. mask_pos = mask_topk * mask_in_gts * mask_gt
    5. select_highest_overlaps: 冲突锚点分配给 CIoU 最大的 GT
    6. 归一化目标分数: target_score = one_hot * norm_align_metric (软标签)
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0, stride=None):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-9
        self.stride = stride or [8, 16, 32]
        self.stride_val = self.stride[0]  # 最小 stride, 用于小 GT 扩展

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, nc):
        """TAL 分配

        Args:
            pd_scores (Tensor): [bs, N, nc] sigmoid 分类分数
            pd_bboxes (Tensor): [bs, N, 4] 预测框 xyxy (像素坐标)
            anc_points (Tensor): [N, 2] 锚点中心 (像素)
            gt_labels (Tensor): [bs, n_max, 1]
            gt_bboxes (Tensor): [bs, n_max, 4] GT xyxy (像素)
            mask_gt (Tensor): [bs, n_max, 1] 有效 GT
            nc (int): 类别数

        Returns:
            target_labels (Tensor): [bs, N] 类别索引
            target_bboxes (Tensor): [bs, N, 4] xyxy
            target_scores (Tensor): [bs, N, nc] 软标签
            fg_mask (Tensor): [bs, N]
        """
        bs, N, _ = pd_scores.shape
        device = pd_scores.device
        n_max = gt_bboxes.shape[1]

        if n_max == 0 or mask_gt.sum() == 0:
            return (torch.full((bs, N), nc, dtype=torch.long, device=device),
                    torch.zeros(bs, N, 4, device=device),
                    torch.zeros(bs, N, nc, device=device),
                    torch.zeros(bs, N, dtype=torch.bool, device=device))

        # 1. 锚点是否在 GT 内 (含小 GT 扩展)
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        # [bs, n_max, N]

        # 2. 对齐度量 (cls^alpha * CIoU^beta)  与 CIoU
        mask_for_metric = mask_in_gts * mask_gt
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_for_metric)
        # both [bs, n_max, N]

        # 3. Top-K 选择
        mask_topk = self.select_topk_candidates(align_metric, mask_gt, mask_in_gts)
        # [bs, n_max, N]

        # 4. mask_pos
        mask_pos = mask_topk * mask_in_gts * mask_gt

        # 5. 冲突解决
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, n_max)

        # 6. 生成目标 (含官方归一化)
        return self.build_targets(
            target_gt_idx, fg_mask, mask_pos, align_metric, overlaps,
            gt_labels, gt_bboxes, mask_gt, nc, device)

    # --------------- helpers ---------------

    def select_candidates_in_gts(self, anc_points, gt_bboxes, mask_gt):
        """锚点中心是否在 GT 框内 [bs, n_max, N]

        与官方一致: 将 w/h < stride_val 的 GT 扩展到至少 stride_val,
        确保小物体也能分配到正样本锚点.
        """
        bs, n_max, _ = gt_bboxes.shape
        N = anc_points.shape[0]

        # 官方逻辑: 小 GT 扩展 (xyxy → xywh → 扩展 → xyxy)
        gt_cx = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
        gt_cy = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
        gt_w = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        gt_h = gt_bboxes[..., 3] - gt_bboxes[..., 1]

        # 小 GT 扩展: w 或 h < stride_val 时, 扩展到 stride_val
        wh_mask = (gt_w < self.stride_val) | (gt_h < self.stride_val)
        wh_mask = wh_mask & (mask_gt.squeeze(-1) > 0)
        gt_w = torch.where(wh_mask, torch.tensor(self.stride_val, dtype=gt_w.dtype, device=gt_w.device), gt_w)
        gt_h = torch.where(wh_mask, torch.tensor(self.stride_val, dtype=gt_h.dtype, device=gt_h.device), gt_h)

        # 重建 xyxy
        gt_x1 = gt_cx - gt_w / 2
        gt_y1 = gt_cy - gt_h / 2
        gt_x2 = gt_cx + gt_w / 2
        gt_y2 = gt_cy + gt_h / 2

        anc_x = anc_points[:, 0].view(1, 1, N)
        anc_y = anc_points[:, 1].view(1, 1, N)

        left   = anc_x - gt_x1.unsqueeze(-1)
        top    = anc_y - gt_y1.unsqueeze(-1)
        right  = gt_x2.unsqueeze(-1) - anc_x
        bottom = gt_y2.unsqueeze(-1) - anc_y

        bbox_deltas = torch.stack([left, top, right, bottom], dim=-1)
        return bbox_deltas.min(dim=-1)[0] > self.eps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """align_metric = cls_score^alpha * CIoU^beta"""
        bs, n_max, N = mask_gt.shape

        # cls_score: 取出每个 GT 对应类别 [bs, n_max, N]
        cls_score = pd_scores.transpose(1, 2).gather(
            1, gt_labels.squeeze(-1).unsqueeze(-1).expand(-1, -1, N)
        ).squeeze(1) * mask_gt.float()

        # CIoU [bs, n_max, N]
        overlaps = self._ciou_matrix(pd_bboxes, gt_bboxes) * mask_gt.float()

        align_metric = (cls_score ** self.alpha) * (overlaps ** self.beta)
        return align_metric, overlaps

    def _ciou_matrix(self, pd_bboxes, gt_bboxes):
        """CIoU 矩阵 [bs, N, 4] × [bs, n_max, 4] -> [bs, n_max, N]"""
        bs, N, _ = pd_bboxes.shape
        n_max = gt_bboxes.shape[1]

        a = pd_bboxes.unsqueeze(1)  # [bs, 1, N, 4]
        b = gt_bboxes.unsqueeze(2)  # [bs, n_max, 1, 4]

        lt = torch.max(a[..., :2], b[..., :2])
        rb = torch.min(a[..., 2:], b[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]

        area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
        area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        union  = area_a + area_b - inter + self.eps
        iou    = inter / union

        cw = torch.max(a[..., 2], b[..., 2]) - torch.min(a[..., 0], b[..., 0])
        ch = torch.max(a[..., 3], b[..., 3]) - torch.min(a[..., 1], b[..., 1])
        c2 = cw ** 2 + ch ** 2 + self.eps
        rho2 = ((b[..., 0] + b[..., 2] - a[..., 0] - a[..., 2]) ** 2 +
                (b[..., 1] + b[..., 3] - a[..., 1] - a[..., 3]) ** 2) / 4
        w_a, h_a = a[..., 2] - a[..., 0], a[..., 3] - a[..., 1]
        w_b, h_b = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(w_b / (h_b + self.eps)) -
            torch.atan(w_a / (h_a + self.eps)), 2)
        with torch.no_grad():
            alpha = v / (v - iou + 1 + self.eps)
        return (iou - (rho2 / c2 + v * alpha)).clamp(min=0)

    def select_topk_candidates(self, metrics, mask_gt, topk_mask=None):
        """每 GT 选 topk 个最高度量锚点 [bs, n_max, N]"""
        bs, n_max, N = metrics.shape
        valid = mask_gt.squeeze(-1).bool().unsqueeze(-1)  # [bs, n_max, 1]
        m = metrics * valid.float()
        if topk_mask is not None:
            m = m * topk_mask.float()

        _, idx = torch.topk(m, min(self.topk, N), dim=-1)  # [bs, n_max, topk]
        mask_topk = torch.zeros_like(m, dtype=torch.bool)
        mask_topk.scatter_(-1, idx, True)
        # 与官方一致: 检查每个 GT 的 topk 最大值是否 > eps (而非每个元素)
        topk_max = m.gather(-1, idx).max(dim=-1)[0]  # [bs, n_max]
        valid_gt = topk_max > self.eps  # [bs, n_max]
        return mask_topk * valid_gt.unsqueeze(-1)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes):
        """冲突解决: 多 GT 竞争的锚点只保留 CIoU 最大的 GT

        与官方一致: 返回 target_gt_idx, fg_mask, mask_pos
        使用向量化操作替代循环
        """
        fg_mask = mask_pos.sum(dim=1)  # [bs, N]

        if fg_mask.max() > 1:
            # 标记冲突锚点
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            # [bs, n_max, N]

            # 每个 anchor 选 overlap 最大的 GT
            max_overlaps_idx = overlaps.argmax(dim=1)  # [bs, N]
            is_max_overlaps = torch.zeros_like(mask_pos)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            # 冲突锚点: 只保留 max overlap GT; 非冲突: 保持原样
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps > 0, mask_pos > 0).float()
            fg_mask = mask_pos.sum(dim=1)

        target_gt_idx = mask_pos.argmax(dim=1)  # [bs, N]
        return target_gt_idx, fg_mask > 0, mask_pos

    def build_targets(self, target_gt_idx, fg_mask, mask_pos, align_metric,
                      overlaps, gt_labels, gt_bboxes, mask_gt, nc, device):
        """构建训练目标 (含官方归一化)

        与官方 Ultralytics 一致的 norm_align 计算方式:
        1. pos_align_metrics = align_metric.amax(dim=-1)  # 每 GT 最大 align
        2. pos_overlaps = (overlaps * mask_pos).amax(dim=-1)  # 每 GT 最大 overlap
        3. norm_align_metric = (align_metric * pos_overlaps / pos_align_metrics).amax(-2)
        4. target_scores = one_hot * norm_align_metric
        """
        bs, n_max, N = mask_pos.shape

        # ---- 官方归一化 ----
        align_metric_pos = align_metric * mask_pos  # 非正样本位置置零
        pos_align_metrics = align_metric_pos.amax(dim=-1, keepdim=True)  # [bs, n_max, 1]
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # [bs, n_max, 1]
        norm_align_metric = (align_metric_pos * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # [bs, N, 1]

        # 构建输出
        target_labels = torch.full((bs, N), nc, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(bs, N, 4, device=device)
        target_scores = torch.zeros(bs, N, nc, device=device)

        for b in range(bs):
            fg = fg_mask[b]
            if fg.sum() == 0:
                continue
            gt_idx = target_gt_idx[b, fg]  # [num_fg]
            target_labels[b, fg] = gt_labels[b, gt_idx, 0].long()
            target_bboxes[b, fg] = gt_bboxes[b, gt_idx]

            # 软标签: one_hot * norm_align_metric
            oh = torch.eye(nc, device=device)[gt_labels[b, gt_idx, 0].long()]  # [num_fg, nc]
            target_scores[b, fg] = oh * norm_align_metric[b, fg]

        return target_labels, target_bboxes, target_scores, fg_mask
