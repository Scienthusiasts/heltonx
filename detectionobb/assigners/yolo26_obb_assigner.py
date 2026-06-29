import torch
import torch.nn as nn
from heltonx.utils.register import MODELS
from detectionobb.utils.obb_iou import batch_probiou
from detectionobb.utils.obb_ops import xywhr2xyxyxyxy


@MODELS.register
class YOLO26OBBAssigner(nn.Module):
    """YOLO26-OBB 旋转框 TaskAlignedAssigner

    与 YOLO26Assigner 的区别:
    - IoU 计算使用 batch_probiou (概率 IoU) 替代 CIoU
    - GT 框格式为 xywhr (5维) 而非 xyxy (4维)
    - select_candidates_in_gts: 使用叉积投影法判断锚点是否在旋转矩形内

    分配流程与 YOLO26Assigner 一致:
    1. select_candidates_in_gts: 锚点落入 GT 旋转框内 (叉积投影法)
    2. get_box_metrics: align_metric = cls^alpha * probiou^beta
    3. select_topk_candidates: 每 GT 选 topk 个对齐度最高的锚点
    4. mask_pos = mask_topk * mask_in_gts * mask_gt
    5. select_highest_overlaps: 冲突锚点分配给 probiou 最大的 GT
    6. 归一化目标分数: target_score = one_hot * norm_align_metric
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0, stride=None):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-9
        self.stride = stride or [8, 16, 32]
        self.stride_val = self.stride[1]  # ★ 与官方 RotatedTaskAlignedAssigner 一致: stride_val=stride[1]

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, nc):
        """TAL 旋转框分配

        Args:
            pd_scores (Tensor): [bs, N, nc] sigmoid 分类分数
            pd_bboxes (Tensor): [bs, N, 5] 预测框 xywhr (像素坐标)
            anc_points (Tensor): [N, 2] 锚点中心 (像素)
            gt_labels (Tensor): [bs, n_max, 1]
            gt_bboxes (Tensor): [bs, n_max, 5] GT xywhr (像素)
            mask_gt (Tensor): [bs, n_max, 1] 有效 GT
            nc (int): 类别数

        Returns:
            target_labels (Tensor): [bs, N] 类别索引
            target_bboxes (Tensor): [bs, N, 5] xywhr
            target_scores (Tensor): [bs, N, nc] 软标签
            fg_mask (Tensor): [bs, N]
        """
        bs, N, _ = pd_scores.shape
        device = pd_scores.device
        n_max = gt_bboxes.shape[1]

        if n_max == 0 or mask_gt.sum() == 0:
            return (torch.full((bs, N), nc, dtype=torch.long, device=device),
                    torch.zeros(bs, N, 5, device=device),
                    torch.zeros(bs, N, nc, device=device),
                    torch.zeros(bs, N, dtype=torch.bool, device=device))

        # 1. 锚点是否在 GT 外接矩形内
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)

        # 2. 对齐度量 (cls^alpha * probiou^beta)
        mask_for_metric = mask_in_gts * mask_gt
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_for_metric)

        # 3. Top-K 选择 (不传 mask_in_gts, 函数内部根据 topk_metrics > eps 自动计算)
        mask_topk = self.select_topk_candidates(align_metric, mask_gt)

        # 4. mask_pos
        mask_pos = mask_topk * mask_in_gts * mask_gt

        # 5. 冲突解决
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, n_max)

        # 6. 生成目标
        return self.build_targets(
            target_gt_idx, fg_mask, mask_pos, align_metric, overlaps,
            gt_labels, gt_bboxes, mask_gt, nc, device)

    # --------------- helpers ---------------

    def select_candidates_in_gts(self, anc_points, gt_bboxes, mask_gt):
        """锚点中心是否在 GT 旋转框内 [bs, n_max, N]

        使用叉积投影法 (与官方 RotatedTaskAlignedAssigner 一致):
        1. 将 xywhr 转为 4 个顶点 xywhr2xyxyxyxy
        2. 取相邻两个顶点构成边向量 ab, ad
        3. 计算 ap (锚点到角点a的向量) 在 ab, ad 上的投影
        4. 判断投影是否在 [0, |ab|] 和 [0, |ad|] 范围内
        """
        bs, n_max, _ = gt_bboxes.shape
        N = anc_points.shape[0]

        # 小 GT 扩展: w 或 h < stride[0] 时扩展到 stride_val (与官方一致)
        # ★ 官方: threshold=stride[0]=8, expand_target=stride_val=stride[1]=16
        gt_bboxes_clone = gt_bboxes.clone()
        wh_mask = gt_bboxes_clone[..., 2:4] < self.stride[0]
        gt_bboxes_clone[..., 2:4] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_clone.dtype, device=gt_bboxes_clone.device),
            gt_bboxes_clone[..., 2:4],
        )

        # (bs, n_max, 5) → (bs, n_max, 4, 2) 四个顶点
        corners = xywhr2xyxyxyxy(gt_bboxes_clone)

        # 取角点: a=corners[:,0], b=corners[:,1], d=corners[:,3]
        a, b, _, d = corners.split(1, dim=-2)  # 各 (bs, n_max, 1, 2)

        # 边向量
        ab = b - a  # (bs, n_max, 1, 2)
        ad = d - a  # (bs, n_max, 1, 2)

        # 锚点到角点 a 的向量 (bs, n_max, N, 2)
        ap = anc_points.unsqueeze(0).unsqueeze(0) - a  # (N,2) → (1,1,N,2) - (bs,n_max,1,2)

        # 投影计算
        norm_ab = (ab * ab).sum(dim=-1)  # (bs, n_max, 1)
        norm_ad = (ad * ad).sum(dim=-1)  # (bs, n_max, 1)
        ap_dot_ab = (ap * ab).sum(dim=-1)  # (bs, n_max, N)
        ap_dot_ad = (ap * ad).sum(dim=-1)  # (bs, n_max, N)

        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """align_metric = cls_score^alpha * probiou^beta"""
        bs, n_max, N = mask_gt.shape
        nc = pd_scores.shape[-1]

        # gather 索引: clamp gt_labels 到 [0, nc-1] 防止越界
        gather_idx = gt_labels.squeeze(-1).clamp(0, nc - 1).unsqueeze(-1).expand(-1, -1, N)

        # cls_score: 取出每个 GT 对应类别 [bs, n_max, N]
        # ★ 不用 squeeze(1)！当 n_max=1 时 squeeze 会去掉真实维度，
        #   导致后续 cls_score * mask_gt.float() 广播错误 (变成 [bs, bs, N])
        cls_score = pd_scores.transpose(1, 2).gather(
            1, gather_idx
        ) * mask_gt.float()

        # probiou [bs, n_max, N]
        overlaps = self._probiou_matrix(pd_bboxes, gt_bboxes) * mask_gt.float()

        align_metric = (cls_score ** self.alpha) * (overlaps ** self.beta)
        return align_metric, overlaps

    def _probiou_matrix(self, pd_bboxes, gt_bboxes):
        """probiou 矩阵 [bs, N, 5] × [bs, n_max, 5] -> [bs, n_max, N]

        与官方 RotatedTaskAlignedAssigner 一致: 不使用 regularize_rboxes,
        probiou 通过协方差矩阵自然处理 (w,h,θ) 和 (h,w,θ+π/2) 的等价性
        """
        bs, N, _ = pd_bboxes.shape
        n_max = gt_bboxes.shape[1]

        overlaps = torch.zeros(bs, n_max, N, device=pd_bboxes.device, dtype=pd_bboxes.dtype)
        for b in range(bs):
            if N > 0 and n_max > 0:
                overlaps[b] = batch_probiou(gt_bboxes[b], pd_bboxes[b])
        return overlaps

    def select_topk_candidates(self, metrics, mask_gt, topk_mask=None):
        """每 GT 选 topk 个最高度量锚点 [bs, n_max, N]

        ★ 与官方 select_topk_candidates 一致:
        - 被多个 GT 同时选中的锚点排除 (count > 1 → 0)
        """
        bs, n_max, N = metrics.shape
        valid = mask_gt.squeeze(-1).bool().unsqueeze(-1)
        m = metrics * valid.float()
        if topk_mask is not None:
            m = m * topk_mask.float()

        topk = min(self.topk, N)
        topk_metrics, topk_idxs = torch.topk(m, topk, dim=-1, largest=True)

        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # ★ 与官方一致: 用 scatter_add 计数, count > 1 的锚点排除
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k+1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes):
        """冲突解决: 多 GT 竞争的锚点只保留 probiou 最大的 GT"""
        fg_mask = mask_pos.sum(dim=1)

        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(dim=1)
            is_max_overlaps = torch.zeros_like(mask_pos)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps > 0, mask_pos > 0).float()
            fg_mask = mask_pos.sum(dim=1)

        target_gt_idx = mask_pos.argmax(dim=1)
        return target_gt_idx, fg_mask > 0, mask_pos

    def build_targets(self, target_gt_idx, fg_mask, mask_pos, align_metric,
                      overlaps, gt_labels, gt_bboxes, mask_gt, nc, device):
        """构建训练目标 (含官方归一化)"""
        bs, n_max, N = mask_pos.shape

        align_metric_pos = align_metric * mask_pos
        pos_align_metrics = align_metric_pos.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric_pos * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)

        target_labels = torch.full((bs, N), nc, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(bs, N, 5, device=device)
        target_scores = torch.zeros(bs, N, nc, device=device)

        for b in range(bs):
            fg = fg_mask[b]
            if fg.sum() == 0:
                continue
            gt_idx = target_gt_idx[b, fg]

            # gt_labels/gt_bboxes 的 dim1 应与 mask_pos 的 n_max 一致
            gt_lbl_vals = gt_labels[b, gt_idx, 0].long().to(device)
            gt_lbl_vals = gt_lbl_vals.clamp(0, nc - 1)  # 安全 clamp

            target_labels[b, fg] = gt_lbl_vals
            target_bboxes[b, fg] = gt_bboxes[b, gt_idx]
            oh = torch.eye(nc, device=device)[gt_lbl_vals]
            target_scores[b, fg] = oh * norm_align_metric[b, fg]

        return target_labels, target_bboxes, target_scores, fg_mask
