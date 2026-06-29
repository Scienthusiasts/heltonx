import torch
import torch.nn as nn
import math
from heltonx.utils.register import MODELS
from detectionobb.utils.obb_ops import dist2rbox, rbox2dist, regularize_rboxes


@MODELS.register
class YOLO26OBBBBoxCoder(nn.Module):
    """YOLO26-OBB anchor-free 旋转框编解码 (与官方 OBB26 一致)

    ★★★ 与官方 YOLO26 OBB26 完全一致的回归方式:
    - BBox: 直接回归 ltrb 距离 (无 sigmoid, 无 DFL), 使用 dist2rbox 解码
    - Angle: 直接输出原始 logits (弧度, 无 sigmoid 变换)
    - reg_max=1 → 不使用 DFL, 回归损失为 L1

    与旧版 sigmoid 解码的区别:
    - 旧版: dx,dy,dw,dh 全用 sigmoid → 有界回归范围
    - 新版: ltrb 直接回归 → 无界回归范围 (unconstrained range)
    - 旧版: angle = (sigmoid(logits) - 0.25) * π → [-π/4, 3π/4]
    - 新版: angle = logits → 无界, 由 sin²(2Δθ) loss 处理周期性

    Args:
        nc (int): 类别数
        reg_max (int): DFL 的 reg_max, 官方 YOLO26 设为 1 (无 DFL)
    """

    def __init__(self, nc, reg_max=1):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.strides = [8, 16, 32]
        # DFL 投影向量 (reg_max=1 时退化为标量 0, 但不使用)
        self.proj = nn.Parameter(torch.arange(reg_max, dtype=torch.float32), requires_grad=False)

    def decode_single(self, feat, lvl):
        """训练用: 提取原始 ltrb 距离和角度 logits

        ★ 与官方 v8OBBLoss.bbox_decode 一致:
        - 当 use_dfl=False (reg_max=1): pred_dist 直接使用原始 ltrb logits
        - 当 use_dfl=True  (reg_max>1): softmax(4×reg_max).matmul(proj) → ltrb 连续值

        Args:
            feat (Tensor): [bs, 5+nc, h, w] Conv2d 输出
            lvl (int): 层索引

        Returns:
            pred_dist (Tensor): [bs, N, 4] ltrb 距离 (anchor 空间)
            pred_angle (Tensor): [bs, N, 1] 角度 logits (弧度, 无 sigmoid)
        """
        bs, _, h, w = feat.shape
        N = h * w

        # 提取 ltrb (前 4 通道, 原始 logits)
        pred_dist = feat[:, :4, :, :].reshape(bs, 4, N).permute(0, 2, 1)  # [bs, N, 4]

        # ★ DFL 解码 (reg_max > 1 时启用, YOLO26 设 reg_max=1 不启用)
        if self.reg_max > 1:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )

        # 提取角度 (第 5 通道, 原始 logits → 弧度)
        pred_angle = feat[:, 4:5, :, :].reshape(bs, 1, N).permute(0, 2, 1)  # [bs, N, 1]

        return pred_dist, pred_angle

    def decode_angle(self, feat):
        """提取角度 logits (与官方 OBB26 一致: 无 sigmoid 变换)

        Args:
            feat (Tensor): [bs, 5+nc, h, w]

        Returns:
            Tensor: [bs, N, 1] 角度 logits (弧度, 无 sigmoid)
        """
        bs, _, h, w = feat.shape
        return feat[:, 4:5, :, :].reshape(bs, h * w, 1)

    def decode_nms_free(self, preds, strides=None):
        """NMS-free 推理: 输出 xywhr 像素坐标 + scores

        ★ 与官方 Detect._get_decode_boxes + OBB.decode_bboxes 一致:
        1. DFL 解码 (reg_max=1 时跳过)
        2. dist2rbox: ltrb + angle + anchors → xywh (anchor 空间)
        3. * strides: 转为像素空间

        Args:
            preds: dict {'box': [[bs,4,h,w],...], 'cls': [[bs,nc,h,w],...], 'angle': [[bs,1,h,w],...]}
            strides (list): 各层 stride

        Returns:
            boxes (Tensor): [bs, N, 5] xywhr 像素坐标 (cx, cy, w, h, angle)
            scores (Tensor): [bs, N, nc] 分类分数 (sigmoid(cls))
        """
        if strides is None:
            strides = self.strides
        all_boxes, all_scores, all_anchors, all_strides = [], [], [], []

        for i in range(len(preds['box'])):
            pred_dist_raw = preds['box'][i]     # [bs, 4, h, w]
            cls_raw = preds['cls'][i]            # [bs, nc, h, w]
            angle_raw = preds['angle'][i]        # [bs, 1, h, w]
            stride = strides[i]
            device = pred_dist_raw.device
            bs = pred_dist_raw.shape[0]
            h, w = pred_dist_raw.shape[2], pred_dist_raw.shape[3]
            N = h * w

            # ltrb 距离: [bs, 4, h, w] → [bs, N, 4]
            pred_dist = pred_dist_raw.reshape(bs, 4, N).permute(0, 2, 1)

            # DFL 解码 (reg_max=1 时跳过)
            if self.reg_max > 1:
                b, a, c = pred_dist.shape
                pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                    self.proj.type(pred_dist.dtype)
                )

            # 角度: [bs, 1, h, w] → [bs, N, 1]
            pred_angle = angle_raw.reshape(bs, 1, N).permute(0, 2, 1)

            # 分类分数: [bs, nc, h, w] → [bs, N, nc]
            scores = torch.sigmoid(cls_raw).reshape(bs, self.nc, N).permute(0, 2, 1)

            # ★ 构建锚点 (grid 坐标, 与官方 make_anchors 一致)
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
            anchors = torch.stack([gx + 0.5, gy + 0.5], dim=-1).reshape(-1, 2)  # grid 坐标

            # ★ dist2rbox 解码: ltrb + angle + anchors → xywh (anchor 空间)
            xywh = dist2rbox(pred_dist, pred_angle, anchors, dim=-1)  # [bs, N, 4]

            # 转为像素空间
            stride_ts = torch.full((N, 1), stride, device=device, dtype=torch.float32)
            xywh_px = xywh * stride_ts  # [bs, N, 4]

            # 拼接 xywhr: [bs, N, 5]
            boxes = torch.cat([xywh_px, pred_angle], dim=-1)  # [bs, N, 5]

            # ★ regularize: 统一 w≥h 约定 + 角度归一化到 [0, π/2)
            boxes = regularize_rboxes(boxes)

            all_boxes.append(boxes)
            all_scores.append(scores)

        return torch.cat(all_boxes, dim=1), torch.cat(all_scores, dim=1)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """训练用: 解码 ltrb + angle → xywhr (anchor 空间)

        ★ 与官方 v8OBBLoss.bbox_decode 一致:
        - DFL 解码后通过 dist2rbox 转为 xywh
        - 拼接角度 → xywhr (anchor 空间, 不乘 stride)
        - anchor_points 为 grid 坐标 (与官方 make_anchors 一致)

        Args:
            anchor_points (Tensor): [N, 2] 锚点 grid 坐标 (anchor 空间, 如 0.5, 1.5, ...)
            pred_dist (Tensor): [bs, N, 4] ltrb 距离
            pred_angle (Tensor): [bs, N, 1] 角度 logits

        Returns:
            Tensor: [bs, N, 5] xywhr (anchor 空间)
        """
        if self.reg_max > 1:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        xywh = dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1)  # [bs, N, 4]
        return torch.cat([xywh, pred_angle], dim=-1)  # [bs, N, 5] xywhr
