import torch
import torch.nn as nn
from heltonx.utils.register import MODELS


@MODELS.register
class YOLO26BBoxCoder(nn.Module):
    """YOLO26 anchor-free bbox 编解码 (4+nc 通道, 无 objectness)

    与官方 Ultralytics YOLOv8 一致: 直接输出 cls 作为分数, 不乘 objectness。

    Args:
        nc (int): 类别数
    """

    def __init__(self, nc):
        super().__init__()
        self.nc = nc
        self.strides = [8, 16, 32]

    def decode_single(self, feat, lvl):
        """训练用: 解码原始预测为特征图尺度 cxcywh

        Args:
            feat (Tensor): [bs, 4+nc, h, w] Conv2d 输出
            lvl (int): 层索引

        Returns:
            Tensor: [bs, 1, h, w, 4] 特征图尺度 cxcywh
        """
        bs, _, h, w = feat.shape
        device = feat.device

        dx = torch.sigmoid(feat[:, 0:1, :, :])
        dy = torch.sigmoid(feat[:, 1:2, :, :])
        dw = torch.sigmoid(feat[:, 2:3, :, :])
        dh = torch.sigmoid(feat[:, 3:4, :, :])

        grid_x = torch.linspace(0, w - 1, w, device=device).repeat(h, 1).repeat(bs, 1, 1, 1)
        grid_y = torch.linspace(0, h - 1, h, device=device).repeat(w, 1).t().repeat(bs, 1, 1, 1)

        cx = grid_x + dx * 2.0 - 0.5
        cy = grid_y + dy * 2.0 - 0.5
        cw = (dw * 4.0) ** 2
        ch = (dh * 4.0) ** 2

        return torch.stack([cx, cy, cw, ch], dim=-1).unsqueeze(1)

    def decode_nms_free(self, preds, strides=None):
        """NMS-free 推理: 直接输出 xyxy 像素坐标 + scores

        score = cls (无 objectness gate, 与 YOLOv8 一致)

        Args:
            preds (List[Tensor]): [bs, 4+nc, h_i, w_i] o2o 分支预测

        Returns:
            boxes (Tensor): [bs, N, 4] xyxy 像素
            scores (Tensor): [bs, N, nc] 分类分数 (sigmoid(cls))
        """
        if strides is None:
            strides = self.strides
        all_boxes, all_scores = [], []
        for i, pred in enumerate(preds):
            bs, _, h, w = pred.shape
            stride = strides[i]
            device = pred.device

            dx = torch.sigmoid(pred[:, 0:1, :, :])
            dy = torch.sigmoid(pred[:, 1:2, :, :])
            dw = torch.sigmoid(pred[:, 2:3, :, :])
            dh = torch.sigmoid(pred[:, 3:4, :, :])
            cls = torch.sigmoid(pred[:, 4:, :, :])  # [bs, nc, h, w]

            gy, gx = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
            gx, gy = gx.view(1, 1, h, w), gy.view(1, 1, h, w)

            cx_px = (gx + (dx * 2.0 - 0.5)) * stride
            cy_px = (gy + (dy * 2.0 - 0.5)) * stride
            cw_px = (dw * 4.0) ** 2 * stride
            ch_px = (dh * 4.0) ** 2 * stride

            x1, y1 = cx_px - cw_px / 2, cy_px - ch_px / 2
            x2, y2 = cx_px + cw_px / 2, cy_px + ch_px / 2

            boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(bs, -1, 4)
            scores = cls.reshape(bs, self.nc, -1).permute(0, 2, 1)
            all_boxes.append(boxes)
            all_scores.append(scores)
        return torch.cat(all_boxes, dim=1), torch.cat(all_scores, dim=1)

    def decode(self, inputs):
        """推理用: 解码多尺度预测为归一化 cxcywh (旧接口兼容)"""
        outputs = []
        for i, inp in enumerate(inputs):
            bs, _, h, w = inp.shape
            stride = self.strides[i]
            dx = torch.sigmoid(inp[:, 0:1, :, :])
            dy = torch.sigmoid(inp[:, 1:2, :, :])
            dw = torch.sigmoid(inp[:, 2:3, :, :])
            dh = torch.sigmoid(inp[:, 3:4, :, :])
            cls_sig = torch.sigmoid(inp[:, 4:, :, :])
            gy, gx = torch.meshgrid(
                torch.arange(h, device=inp.device, dtype=torch.float32),
                torch.arange(w, device=inp.device, dtype=torch.float32), indexing='ij')
            gx = gx.view(1, 1, h, w).expand(bs, 1, h, w)
            gy = gy.view(1, 1, h, w).expand(bs, 1, h, w)
            cx_px = (gx + (dx * 2.0 - 0.5)) * stride
            cy_px = (gy + (dy * 2.0 - 0.5)) * stride
            cw_px = (dw * 4.0) ** 2 * stride
            ch_px = (dh * 4.0) ** 2 * stride
            img_w = stride * w
            img_h = stride * h
            merged = torch.cat([
                cx_px / img_w, cy_px / img_h, cw_px / img_w, ch_px / img_h, cls_sig
            ], dim=1)
            merged = merged.reshape(bs, 4 + self.nc, -1).permute(0, 2, 1)
            outputs.append(merged)
        return outputs
