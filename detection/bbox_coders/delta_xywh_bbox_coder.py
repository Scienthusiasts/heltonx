import torch
import torch.nn as nn
from heltonx.utils.register import MODELS


@MODELS.register
class DeltaXYWHBBoxCoder(nn.Module):
    """Delta XYWH BBox 编解码器 (Faster R-CNN 标准编码方式)

    编码: 将 gt_bboxes 相对于 anchors/proposals 编码为 delta (dx, dy, dw, dh)
    解码: 将 delta 还原为绝对坐标 (xyxy)

    Args:
        target_means (Tuple[float]): delta 的均值偏移，默认 (0., 0., 0., 0.)
        target_stds (Tuple[float]): delta 的标准差缩放，默认 (1., 1., 1., 1.)
        clip_border (bool): 解码后是否裁剪到图像边界
    """

    def __init__(self, target_means=(0., 0., 0., 0.), target_stds=(1., 1., 1., 1.), clip_border=True):
        super().__init__()
        self.means = torch.tensor(target_means, dtype=torch.float32)
        self.stds = torch.tensor(target_stds, dtype=torch.float32)
        self.clip_border = clip_border

    def encode(self, bboxes, reference_boxes):
        """编码: bboxes 相对于 reference_boxes 的 delta

        Args:
            bboxes (Tensor): [N, 4] xyxy，待编码的框 (GT)
            reference_boxes (Tensor): [N, 4] xyxy，参考框 (anchors 或 proposals)

        Returns:
            Tensor: [N, 4] delta (dx, dy, dw, dh)
        """
        assert bboxes.shape[0] == reference_boxes.shape[0]

        # 转换为 cxcywh
        bboxes = self.xyxy2cxcywh(bboxes)
        reference_boxes = self.xyxy2cxcywh(reference_boxes)

        # 计算 delta
        # dx = (bbox_cx - ref_cx) / ref_w
        # dy = (bbox_cy - ref_cy) / ref_h
        # dw = log(bbox_w / ref_w)
        # dh = log(bbox_h / ref_h)
        dx = (bboxes[:, 0] - reference_boxes[:, 0]) / reference_boxes[:, 2]
        dy = (bboxes[:, 1] - reference_boxes[:, 1]) / reference_boxes[:, 3]
        dw = torch.log(bboxes[:, 2] / (reference_boxes[:, 2] + 1e-7))
        dh = torch.log(bboxes[:, 3] / (reference_boxes[:, 3] + 1e-7))

        deltas = torch.stack([dx, dy, dw, dh], dim=-1)

        # 标准化
        means = self.means.to(deltas.device)
        stds = self.stds.to(deltas.device)
        deltas = (deltas - means) / stds

        return deltas

    def decode(self, deltas, reference_boxes, max_shape=None):
        """解码: delta + reference_boxes → 绝对坐标 xyxy

        Args:
            deltas (Tensor): [N, 4] (dx, dy, dw, dh)
            reference_boxes (Tensor): [N, 4] xyxy，参考框
            max_shape (Tuple[int, int], optional): (H, W) 图像尺寸，用于裁剪

        Returns:
            Tensor: [N, 4] xyxy 解码后的框
        """
        assert deltas.shape[0] == reference_boxes.shape[0]

        # 反标准化
        means = self.means.to(deltas.device)
        stds = self.stds.to(deltas.device)
        deltas = deltas * stds + means

        # 参考框转 cxcywh
        ref = self.xyxy2cxcywh(reference_boxes)

        # 解码
        cx = deltas[:, 0] * ref[:, 2] + ref[:, 0]
        cy = deltas[:, 1] * ref[:, 3] + ref[:, 1]
        w = torch.exp(deltas[:, 2]) * ref[:, 2]
        h = torch.exp(deltas[:, 3]) * ref[:, 3]

        # 转回 xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        decoded = torch.stack([x1, y1, x2, y2], dim=-1)

        if self.clip_border and max_shape is not None:
            h_img, w_img = max_shape
            decoded[:, 0::2].clamp_(min=0, max=w_img)
            decoded[:, 1::2].clamp_(min=0, max=h_img)

        return decoded

    @staticmethod
    def xyxy2cxcywh(boxes):
        """xyxy → cxcywh
        Args:
            boxes (Tensor): [N, 4] xyxy
        Returns:
            Tensor: [N, 4] cxcywh
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)

    @staticmethod
    def cxcywh2xyxy(boxes):
        """cxcywh → xyxy
        Args:
            boxes (Tensor): [N, 4] cxcywh
        Returns:
            Tensor: [N, 4] xyxy
        """
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)


if __name__ == '__main__':
    # 验证编解码一致性
    coder = DeltaXYWHBBoxCoder()
    anchors = torch.tensor([
        [0, 0, 100, 100],
        [50, 50, 150, 150],
    ], dtype=torch.float32)
    gt = torch.tensor([
        [10, 10, 110, 110],
        [60, 60, 160, 170],
    ], dtype=torch.float32)

    deltas = coder.encode(gt, anchors)
    decoded = coder.decode(deltas, anchors, max_shape=(640, 640))
    print("Original GT:", gt)
    print("Decoded:", decoded)
    print("Diff:", (gt - decoded).abs().max())
