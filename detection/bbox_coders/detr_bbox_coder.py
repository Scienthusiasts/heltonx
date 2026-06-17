import torch
import torch.nn as nn
from heltonx.utils.register import MODELS
from detection.utils.detr_utils import box_cxcywh_to_xyxy


@MODELS.register
class DETRBBoxCoder(nn.Module):
    """DETR bbox 解码器

    将网络输出的归一化 cxcywh 解码为原图尺度的 xyxy，
    并根据置信度过滤背景类。

    Args:
        img_size (list): 输入图像尺寸 [H, W]
    """

    def __init__(self, img_size):
        super(DETRBBoxCoder, self).__init__()
        self.img_h = img_size[0]
        self.img_w = img_size[1]

    def decode(self, cls_preds, box_preds, img_h=None, img_w=None):
        """将预测结果解码为最终检测结果

        Args:
            cls_preds: [bs, num_queries, nc] 分类 logits
            box_preds: [bs, num_queries, 4]  归一化 cxcywh
            img_h: 可选，模型实际输入高度。不传则用 config 的固定 img_size
            img_w: 可选，模型实际输入宽度。不传则用 config 的固定 img_size

        Returns:
            list[Tensor]: 每个 batch 的检测结果 [num_det, 6=(x1, y1, x2, y2, score, class)]
        """
        bs = cls_preds.shape[0]
        # 分类概率
        cls_probs = cls_preds.softmax(-1)  # [bs, num_queries, nc]
        # 归一化 cxcywh -> xyxy
        box_xyxy = box_cxcywh_to_xyxy(box_preds)  # [bs, num_queries, 4]
        # 映射到实际输入尺寸（推理时必须与模型实际输入尺寸一致，而非config固定值）
        if img_h is None:
            img_h = self.img_h
        if img_w is None:
            img_w = self.img_w
        scale_f = torch.tensor([img_w, img_h, img_w, img_h],
                               device=box_xyxy.device, dtype=box_xyxy.dtype)
        box_xyxy = box_xyxy * scale_f

        results = []
        for b in range(bs):
            # 排除背景类 (最后一类)，取前景类别最大概率
            fg_probs = cls_probs[b, :, :-1]  # [num_queries, nc-1]
            scores, labels = fg_probs.max(dim=-1)  # [num_queries]
            boxes = box_xyxy[b]  # [num_queries, 4]
            results.append(torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).float()], dim=-1))

        return results
