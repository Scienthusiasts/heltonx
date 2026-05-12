import torch
import torch.nn as nn
import numpy as np

from heltonx.utils.register import MODELS
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix


@MODELS.register
class DETR(nn.Module):
    """完整 DETR 检测网络

    组装: backbone + DETRTransformer(fpn) + DETRHead + DETRBBoxCoder

    Args:
        backbone (nn.Module):     骨干网络 (TIMMBackbone 等)
        fpn (nn.Module):          DETRTransformer (占 fpn 槽位)
        head (nn.Module):         DETRHead 检测头 (内含 loss 和 assigner)
        bbox_coder (nn.Module):   DETRBBoxCoder 解码器
        img_size (list):          输入图像尺寸 [H, W]
        nc (int):                 前景类别数
        load_ckpt (str):          预训练权重路径
        nms_score_thr (float):    置信度过滤阈值 (DETR 不做 NMS, 仅过滤低置信度)
    """

    def __init__(self, backbone, fpn, head, bbox_coder, img_size, nc, load_ckpt, score_thr=0.05):
        super().__init__()
        self.nc = nc
        self.img_size = img_size
        self.score_thr = score_thr
        self.bbox_coder = bbox_coder

        # 网络组件
        self.backbone = backbone
        self.fpn = fpn
        self.head = head

        # 导入预训练权重
        if load_ckpt:
            self = load_state_dict_with_prefix(self, load_ckpt)

    def _xywh_to_norm_cxcywh(self, batch_bboxes, img_h, img_w):
        """将数据集输出的未归一化 xywh 转为 DETR 需要的归一化 cxcywh

        Args:
            batch_bboxes: list[Tensor], 每个 [num_gt, 4] 未归一化 xywh
            img_h: 图像高度
            img_w: 图像宽度

        Returns:
            list[Tensor], 每个 [num_gt, 4] 归一化 cxcywh
        """
        converted = []
        for bboxes in batch_bboxes:
            if len(bboxes) == 0:
                converted.append(bboxes)
                continue
            # xywh -> cxcywh
            cxcywh = bboxes.clone()
            cxcywh[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2  # cx = x + w/2
            cxcywh[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2  # cy = y + h/2
            # w, h 保持不变
            # 归一化
            cxcywh[:, 0] /= img_w
            cxcywh[:, 1] /= img_h
            cxcywh[:, 2] /= img_w
            cxcywh[:, 3] /= img_h
            converted.append(cxcywh)
        return converted

    def forward(self, datas, return_loss=True):
        """一个 batch 的前向流程

        Args:
            datas: 训练时为 [batch_imgs, batch_bboxes, batch_labels]
                   推理时为 batch_imgs
            return_loss: 是否计算损失

        Returns:
            训练时: loss dict
            推理时: (cls_preds, box_preds)
        """
        if return_loss:
            batch_imgs, batch_bboxes, batch_labels = datas[0], datas[1], datas[2]
            # 将 xywh -> 归一化 cxcywh (DETR 需要的格式)
            img_h, img_w = batch_imgs.shape[2], batch_imgs.shape[3]
            batch_bboxes = self._xywh_to_norm_cxcywh(batch_bboxes, img_h, img_w)
            # 前向
            backbone_feat = self.backbone(batch_imgs)
            transformer_output = self.fpn(backbone_feat)
            hs_all, _ = transformer_output
            cls_preds, box_preds = self.head(transformer_output)
            # 计算损失 (匈牙利匹配 + focal/l1/giou 三项 + Auxiliary Loss)
            loss = self.head.loss(cls_preds, box_preds, hs_all, batch_labels, batch_bboxes)
            return loss
        else:
            batch_imgs = datas
            backbone_feat = self.backbone(batch_imgs)
            transformer_output = self.fpn(backbone_feat)
            cls_preds, box_preds = self.head(transformer_output)
            return cls_preds, box_preds

    def infer(self, image, vis_heatmap=None, save_vis_path=None):
        """推理一张图/一帧

        Args:
            image: 输入图像 [1, 3, H, W]

        Returns:
            boxes:       [obj_nums, 4=(x0, y0, x1, y1)]
            box_scores:  [obj_nums]
            box_classes: [obj_nums]
        """
        H, W = image.shape[2:]
        with torch.no_grad():
            cls_preds, box_preds = self.forward(image, return_loss=False)
            # 解码: [bs, num_det, 6=(x1, y1, x2, y2, score, class)]
            predictions = self.bbox_coder.decode(cls_preds, box_preds)
            # 取 batch[0]
            results = predictions[0]

            # 过滤低置信度
            mask = results[:, 4] >= self.score_thr
            results = results[mask]

            if len(results) == 0:
                return [], [], []

            box_classes = results[:, 5].cpu().numpy().astype('int32')
            box_scores = results[:, 4].cpu().numpy()
            boxes = results[:, :4].cpu().numpy()

            return boxes, box_scores, box_classes
