import torch
import torch.nn as nn
import numpy as np
from heltonx.utils.register import MODELS
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix


@MODELS.register
class YOLO26OBB(nn.Module):
    """YOLO26-OBB 有向目标检测器 (与官方 OBB26 一致)

    ★★★ 与官方 YOLO26 OBB26 完全一致的回归方式:
    - BBox: ltrb 直接回归 + dist2rbox 解码 (无 sigmoid, 无 DFL)
    - Angle: 原始 logits (弧度, 无 sigmoid 变换)
    - 推理: o2o 头 → decode_nms_free (dist2rbox + stride + regularize)
    - 训练: o2m + o2o 双头 + progressive loss
    """

    def __init__(self, img_size, backbone, fpn, head, nc,
                 score_thr=0.001, max_det=300, load_ckpt=None):
        super().__init__()
        self.img_size = img_size
        self.nc = nc
        self.score_thr = score_thr
        self.max_det = max_det
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.bbox_coder = self.head.loss_fn.bbox_coder

        if load_ckpt:
            self = load_state_dict_with_prefix(self, load_ckpt)

    def _ultralytics_key_map(self):
        """返回 ultralytics 数字索引键到本模型命名键的映射

        backbone + FPN 映射与 YOLO26 完全一致 (可复用)
        OBB26 head 结构不同, 无法直接映射
        """
        return {
            '0': 'backbone.stem.0',
            '1': 'backbone.stem.1',
            '2': 'backbone.stem.2',
            '3': 'backbone.dark3.0',
            '4': 'backbone.dark3.1',
            '5': 'backbone.dark4.0',
            '6': 'backbone.dark4.1',
            '7': 'backbone.dark5.0',
            '8': 'backbone.dark5.1',
            '9': 'backbone.dark5.2',
            '10': 'backbone.dark5.3',
            '13': 'fpn.td_p4_C3k2',
            '16': 'fpn.td_p3_C3k2',
            '17': 'fpn.p3_downsample',
            '19': 'fpn.bu_p4_C3k2',
            '20': 'fpn.p4_downsample',
            '22': 'fpn.bu_p5_C3k2',
        }

    def update_progressive(self, cur_epoch):
        if hasattr(self.head, 'update_progressive'):
            self.head.update_progressive(cur_epoch)

    def forward(self, datas, return_loss=True):
        if return_loss:
            batch_imgs, batch_bboxes, batch_labels = datas[0], datas[1], datas[2]
            backbone_feat = self.backbone(batch_imgs)
            p = self.fpn(backbone_feat)
            return self.head.loss(p, batch_bboxes, batch_labels)
        else:
            backbone_feat = self.backbone(datas)
            p = self.fpn(backbone_feat)
            return self.head(p)

    def infer(self, image, **kwargs):
        """NMS-free 推理: 先 backbone+FPN, 再 o2o 头 + 解码 + top-k

        ★ 修复: 正确处理 batch 维度, 逐图返回预测

        Args:
            image: [bs, 3, H, W] 或 [3, H, W] 单图张量

        Returns:
            当 bs=1 时 (评估场景):
                boxes: [N, 5] xywhr 像素坐标
                scores: [N] 最高分类分数
                cls: [N] 类别索引
            当 bs>1 时:
                list of (boxes, scores, cls), 每图一组
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        bs = image.shape[0]
        H, W = image.shape[2:]

        with torch.no_grad():
            backbone_feat = self.backbone(image)
            fpn_feat = self.fpn(backbone_feat)
            o2o_preds = self.head.forward_o2o(fpn_feat)
            # BBox coder 解码到 xywhr 像素坐标 + scores
            boxes_cat, scores_cat = self.bbox_coder.decode_nms_free(o2o_preds)
            # boxes_cat: [bs, N, 5], scores_cat: [bs, N, nc]

        results = []
        for b in range(bs):
            max_scores, max_cls = scores_cat[b].max(dim=-1)
            keep = max_scores > self.score_thr
            if keep.sum() == 0:
                results.append((np.zeros((0, 5), dtype=np.float32),
                                np.zeros(0, dtype=np.float32),
                                np.zeros(0, dtype=np.int32)))
                continue

            boxes_f = boxes_cat[b][keep]
            scores_f = max_scores[keep]
            cls_f = max_cls[keep]

            if len(scores_f) > self.max_det:
                _, top_idx = torch.topk(scores_f, self.max_det)
                boxes_f, scores_f, cls_f = boxes_f[top_idx], scores_f[top_idx], cls_f[top_idx]

            results.append((boxes_f.cpu().numpy(), scores_f.cpu().numpy(),
                            cls_f.cpu().numpy().astype('int32')))

        if bs == 1:
            return results[0]
        return results
