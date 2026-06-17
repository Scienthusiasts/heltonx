import torch
import torch.nn as nn
import numpy as np
from heltonx.utils.register import MODELS
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix


@MODELS.register
class YOLO26(nn.Module):
    """YOLO26 NMS-free 检测器

    推理: o2o 头 → top-k 排序 (无需 NMS)
    训练: o2m + o2o 双头 + progressive loss (由 loss_fn 处理)
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
        # 从 head.loss_fn 获取 bbox_coder, 避免重复构建
        self.bbox_coder = self.head.loss_fn.bbox_coder

        if load_ckpt:
            self = load_state_dict_with_prefix(self, load_ckpt)

    def _ultralytics_key_map(self):
        """返回 ultralytics 数字索引键到本模型命名键的映射

        ultralytics DetectionModel 使用扁平 nn.Sequential，键名为 model.0, model.1, ...
        本模型使用命名子模块: backbone.stem/dark3/dark4/dark5, fpn.td_p4_C3k2/..., head.p_heads/...

        映射关系 (基于 ultralytics yolo26.yaml):
            Backbone (0-10):
                0  -> backbone.stem.0   (Conv)
                1  -> backbone.stem.1   (Conv)
                2  -> backbone.stem.2   (C3k2)
                3  -> backbone.dark3.0  (Conv)
                4  -> backbone.dark3.1  (C3k2)
                5  -> backbone.dark4.0  (Conv)
                6  -> backbone.dark4.1  (C3k2)
                7  -> backbone.dark5.0  (Conv)
                8  -> backbone.dark5.1  (C3k2)
                9  -> backbone.dark5.2  (SPPF)
                10 -> backbone.dark5.3  (C2PSA)
            FPN (11=Upsample, 12/15/18/21=Concat 无参数):
                13 -> fpn.td_p4_C3k2    (C3k2)
                16 -> fpn.td_p3_C3k2    (C3k2)
                17 -> fpn.p3_downsample  (Conv)
                19 -> fpn.bu_p4_C3k2    (C3k2)
                20 -> fpn.p4_downsample  (Conv)
                22 -> fpn.bu_p5_C3k2    (C3k2)
            Head (23=Detect):
                结构不同 (ultralytics: cv2/cv3 分离 bbox/cls, 本项目: o2m/o2o 合并输出)
                无法直接映射，head 权重将使用随机初始化
        """
        return {
            # Backbone
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
            # FPN
            '13': 'fpn.td_p4_C3k2',
            '16': 'fpn.td_p3_C3k2',
            '17': 'fpn.p3_downsample',
            '19': 'fpn.bu_p4_C3k2',
            '20': 'fpn.p4_downsample',
            '22': 'fpn.bu_p5_C3k2',
        }

    def update_progressive(self, cur_epoch):
        """epoch 级别更新 progressive loss 权重"""
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
        """NMS-free 推理: 先 backbone+FPN, 再 o2o 头 + 解码 + top-k"""
        H, W = image.shape[2:]

        with torch.no_grad():
            backbone_feat = self.backbone(image)
            fpn_feat = self.fpn(backbone_feat)
            o2o_preds = self.head.forward_o2o(fpn_feat)
            # BBox coder 解码到 xyxy 像素坐标 + scores
            boxes_cat, scores_cat = self.bbox_coder.decode_nms_free(o2o_preds)

            # 取每类的最高分
            max_scores, max_cls = scores_cat.max(dim=-1)
            keep = max_scores > self.score_thr
            if keep.sum() == 0:
                return [], [], []

            boxes_f, scores_f, cls_f = boxes_cat[keep], max_scores[keep], max_cls[keep]

            if len(scores_f) > self.max_det:
                _, top_idx = torch.topk(scores_f, self.max_det)
                boxes_f, scores_f, cls_f = boxes_f[top_idx], scores_f[top_idx], cls_f[top_idx]

            boxes_f[:, 0].clamp_(0, W)
            boxes_f[:, 1].clamp_(0, H)
            boxes_f[:, 2].clamp_(0, W)
            boxes_f[:, 3].clamp_(0, H)

            return (boxes_f.cpu().numpy(), scores_f.cpu().numpy(),
                    cls_f.cpu().numpy().astype('int32'))
