import torch
import torch.nn as nn
import numpy as np
from heltonx.utils.register import MODELS
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix


@MODELS.register
class FasterRCNN(nn.Module):
    """Faster R-CNN with FPN 检测器

    组装: Backbone + FPN + RPNHead + StandardRoIHead

    Args:
        backbone (nn.Module): 骨干网络
        fpn (nn.Module): FPN 特征金字塔
        rpn_head (nn.Module): RPN 检测头
        roi_head (nn.Module): RoI Head (第二阶段)
        img_size (list): 输入图像尺寸 [H, W]
        nc (int): 前景类别数
        load_ckpt (str): 预训练权重路径
    """

    def __init__(self, backbone, fpn, rpn_head, roi_head, img_size, nc, load_ckpt=None):
        super().__init__()
        self.nc = nc
        self.img_size = img_size

        self.backbone = backbone
        self.fpn = fpn
        self.rpn_head = rpn_head
        self.roi_head = roi_head

        if load_ckpt:
            self = load_state_dict_with_prefix(self, load_ckpt)

    def forward(self, datas, return_loss=True):
        """一个 batch 的前向流程

        Args:
            datas: 训练时为 [batch_imgs, batch_bboxes, batch_labels]
                   推理时为 batch_imgs
            return_loss: 是否计算损失

        Returns:
            训练时: loss dict
            推理时: List[Tensor] 每张图检测结果 [num_det, 6=(x1,y1,x2,y2,score,class)]
        """
        if return_loss:
            batch_imgs, batch_bboxes, batch_labels = datas[0], datas[1], datas[2]
            img_shapes = [batch_imgs.shape[2:]] * batch_imgs.size(0)

            # Backbone + FPN
            backbone_feat = self.backbone(batch_imgs)
            fpn_feat = self.fpn(backbone_feat)

            # RPN 损失
            rpn_loss = self.rpn_head.loss(fpn_feat, batch_bboxes, batch_labels)

            # RPN proposals (用于 RoI Head 训练)
            with torch.no_grad():
                proposals = self.rpn_head.get_proposals(fpn_feat, img_shapes)

            # RoI Head 损失
            roi_loss = self.roi_head.loss(fpn_feat, proposals, batch_bboxes, batch_labels)

            losses = {**rpn_loss, **roi_loss}
            return losses
        else:
            batch_imgs = datas
            img_shapes = [batch_imgs.shape[2:]] * batch_imgs.size(0)

            # Backbone + FPN
            backbone_feat = self.backbone(batch_imgs)
            fpn_feat = self.fpn(backbone_feat)

            # RPN proposals
            proposals = self.rpn_head.get_proposals(fpn_feat, img_shapes)

            # RoI Head 检测
            det_results = self.roi_head.get_bboxes(fpn_feat, proposals, img_shapes)

            return det_results

    def infer(self, image, vis_heatmap=False, save_vis_path=None):
        """推理一张图/一帧

        Args:
            image (Tensor): [1, 3, H, W]

        Returns:
            boxes (np.ndarray): [obj_nums, 4=(x0, y0, x1, y1)]
            box_scores (np.ndarray): [obj_nums]
            box_classes (np.ndarray): [obj_nums]
        """
        with torch.no_grad():
            det_results = self.forward(image, return_loss=False)
            results = det_results[0]

            if len(results) == 0:
                return [], [], []

            box_classes = results[:, 5].cpu().numpy().astype('int32')
            box_scores = results[:, 4].cpu().numpy()
            boxes = results[:, :4].cpu().numpy()

            return boxes, box_scores, box_classes


if __name__ == '__main__':
    from detection.models.backbones.timm_resnet import TIMMResNet
    from detection.models.necks.fpn import FPN
    from detection.models.dense_heads.rpn_head import RPNHead
    from detection.models.roi_heads.standard_roi_head import StandardRoIHead
    from detection.models.roi_heads.bbox_head import BBoxHead
    from detection.utils.anchor_generator import AnchorGenerator
    from detection.assigners.max_iou_assigner import MaxIoUAssigner
    from detection.bbox_coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
    from detection.losses.loss import BCELoss, CrossEntropyLoss, SmoothL1Loss
    from detection.utils.roi_utils import RoIAlign

    img_size = [640, 640]
    num_classes = 80

    backbone = TIMMResNet(modelType='resnet50.a1_in1k', out_layers=[1, 2, 3, 4])
    fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channel=256, num_extra_levels=1)
    anchor_generator = AnchorGenerator(
        strides=[4, 8, 16, 32, 64],
        ratios=[0.5, 1.0, 2.0],
        scales=[8]
    )
    rpn_head = RPNHead(
        in_channels=256,
        featmap_strides=[4, 8, 16, 32, 64],
        anchor_generator=anchor_generator,
        assigner=MaxIoUAssigner(pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3),
        bbox_coder=DeltaXYWHBBoxCoder(),
        cls_loss=BCELoss(reduction='mean'),
        reg_loss=SmoothL1Loss(beta=1.0/9.0, reduction='mean'),
        num_samples=256,
        pos_fraction=0.5,
        nms_pre=2000,
        max_per_img=1000,
        nms_thr=0.7
    )
    bbox_head = BBoxHead(in_channels=256, num_classes=num_classes, reg_class_agnostic=False)
    roi_head = StandardRoIHead(
        bbox_roi_extractor=RoIAlign(
            output_size=(7, 7),
            use_multilevel=True,
            featmap_strides=[4, 8, 16, 32, 64]
        ),
        bbox_head=bbox_head,
        assigner=MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5),
        bbox_coder=DeltaXYWHBBoxCoder(target_means=(0., 0., 0., 0.), target_stds=(0.1, 0.1, 0.2, 0.2)),
        cls_loss=CrossEntropyLoss(reduction='mean'),
        reg_loss=SmoothL1Loss(beta=1.0/9.0, reduction='mean'),
        num_samples=512,
        pos_fraction=0.25
    )

    model = FasterRCNN(
        backbone=backbone,
        fpn=fpn,
        rpn_head=rpn_head,
        roi_head=roi_head,
        img_size=img_size,
        nc=num_classes
    )

    # 验证训练前向
    batch_imgs = torch.randn(2, 3, 640, 640)
    batch_bboxes = [
        torch.tensor([[50, 50, 100, 100], [200, 200, 150, 150]], dtype=torch.float32),
        torch.tensor([[100, 100, 120, 120]], dtype=torch.float32),
    ]
    batch_labels = [
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    ]
    losses = model([batch_imgs, batch_bboxes, batch_labels], return_loss=True)
    print("Training losses:", losses)

    # 验证推理
    boxes, scores, classes = model.infer(batch_imgs[0:1])
    print("Inference boxes:", boxes.shape if len(boxes) > 0 else 0)
