import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from heltonx.utils.register import MODELS
from heltonx.utils.utils import init_weights


@MODELS.register
class RPNHead(nn.Module):
    """RPN 检测头 (Region Proposal Network)

    为 Faster R-CNN 第一阶段提供候选框生成。
    在 FPN 多层特征图上共享同一组卷积权重进行前景/背景分类和 bbox 回归。

    Args:
        in_channels (int): 输入特征图通道数 (FPN 输出通道)
        featmap_strides (List[int]): 每层特征图的下采样率, e.g. [4,8,16,32,64]
        anchor_generator (nn.Module): Anchor 生成器
        assigner (nn.Module): 正负样本分配器 (MaxIoUAssigner)
        bbox_coder (nn.Module): BBox 编解码器 (DeltaXYWHBBoxCoder)
        cls_loss (nn.Module): 分类损失函数 (BCELoss)
        reg_loss (nn.Module): 回归损失函数 (SmoothL1Loss)
        num_samples (int): 每张图采样 anchor 总数, 默认 256
        pos_fraction (float): 正样本比例, 默认 0.5
        nms_pre (int): NMS 前保留的最大候选框数 (每层), 默认 2000
        max_per_img (int): 每张图最终保留的最大候选框数, 默认 1000
        nms_thr (float): NMS IoU 阈值, 默认 0.7
        min_bbox_size (float): 最小 bbox 尺寸, 小于此值的过滤
    """

    def __init__(self, in_channels, featmap_strides, anchor_generator,
                 assigner, bbox_coder, cls_loss, reg_loss,
                 num_samples=256, pos_fraction=0.5,
                 nms_pre=2000, max_per_img=1000, nms_thr=0.7,
                 min_bbox_size=0):
        super().__init__()
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.num_levels = len(featmap_strides)

        self.anchor_generator = anchor_generator
        self.num_anchors = anchor_generator.num_base_anchors_per_level()

        self.assigner = assigner
        self.bbox_coder = bbox_coder
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss

        self.num_samples = num_samples
        self.pos_fraction = pos_fraction
        self.pos_samples = int(num_samples * pos_fraction)
        self.neg_samples = num_samples - self.pos_samples

        self.nms_pre = nms_pre
        self.max_per_img = max_per_img
        self.nms_thr = nms_thr
        self.min_bbox_size = min_bbox_size

        # 共享卷积 + 分类/回归头
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(in_channels, self.num_anchors, 1)
        self.rpn_reg = nn.Conv2d(in_channels, self.num_anchors * 4, 1)

        init_weights(self.rpn_conv, 'normal', 0, 0.01)
        init_weights(self.rpn_cls, 'normal', 0, 0.01)
        init_weights(self.rpn_reg, 'normal', 0, 0.01)

    def forward_single(self, x):
        """单层前向传播"""
        x = F.relu(self.rpn_conv(x))
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def forward(self, feats):
        """多层前向传播

        Args:
            feats (List[Tensor]): FPN 输出特征图

        Returns:
            cls_scores (List[Tensor]): 每层 [B, num_anchors, H, W]
            bbox_preds (List[Tensor]): 每层 [B, num_anchors*4, H, W]
        """
        cls_scores = []
        bbox_preds = []
        for feat in feats:
            cls_score, bbox_pred = self.forward_single(feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds

    def loss(self, feats, batch_bboxes, batch_labels):
        """计算 RPN 损失

        Args:
            feats (List[Tensor]): FPN 特征图
            batch_bboxes (List[Tensor]): 每张图 GT bbox, xywh 格式
            batch_labels (List[Tensor]): 每张图 GT 类别

        Returns:
            dict: {"rpn_cls_loss": ..., "rpn_reg_loss": ...}
        """
        featmap_sizes = [feat.shape[-2:] for feat in feats]
        device = feats[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device)

        cls_scores, bbox_preds = self.forward(feats)
        bs = len(batch_bboxes)

        total_cls_loss = 0.
        total_reg_loss = 0.
        num_total_pos = 0
        num_total_samples = 0

        for img_idx in range(bs):
            gt_bboxes = batch_bboxes[img_idx]
            # 将 xywh -> xyxy
            if gt_bboxes.numel() > 0:
                gt_bboxes_xyxy = self._xywh2xyxy(gt_bboxes)
            else:
                gt_bboxes_xyxy = gt_bboxes.new_zeros((0, 4))

            img_cls_preds = []
            img_reg_preds = []
            img_anchors = []

            for lvl in range(self.num_levels):
                H, W = featmap_sizes[lvl]
                cls_score = cls_scores[lvl][img_idx]   # [num_anchors, H, W]
                bbox_pred = bbox_preds[lvl][img_idx]   # [num_anchors*4, H, W]

                cls_pred = cls_score.permute(1, 2, 0).reshape(-1)           # [H*W*num_anchors]
                reg_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)        # [H*W*num_anchors, 4]
                anchors = mlvl_anchors[lvl]                                  # [H*W*num_anchors, 4]

                img_cls_preds.append(cls_pred)
                img_reg_preds.append(reg_pred)
                img_anchors.append(anchors)

            all_cls_preds = torch.cat(img_cls_preds)      # [total_anchors]
            all_reg_preds = torch.cat(img_reg_preds)      # [total_anchors, 4]
            all_anchors = torch.cat(img_anchors)          # [total_anchors, 4]

            # 正负样本分配
            assigned_gt_inds, _ = self.assigner(all_anchors, gt_bboxes_xyxy)

            pos_inds = (assigned_gt_inds > 0).nonzero(as_tuple=False).squeeze(1)
            neg_inds = (assigned_gt_inds == 0).nonzero(as_tuple=False).squeeze(1)

            # 采样限制
            if pos_inds.numel() > self.pos_samples:
                perm = torch.randperm(pos_inds.numel(), device=device)[:self.pos_samples]
                pos_inds = pos_inds[perm]
            if neg_inds.numel() > self.neg_samples:
                perm = torch.randperm(neg_inds.numel(), device=device)[:self.neg_samples]
                neg_inds = neg_inds[perm]

            # 回归损失 (仅正样本)
            if pos_inds.numel() > 0:
                pos_reg_preds = all_reg_preds[pos_inds]
                pos_anchors = all_anchors[pos_inds]
                pos_gt_inds = assigned_gt_inds[pos_inds] - 1
                pos_gt_bboxes = gt_bboxes_xyxy[pos_gt_inds]

                reg_targets = self.bbox_coder.encode(pos_gt_bboxes, pos_anchors)
                # reg_loss 配置为 reduction='mean'，先还原为 sum，最后全局平均
                total_reg_loss += self.reg_loss(pos_reg_preds, reg_targets) * pos_inds.numel()
                num_total_pos += pos_inds.numel()

            # 分类损失 (采样后的正负样本)
            sampled_inds = torch.cat([pos_inds, neg_inds])
            sampled_cls_preds = all_cls_preds[sampled_inds]
            sampled_cls_targets = torch.cat([
                torch.ones(pos_inds.numel(), device=device),
                torch.zeros(neg_inds.numel(), device=device)
            ])
            # cls_loss 配置为 reduction='mean'，先还原为 sum，最后全局平均
            total_cls_loss += self.cls_loss(sampled_cls_preds, sampled_cls_targets) * sampled_inds.numel()
            num_total_samples += sampled_inds.numel()

        # MMDet 做法: RPN reg loss 也除以总采样数（含正负样本），而非仅正样本数
        losses = dict(
            rpn_cls_loss=total_cls_loss / max(num_total_samples, 1),
            # rpn_reg_loss=total_reg_loss / max(num_total_samples, 1)
            rpn_reg_loss=total_reg_loss /max(num_total_pos, 1)
        )
        return losses

    def get_proposals(self, feats, img_shapes):
        """生成候选框 (推理时使用)

        Args:
            feats (List[Tensor]): FPN 特征图
            img_shapes (List[Tuple[int, int]]): 每张图的实际尺寸 (H, W)

        Returns:
            List[Tensor]: 每张图的 proposals, 每个 [num_proposals, 4] xyxy
        """
        featmap_sizes = [feat.shape[-2:] for feat in feats]
        device = feats[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device)
        cls_scores, bbox_preds = self.forward(feats)
        bs = feats[0].size(0)

        result_list = []
        for img_idx in range(bs):
            mlvl_proposals = []
            mlvl_scores = []
            for lvl in range(self.num_levels):
                H, W = featmap_sizes[lvl]
                anchors = mlvl_anchors[lvl]  # [H*W*num_anchors, 4]

                cls_score = cls_scores[lvl][img_idx]   # [num_anchors, H, W]
                bbox_pred = bbox_preds[lvl][img_idx]   # [num_anchors*4, H, W]

                # reshape
                cls_score = cls_score.permute(1, 2, 0).reshape(-1)          # [H*W*num_anchors]
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)       # [H*W*num_anchors, 4]

                # 按分数取 topk pre_nms (使用 stable sort 保证确定性)
                if cls_score.numel() > self.nms_pre:
                    sorted_scores, sorted_inds = torch.sort(cls_score, descending=True, stable=True)
                    keep = sorted_inds[:self.nms_pre]
                    cls_score = cls_score[keep]
                    bbox_pred = bbox_pred[keep]
                    anchors = anchors[keep]

                # 解码
                proposals = self.bbox_coder.decode(bbox_pred, anchors,
                                                    max_shape=img_shapes[img_idx])

                # 过滤极小框
                if self.min_bbox_size > 0:
                    w = proposals[:, 2] - proposals[:, 0]
                    h = proposals[:, 3] - proposals[:, 1]
                    valid = (w >= self.min_bbox_size) & (h >= self.min_bbox_size)
                    proposals = proposals[valid]
                    cls_score = cls_score[valid]

                mlvl_proposals.append(proposals)
                mlvl_scores.append(cls_score)

            # 合并所有层
            proposals = torch.cat(mlvl_proposals, dim=0)
            scores = torch.cat(mlvl_scores, dim=0)

            # NMS
            if proposals.numel() > 0:
                keep = nms(proposals, scores, self.nms_thr)
                proposals = proposals[keep]
                scores = scores[keep]

            # 限制每张图最大数量 (按分数降序取 topk)
            if proposals.size(0) > self.max_per_img:
                _, sorted_inds = scores.sort(descending=True, stable=True)
                proposals = proposals[sorted_inds[:self.max_per_img]]

            result_list.append(proposals)

        return result_list

    @staticmethod
    def _xywh2xyxy(boxes):
        """xywh -> xyxy"""
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return torch.stack([x1, y1, x2, y2], dim=-1)


if __name__ == '__main__':
    from detection.utils.anchor_generator import AnchorGenerator
    from detection.assigners.max_iou_assigner import MaxIoUAssigner
    from detection.bbox_coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
    from detection.losses.loss import BCELoss, SmoothL1Loss

    anchor_gen = AnchorGenerator(
        strides=[4, 8, 16, 32, 64],
        ratios=[0.5, 1.0, 2.0],
        scales=[8]
    )
    assigner = MaxIoUAssigner(pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3)
    bbox_coder = DeltaXYWHBBoxCoder(target_means=(0., 0., 0., 0.), target_stds=(1., 1., 1., 1.))
    cls_loss = BCELoss(reduction='mean')
    reg_loss = SmoothL1Loss(beta=1.0/9.0, reduction='mean')

    rpn_head = RPNHead(
        in_channels=256,
        featmap_strides=[4, 8, 16, 32, 64],
        anchor_generator=anchor_gen,
        assigner=assigner,
        bbox_coder=bbox_coder,
        cls_loss=cls_loss,
        reg_loss=reg_loss
    )

    feats = [
        torch.randn(2, 256, 160, 160),
        torch.randn(2, 256, 80, 80),
        torch.randn(2, 256, 40, 40),
        torch.randn(2, 256, 20, 20),
        torch.randn(2, 256, 10, 10),
    ]

    batch_bboxes = [
        torch.tensor([[50, 50, 100, 100], [200, 200, 150, 150]], dtype=torch.float32),
        torch.tensor([[30, 30, 80, 80]], dtype=torch.float32),
    ]
    batch_labels = [
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    ]

    losses = rpn_head.loss(feats, batch_bboxes, batch_labels)
    print("RPN losses:", losses)

    proposals = rpn_head.get_proposals(feats, [(640, 640), (640, 640)])
    for i, prop in enumerate(proposals):
        print(f"Image {i} proposals shape:", prop.shape)
