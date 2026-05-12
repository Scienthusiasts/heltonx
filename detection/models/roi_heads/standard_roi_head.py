import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from heltonx.utils.register import MODELS
from detection.utils.roi_utils import bbox2roi, multilevel_roi_align


@MODELS.register
class StandardRoIHead(nn.Module):
    """Standard RoI Head (Faster R-CNN 第二阶段)

    组合 RoI Align + BBox Head，对 RPN 生成的 proposals 进行
    精细分类和边界框回归。

    Args:
        bbox_roi_extractor (nn.Module): RoI 特征提取器 (RoIAlign)
        bbox_head (nn.Module): BBox Head (分类 + 回归)
        assigner (nn.Module): proposals 正负样本分配器
        bbox_coder (nn.Module): bbox 编解码器
        cls_loss (nn.Module): 分类损失函数 (CrossEntropyLoss)
        reg_loss (nn.Module): 回归损失函数 (SmoothL1Loss)
        num_samples (int): 每张图采样 proposal 数, 默认 512
        pos_fraction (float): 正样本比例, 默认 0.25
        score_thr (float): 推理时置信度阈值, 默认 0.05
        max_per_img (int): 推理时每张图最大保留框数, 默认 100
        nms_thr (float): 推理时 NMS IoU 阈值, 默认 0.5
    """

    def __init__(self, bbox_roi_extractor, bbox_head, assigner, bbox_coder,
                 cls_loss, reg_loss, num_samples=512, pos_fraction=0.25,
                 score_thr=0.05, max_per_img=100, nms_thr=0.5):
        super().__init__()
        self.bbox_roi_extractor = bbox_roi_extractor
        self.bbox_head = bbox_head
        self.assigner = assigner
        self.bbox_coder = bbox_coder
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss

        self.num_samples = num_samples
        self.pos_samples = int(num_samples * pos_fraction)
        self.neg_samples = num_samples - self.pos_samples

        self.score_thr = score_thr
        self.max_per_img = max_per_img
        self.nms_thr = nms_thr

    def forward(self, feats, proposals):
        """前向传播 (RoI Align + BBox Head)

        Args:
            feats (List[Tensor]): FPN 特征图
            proposals (Tensor): [N, 5] 格式为 [batch_ind, x1, y1, x2, y2]

        Returns:
            cls_score (Tensor): [N, num_classes+1]
            bbox_pred (Tensor): [N, 4] 或 [N, num_classes*4]
        """
        rois = proposals
        roi_feats = self.bbox_roi_extractor(feats, rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        return cls_score, bbox_pred

    def loss(self, feats, proposals, batch_bboxes, batch_labels):
        """计算 RoI Head 损失

        Args:
            feats (List[Tensor]): FPN 特征图
            proposals (List[Tensor]): 每张图 proposals [num_proposals, 4] xyxy
            batch_bboxes (List[Tensor]): GT bbox, xywh 格式
            batch_labels (List[Tensor]): GT 类别

        Returns:
            dict: {"roi_cls_loss": ..., "roi_reg_loss": ...}
        """
        device = feats[0].device
        bs = len(proposals)

        # 收集采样的 proposals 及其标签
        sampled_proposals = []
        sampled_labels = []
        sampled_reg_targets = []
        sampled_pos_inds = []
        total_samples = 0

        for img_idx in range(bs):
            gt_bboxes = batch_bboxes[img_idx]
            if gt_bboxes.numel() > 0:
                gt_bboxes_xyxy = self._xywh2xyxy(gt_bboxes)
            else:
                gt_bboxes_xyxy = gt_bboxes.new_zeros((0, 4))
            gt_labels = batch_labels[img_idx]

            img_proposals = proposals[img_idx]

            # 将 GT 框加入 proposals (Faster R-CNN 标准做法)
            # 训练初期 RPN 可能无法产生高质量 proposals，直接加入 GT 框确保
            # RoI Head 始终有高质量正样本（GT 框与自身 IoU=1.0，回归目标为 0）
            if gt_bboxes_xyxy.numel() > 0:
                img_proposals = torch.cat([img_proposals, gt_bboxes_xyxy], dim=0)

            # GT 标签 +1：数据集输出 0-indexed 标签 (0~79)，
            # 但 BBoxHead 有 81 个输出 (0=背景, 1~80=前景)，CrossEntropyLoss 期望 0=背景
            # 因此需要将前景标签从 0~79 映射到 1~80，保留 0 给背景
            gt_labels = gt_labels + 1

            # 分配
            assigned_gt_inds, assigned_labels = self.assigner(
                img_proposals, gt_bboxes_xyxy, gt_labels
            )

            pos_inds = (assigned_gt_inds > 0).nonzero(as_tuple=False).squeeze(1)
            neg_inds = (assigned_gt_inds == 0).nonzero(as_tuple=False).squeeze(1)

            # 采样
            if pos_inds.numel() > self.pos_samples:
                perm = torch.randperm(pos_inds.numel(), device=device)[:self.pos_samples]
                pos_inds = pos_inds[perm]
            if neg_inds.numel() > self.neg_samples:
                perm = torch.randperm(neg_inds.numel(), device=device)[:self.neg_samples]
                neg_inds = neg_inds[perm]

            img_sampled_inds = torch.cat([pos_inds, neg_inds])
            img_sampled_proposals = img_proposals[img_sampled_inds]
            img_sampled_labels = assigned_labels[img_sampled_inds]

            # 回归目标 (仅正样本有效)
            img_reg_targets = img_proposals.new_zeros((img_sampled_inds.numel(), 4))
            if pos_inds.numel() > 0:
                pos_proposals = img_proposals[pos_inds]
                pos_gt_inds = assigned_gt_inds[pos_inds] - 1
                pos_gt_bboxes = gt_bboxes_xyxy[pos_gt_inds]
                pos_reg_targets = self.bbox_coder.encode(pos_gt_bboxes, pos_proposals)
                # 填入采样后的位置
                pos_mask = torch.zeros(img_sampled_inds.numel(), dtype=torch.bool, device=device)
                pos_mask[:pos_inds.numel()] = True
                img_reg_targets[pos_mask] = pos_reg_targets

            # 添加 batch_ind
            batch_inds = torch.full((img_sampled_proposals.size(0), 1), img_idx,
                                    dtype=torch.float32, device=device)
            img_sampled_proposals = torch.cat([batch_inds, img_sampled_proposals], dim=1)

            sampled_proposals.append(img_sampled_proposals)
            sampled_labels.append(img_sampled_labels)
            sampled_reg_targets.append(img_reg_targets)
            sampled_pos_inds.append(total_samples + torch.arange(pos_inds.numel(), device=device))
            total_samples += img_sampled_inds.numel()

        if total_samples == 0:
            # 没有任何样本，返回0损失
            losses = dict(
                roi_cls_loss=feats[0].new_tensor(0., requires_grad=True),
                roi_reg_loss=feats[0].new_tensor(0., requires_grad=True)
            )
            return losses

        # 拼接所有图
        all_proposals = torch.cat(sampled_proposals, dim=0)  # [N, 5]
        all_labels = torch.cat(sampled_labels, dim=0)        # [N]
        all_reg_targets = torch.cat(sampled_reg_targets, dim=0)  # [N, 4]
        all_pos_inds = torch.cat(sampled_pos_inds, dim=0)    # [num_pos]

        # RoI Align + BBox Head
        cls_score, bbox_pred = self.forward(feats, all_proposals)

        # 分类损失
        cls_loss = self.cls_loss(cls_score, all_labels)

        # 回归损失 (仅正样本，mean over 正样本)
        if all_pos_inds.numel() > 0:
            pos_bbox_pred = bbox_pred[all_pos_inds]
            pos_reg_targets = all_reg_targets[all_pos_inds]

            # class-aware: 提取对应类别的回归预测
            if not self.bbox_head.reg_class_agnostic:
                pos_labels = all_labels[all_pos_inds] - 1  # 0-based, 正样本标签 >=1
                num_classes = self.bbox_head.num_classes
                pos_bbox_pred = pos_bbox_pred.view(-1, num_classes, 4)
                pos_bbox_pred = pos_bbox_pred[torch.arange(pos_labels.numel()), pos_labels]

            # reg_loss 配置为 reduction='mean'，直接取正样本上的均值
            # 与参考仓库一致：仅除以正样本数，不除以总采样数
            reg_loss = self.reg_loss(pos_bbox_pred, pos_reg_targets)
        else:
            reg_loss = cls_score.new_tensor(0.)

        losses = dict(
            roi_cls_loss=cls_loss,
            roi_reg_loss=reg_loss
        )
        return losses

    def get_bboxes(self, feats, proposals, img_shapes):
        """推理：对 proposals 进行分类和回归，输出最终检测结果

        Args:
            feats (List[Tensor]): FPN 特征图
            proposals (List[Tensor]): 每张图 proposals [num, 4] xyxy
            img_shapes (List[Tuple[int, int]]): 每张图尺寸 (H, W)

        Returns:
            List[Tensor]: 每张图检测结果 [num_det, 6=(x1,y1,x2,y2,score,class)]
        """
        # 拼接 proposals
        rois = bbox2roi(proposals)
        if rois.numel() == 0:
            return [rois.new_zeros((0, 6)) for _ in range(len(proposals))]

        cls_score, bbox_pred = self.forward(feats, rois)
        num_proposals_per_img = [p.size(0) for p in proposals]

        # softmax 得到概率
        scores = F.softmax(cls_score, dim=1)  # [N, num_classes+1]

        results = []
        start = 0
        for i, num in enumerate(num_proposals_per_img):
            end = start + num
            img_scores = scores[start:end]       # [num, num_classes+1]
            img_bbox_pred = bbox_pred[start:end]  # [num, 4] 或 [num, num_classes*4]
            img_proposals = proposals[i]          # [num, 4]
            img_shape = img_shapes[i]

            # 排除背景类
            img_scores = img_scores[:, 1:]  # [num, num_classes]

            # 对每个 proposal，取最高分类别
            max_scores, labels = img_scores.max(dim=1)  # [num]

            # 过滤低置信度
            valid_mask = max_scores >= self.score_thr
            if not valid_mask.any():
                results.append(img_proposals.new_zeros((0, 6)))
                start = end
                continue

            valid_scores = max_scores[valid_mask]
            valid_labels = labels[valid_mask]
            valid_bbox_pred = img_bbox_pred[valid_mask]
            valid_proposals = img_proposals[valid_mask]

            # 解码 bbox (class-aware)
            if not self.bbox_head.reg_class_agnostic:
                num_classes = self.bbox_head.num_classes
                valid_bbox_pred = valid_bbox_pred.view(-1, num_classes, 4)
                valid_bbox_pred = valid_bbox_pred[torch.arange(valid_labels.numel()), valid_labels]

            bboxes = self.bbox_coder.decode(valid_bbox_pred, valid_proposals,
                                            max_shape=img_shape)

            # 按类别 NMS
            det_results = []
            unique_labels = valid_labels.unique()
            for cls in unique_labels:
                cls_mask = valid_labels == cls
                cls_bboxes = bboxes[cls_mask]
                cls_scores = valid_scores[cls_mask]

                keep = nms(cls_bboxes, cls_scores, self.nms_thr)
                cls_bboxes = cls_bboxes[keep]
                cls_scores = cls_scores[keep]
                cls_labels = cls.new_full((keep.numel(),), cls, dtype=torch.long)

                det = torch.cat([
                    cls_bboxes,
                    cls_scores.unsqueeze(1),
                    cls_labels.unsqueeze(1).float()
                ], dim=1)
                det_results.append(det)

            if len(det_results) == 0:
                results.append(img_proposals.new_zeros((0, 6)))
            else:
                det_results = torch.cat(det_results, dim=0)
                # 按分数排序，取 topk
                if det_results.size(0) > self.max_per_img:
                    _, topk_inds = det_results[:, 4].sort(descending=True, stable=True)
                    det_results = det_results[topk_inds[:self.max_per_img]]
                results.append(det_results)

            start = end

        return results

    @staticmethod
    def _xywh2xyxy(boxes):
        """xywh -> xyxy"""
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return torch.stack([x1, y1, x2, y2], dim=-1)

