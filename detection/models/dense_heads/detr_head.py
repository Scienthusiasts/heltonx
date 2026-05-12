import torch
import torch.nn as nn
import torch.distributed as dist
from heltonx.utils.register import MODELS
from heltonx.utils.utils import init_weights
from detection.utils.detr_utils import box_cxcywh_to_xyxy


class MLP(nn.Module):
    """简单 3 层 MLP (DETR 回归头使用)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@MODELS.register
class DETRHead(nn.Module):
    """DETR 检测头

    负责前向传播（分类 + 回归）和损失计算（含 Auxiliary Loss）。
    所有 decoder 层共享同一个预测 FFN（与 DETR 官方实现一致）。

    Args:
        nc (int):           前景类别数 (不含背景)
        hidden_dim (int):   Transformer 输出维度
        num_queries (int):  Object Query 数量
        num_decoder_layers (int): Decoder 层数（用于 Auxiliary Loss）
        cls_loss (nn.Module):  分类损失模块 (如 DETRCrossEntropyLoss)
        l1_loss (nn.Module):    L1 回归损失模块 (如 DETRL1Loss)
        giou_loss (nn.Module): GIoU 回归损失模块 (如 DETRGiouLoss)
        assigner (nn.Module):  样本分配器 (如 HungarianAssigner)
    """

    def __init__(self, nc, hidden_dim=256, num_queries=100, num_decoder_layers=6,
                 cls_loss=None, l1_loss=None, giou_loss=None, assigner=None,
                 cls_loss_weight=1.0, l1_loss_weight=5.0, giou_loss_weight=2.0):
        super().__init__()
        self.nc = nc
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.cls_loss_weight = cls_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.giou_loss_weight = giou_loss_weight

        # 共享的分类 FFN: hidden_dim -> nc+1 (含背景类)
        self.cls_head = nn.Linear(hidden_dim, nc + 1)
        # 共享的回归 FFN: hidden_dim -> 4 (cxcywh)
        self.reg_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # 权重初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 分类头偏置初始化 (使初始预测偏向背景)
        prior = 0.01
        nn.init.constant_(
            self.cls_head.bias[-1],
            -torch.log(torch.tensor((1 - prior) / prior))
        )

        # 损失模块
        self.cls_loss = cls_loss
        self.l1_loss = l1_loss
        self.giou_loss = giou_loss
        # 样本分配器
        self.assigner = assigner

    def forward(self, transformer_output):
        """前向传播（仅使用最终层预测）

        Args:
            transformer_output: (hs_all, init_ref) 来自 DETRTransformer
                hs_all:   [num_decoder_layers, B, num_queries, hidden_dim] 所有 Decoder 层输出（经 LayerNorm）
                init_ref: [B, num_queries, 4]

        Returns:
            cls_preds: [B, num_queries, nc+1] 最终层分类 logits
            box_preds: [B, num_queries, 4]    最终层归一化 cxcywh
        """
        hs_all, _ = transformer_output
        # 最终层（最后一层 decoder 输出，已由 decoder 内部 LayerNorm 处理）
        cls_preds = self.cls_head(hs_all[-1])
        box_preds = self.reg_head(hs_all[-1]).sigmoid()
        return cls_preds, box_preds

    def _get_num_boxes(self, batch_labels, device):
        """计算归一化因子 num_boxes（支持 DDP all_reduce）

        Args:
            batch_labels: list[Tensor], 每个 [num_gt]
            device: torch device

        Returns:
            int: GT 框总数（DDP 时为所有卡的均值）
        """
        num_boxes = sum(len(b) for b in batch_labels)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()
        else:
            num_boxes = torch.clamp(num_boxes, min=1).item()
        return num_boxes

    def _compute_single_loss(self, cls_preds, box_preds, batch_labels, batch_bboxes, num_boxes):
        """计算单层损失

        Args:
            cls_preds:     [B, num_queries, nc+1] 分类 logits
            box_preds:     [B, num_queries, 4]   归一化 cxcywh
            batch_labels:  list[Tensor], 每个 [num_gt]
            batch_bboxes:  list[Tensor], 每个 [num_gt, 4] 归一化 cxcywh
            num_boxes:     int, GT 框总数

        Returns:
            dict: cls_loss, l1_loss, giou_loss
        """
        bs = cls_preds.shape[0]
        num_queries = cls_preds.shape[1]

        # 1. 匈牙利匹配
        indices = self.assigner(
            cls_preds.detach(), box_preds.detach(),
            batch_labels, batch_bboxes
        )

        # 2. 构造分类目标：类别索引（nc 表示背景）
        cls_targets = torch.full((bs, num_queries), self.nc,
                                 dtype=torch.long, device=cls_preds.device)
        for b in range(bs):
            query_idx, gt_idx = indices[b]
            if len(query_idx) == 0:
                continue
            cls_targets[b, query_idx] = batch_labels[b][gt_idx]

        # 3. 构造回归目标
        box_targets = torch.zeros_like(box_preds)
        for b in range(bs):
            query_idx, gt_idx = indices[b]
            if len(query_idx) == 0:
                continue
            box_targets[b, query_idx] = batch_bboxes[b][gt_idx].to(box_targets.dtype)

        # 4. 匹配 mask
        matched_masks = torch.zeros(bs, num_queries, dtype=torch.bool, device=cls_preds.device)
        for b in range(bs):
            if len(indices[b][0]) > 0:
                matched_masks[b, indices[b][0]] = True

        # 5. 各子 loss（乘以权重）
        cls_loss = self.cls_loss_weight * self.cls_loss(cls_preds, cls_targets)
        l1_loss = self.l1_loss_weight * self.l1_loss(box_preds, box_targets, matched_masks, num_boxes)
        giou_loss = self.giou_loss_weight * self.giou_loss(box_preds, box_targets, matched_masks, num_boxes)

        return dict(
            cls_loss=cls_loss,
            l1_loss=l1_loss,
            giou_loss=giou_loss,
        )

    def loss(self, cls_preds, box_preds, hs_all, batch_labels, batch_bboxes):
        """计算 DETR 总损失（含 Auxiliary Loss）

        与 MMDetection 官方实现一致：
        - 每层 decoder 独立做匈牙利匹配
        - 辅助损失与主层使用相同权重（不取平均）
        - 最后一层作为主损失（loss_cls / loss_bbox / loss_iou）
        - 前面各层作为辅助损失（d0 ~ d4）
        - DDP 训练时 num_boxes 做 all_reduce

        Args:
            cls_preds:     [B, num_queries, nc+1] 最终层分类 logits
            box_preds:     [B, num_queries, 4]   最终层归一化 cxcywh
            hs_all:        [num_decoder_layers, B, num_queries, hidden_dim] 所有 Decoder 层输出
            batch_labels:  list[Tensor], 每个 [num_gt]
            batch_bboxes:  list[Tensor], 每个 [num_gt, 4] 归一化 cxcywh

        Returns:
            dict: 主损失 loss_cls/loss_bbox/loss_iou + 辅助损失 d{i}.loss_cls/loss_bbox/loss_iou
        """
        num_boxes = self._get_num_boxes(batch_labels, cls_preds.device)

        # 最终层损失（主损失，键名与 MMDetection 一致）
        losses = self._compute_single_loss(cls_preds, box_preds, batch_labels, batch_bboxes, num_boxes)
        loss_dict = {
            'loss_cls': losses['cls_loss'],
            'loss_bbox': losses['l1_loss'],
            'loss_iou': losses['giou_loss'],
        }

        # Auxiliary Loss: 前 num_decoder_layers-1 层作为辅助损失
        # MMDet 命名: d0 ~ d4 对应前 5 层，最后一层无主损失前缀
        for i in range(self.num_decoder_layers - 1):
            aux_cls = self.cls_head(hs_all[i])
            aux_box = self.reg_head(hs_all[i]).sigmoid()
            aux_losses = self._compute_single_loss(aux_cls, aux_box, batch_labels, batch_bboxes, num_boxes)
            loss_dict[f'd{i}.loss_cls'] = aux_losses['cls_loss']
            loss_dict[f'd{i}.loss_bbox'] = aux_losses['l1_loss']
            loss_dict[f'd{i}.loss_iou'] = aux_losses['giou_loss']

        return loss_dict
