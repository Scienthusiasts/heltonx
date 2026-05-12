import torch
import torch.nn as nn
import math
from heltonx.utils.register import MODELS
from heltonx.utils.utils import init_weights


@MODELS.register
class BBoxFCHead(nn.Module):
    """Faster R-CNN BBox Head

    对 RoI Align 后的固定尺寸特征进行分类和精细回归。
    使用两个共享的 FC 层提取特征，然后分别送入分类和回归分支。

    Args:
        in_channels (int): 输入特征通道数 (RoI Align 输出通道)
        roi_feat_size (int or Tuple[int, int]): RoI 特征空间尺寸, 默认 7
        num_classes (int): 前景类别数 (不含背景)
        fc_out_channels (int): FC 层输出维度, 默认 1024
        num_fcs (int): FC 层数量, 默认 2
        with_cls (bool): 是否启用分类分支
        with_reg (bool): 是否启用回归分支
        reg_class_agnostic (bool): 回归是否与类别无关, False 则为 class-aware
    """

    def __init__(self, in_channels, roi_feat_size=7, num_classes=80,
                 fc_out_channels=1024, num_fcs=2,
                 with_cls=True, with_reg=True, reg_class_agnostic=False):
        super().__init__()
        self.in_channels = in_channels
        if isinstance(roi_feat_size, int):
            roi_feat_size = (roi_feat_size, roi_feat_size)
        self.roi_feat_size = roi_feat_size
        self.num_classes = num_classes
        self.fc_out_channels = fc_out_channels
        self.num_fcs = num_fcs
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.reg_class_agnostic = reg_class_agnostic

        # ROI 特征展平后的维度
        self.in_features = in_channels * roi_feat_size[0] * roi_feat_size[1]

        # 共享 FC 层
        fcs = []
        in_dim = self.in_features
        for _ in range(num_fcs):
            fcs.append(nn.Linear(in_dim, fc_out_channels))
            fcs.append(nn.ReLU(inplace=True))
            in_dim = fc_out_channels
        self.fcs = nn.Sequential(*fcs)

        # 分类分支
        if self.with_cls:
            self.fc_cls = nn.Linear(fc_out_channels, num_classes + 1)

        # 回归分支
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(fc_out_channels, out_dim_reg)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, 'normal', 0, 0.001)
            elif isinstance(m, nn.Conv2d):
                init_weights(m, 'normal', 0, 0.001)

        # 分类头偏置初始化：让初始预测倾向背景类，降低初始分类 loss
        if self.with_cls:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.fc_cls.bias, bias_value)

    def forward(self, x):
        """前向传播

        Args:
            x (Tensor): [N, in_channels, roi_h, roi_w]

        Returns:
            cls_score (Tensor or None): [N, num_classes+1] 分类 logits
            bbox_pred (Tensor or None): [N, 4] 或 [N, num_classes*4] 回归 delta
        """
        # 展平
        x = x.view(x.size(0), -1)
        x = self.fcs(x)

        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None

        return cls_score, bbox_pred


if __name__ == '__main__':
    bbox_head = BBoxFCHead(
        in_channels=256,
        roi_feat_size=7,
        num_classes=80,
        fc_out_channels=1024,
        reg_class_agnostic=False
    )
    x = torch.randn(128, 256, 7, 7)
    cls_score, bbox_pred = bbox_head(x)
    print("cls_score:", cls_score.shape)   # [128, 81]
    print("bbox_pred:", bbox_pred.shape)   # [128, 320]
