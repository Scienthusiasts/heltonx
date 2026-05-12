import torch
import torch.nn as nn
import math
from heltonx.utils.register import MODELS
from heltonx.utils.utils import init_weights


class CustomBatchNorm2d(nn.Module):
    """当 bs=1 时跳过 BN，避免 BN 报错"""

    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        if x.size(0) == 1:
            return x
        else:
            return self.bn(x)


@MODELS.register
class BBoxConvHead(nn.Module):
    """Faster R-CNN BBox Head (Conv-based)

    对 RoI Align 后的固定尺寸特征进行分类和精细回归。
    使用多层 1x1 卷积 + BN + ReLU 替代 FC 层提取特征，再通过
    自适应平均池化将 nxn 特征压缩为 1x1，最后接分类和回归 FC 分支。

    Args:
        in_channels (int): 输入特征通道数 (RoI Align 输出通道)
        roi_feat_size (int or Tuple[int, int]): RoI 特征空间尺寸, 默认 7
        num_classes (int): 前景类别数 (不含背景)
        conv_out_channels (int): 卷积层输出通道数, 默认 1024
        with_cls (bool): 是否启用分类分支
        with_reg (bool): 是否启用回归分支
        reg_class_agnostic (bool): 回归是否与类别无关, False 则为 class-aware
    """

    def __init__(self, in_channels, roi_feat_size=7, num_classes=80,
                 conv_out_channels=1024, num_convs=2,
                 with_cls=True, with_reg=True, reg_class_agnostic=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_out_channels = conv_out_channels
        self.num_convs = num_convs
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.reg_class_agnostic = reg_class_agnostic

        # 共享卷积层
        convs = []
        ch = in_channels
        for _ in range(num_convs):
            convs.append(self._make_conv_block(ch, conv_out_channels))
            ch = conv_out_channels
        self.convs = nn.Sequential(*convs)

        # 自适应平均池化: [N, C, H, W] -> [N, C, 1, 1]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 分类分支
        if self.with_cls:
            self.fc_cls = nn.Linear(conv_out_channels, num_classes + 1)

        # 回归分支
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(conv_out_channels, out_dim_reg)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, 'normal', 0, 0.001)
            elif isinstance(m, nn.Conv2d):
                init_weights(m, 'normal', 0, 0.01)

        # 分类头偏置初始化：让初始预测倾向背景类，降低初始分类 loss
        if self.with_cls:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.fc_cls.bias, bias_value)

    @staticmethod
    def _make_conv_block(in_channels, out_channels):
        """1x1 Conv + BN + ReLU"""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False))
        # layers.append(CustomBatchNorm2d(out_channels))
        # layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播

        Args:
            x (Tensor): [N, in_channels, roi_h, roi_w]

        Returns:
            cls_score (Tensor or None): [N, num_classes+1] 分类 logits
            bbox_pred (Tensor or None): [N, 4] 或 [N, num_classes*4] 回归 delta
        """
        x = self.convs(x)       # [N, conv_out_channels, roi_h, roi_w]
        x = self.avg_pool(x)    # [N, conv_out_channels, 1, 1]
        x = x.view(x.size(0), -1)  # [N, conv_out_channels]

        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None

        return cls_score, bbox_pred


if __name__ == '__main__':
    bbox_head = BBoxConvHead(
        in_channels=256,
        roi_feat_size=7,
        num_classes=80,
        conv_out_channels=1024,
        num_convs=2,
        reg_class_agnostic=False
    )
    x = torch.randn(128, 256, 7, 7)
    cls_score, bbox_pred = bbox_head(x)
    print("cls_score:", cls_score.shape)   # [128, 81]
    print("bbox_pred:", bbox_pred.shape)   # [128, 320]
