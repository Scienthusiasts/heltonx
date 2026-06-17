import torch
import torch.nn as nn
import math
from heltonx.utils.register import MODELS
from heltonx.utils.utils import init_weights


@MODELS.register
class YOLO26Head(nn.Module):
    """YOLO26 双头检测 (网络结构, 不含 loss 逻辑)

    双分支:
    - o2m (one-to-many): 训练用, 多正样本
    - o2o (one-to-one):  推理用, NMS-free

    loss 逻辑已解耦到 loss_fn 模块 (YOLO26Loss)

    Args:
        phi (str): 'n'/'s'/'m'/'l'/'x'
        nc (int): 类别数
        layers_num (int): 特征层数
        loss_fn (nn.Module): YOLO26Loss 实例
    """

    # 与 YOLO26Backbone 一致的缩放系数
    width_dict = {'n': 0.25, 's': 0.50, 'm': 1.00, 'l': 1.00, 'x': 1.50}
    max_channels_dict = {'n': 1024, 's': 1024, 'm': 512, 'l': 512, 'x': 512}

    def __init__(self, phi, nc, img_size=(640, 640), layers_num=3, loss_fn=None):
        super().__init__()
        wid_mul = self.width_dict[phi]
        max_ch = self.max_channels_dict[phi]

        self.nl = layers_num
        self.nc = nc
        self.strides = [8, 16, 32]

        # P3/P4/P5 通道数 (与 YOLO26PAFPN 输出一致)
        # ultralytics 公式: min(base, max_ch) * width
        ch_list = [
            max(round(min(256, max_ch) * wid_mul), 8),   # P3 (FPN base=256)
            max(round(min(512, max_ch) * wid_mul), 8),   # P4 (FPN base=512)
            max(round(min(1024, max_ch) * wid_mul), 8),  # P5 (FPN base=1024)
        ]

        self.p_heads = nn.ModuleList([
            DualConvHead(in_channels=ch_list[i], nc=nc,
                         stride=self.strides[i])
            for i in range(self.nl)
        ])

        self.loss_fn = loss_fn

    def forward(self, x):
        o2m_out, o2o_out = [], []
        for i, head in enumerate(self.p_heads):
            m, o = head(x[i])
            o2m_out.append(m)
            o2o_out.append(o)
        return {'o2m': o2m_out, 'o2o': o2o_out}

    def forward_o2o(self, x):
        return [head.forward_o2o(x[i]) for i, head in enumerate(self.p_heads)]

    def loss(self, x, batch_bboxes, batch_labels):
        o2m_preds, o2o_preds = [], []
        for i, head in enumerate(self.p_heads):
            m, o = head(x[i])
            o2m_preds.append(m)
            o2o_preds.append(o)
        return self.loss_fn(o2m_preds, o2o_preds, batch_bboxes, batch_labels)

    def update_progressive(self, cur_epoch):
        """epoch 级别更新 progressive loss 权重"""
        if self.loss_fn is not None:
            self.loss_fn.update_progressive(cur_epoch)


class DualConvHead(nn.Module):
    """单层双分支卷积头 (o2m + o2o)

    与官方 Ultralytics YOLOv8 一致:
    - 输出 4+nc 通道 (cx,cy,w,h + cls) — 无 objectness
    - 卷积权重: normal(0, 0.01)
    - bbox bias: 0
    - cls bias: log(5/nc/(640/stride)^2) (初始sigmoid≈0.01, 偏向背景)
    """

    def __init__(self, in_channels, nc, stride=8):
        super().__init__()
        self.out_ch = 4 + nc
        self.o2m_head = nn.Conv2d(in_channels, self.out_ch, 1)
        self.o2o_head = nn.Conv2d(in_channels, self.out_ch, 1)

        for head in [self.o2m_head, self.o2o_head]:
            init_weights(head, 'normal', 0, 0.01)
            cls_bias = math.log(5 / nc / (640 / stride) ** 2)
            head.bias.data[0:2] = 0.0    # dx,dy bias=0 → sigmoid(0)=0.5, center at grid cell center
            head.bias.data[2:4] = 0.0    # dw,dh bias=0 → (0.5*4)²=4 cells, 初始框适中
            head.bias.data[4:] = cls_bias  # channels 4+: cls

    def forward(self, x):
        return self.o2m_head(x), self.o2o_head(x)

    def forward_o2o(self, x):
        return self.o2o_head(x)
