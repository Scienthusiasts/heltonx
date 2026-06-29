import torch
import torch.nn as nn
import copy
import math
from heltonx.utils.register import MODELS


def _conv_bn_act(c1, c2, k=1, s=1, g=1):
    """Conv2d + BatchNorm2d + SiLU (与官方 ultralytics Conv 一致)"""
    p = k // 2 if k > 1 else 0
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True),
    )


@MODELS.register
class YOLO26OBBHead(nn.Module):
    """YOLO26-OBB 双头检测 (网络结构, 不含 loss 逻辑)

    ★★★ 与官方 OBB26 完全一致的多层 head 结构:
    - box_head: Conv(ch,c2,3) → Conv(c2,c2,3) → Conv2d(c2,4,1)  (3 层)
    - cls_head: DWConv→Conv→DWConv→Conv→Conv2d(c3,nc,1)          (5 层)
    - angle_head: Conv(ch,c4,3) → Conv(c4,c4,3) → Conv2d(c4,1,1) (3 层)

    输出格式:
    - box: [bs, 4, N] ltrb 原始 logits (无 sigmoid, 无 DFL)
    - cls: [bs, nc, N] 分类 logits
    - angle: [bs, 1, N] 角度 logits (弧度, 无 sigmoid)

    Args:
        phi (str): 'n'/'s'/'m'/'l'/'x'
        nc (int): 类别数
        layers_num (int): 特征层数
        loss_fn (nn.Module): YOLO26OBBLoss 实例
    """

    width_dict = {'n': 0.25, 's': 0.50, 'm': 1.00, 'l': 1.00, 'x': 1.50}
    max_channels_dict = {'n': 1024, 's': 1024, 'm': 512, 'l': 512, 'x': 512}

    def __init__(self, phi, nc, img_size=(640, 640), layers_num=3, loss_fn=None):
        super().__init__()
        wid_mul = self.width_dict[phi]
        max_ch = self.max_channels_dict[phi]

        self.nl = layers_num
        self.nc = nc
        self.strides = [8, 16, 32]

        ch_list = [
            max(round(min(256, max_ch) * wid_mul), 8),
            max(round(min(512, max_ch) * wid_mul), 8),
            max(round(min(1024, max_ch) * wid_mul), 8),
        ]

        self.p_heads = nn.ModuleList([
            DualOBBConvHead(in_channels=ch_list[i], nc=nc,
                            stride=self.strides[i])
            for i in range(self.nl)
        ])

        self.loss_fn = loss_fn

    def forward(self, x):
        o2m_out = {'box': [], 'cls': [], 'angle': []}
        o2o_out = {'box': [], 'cls': [], 'angle': []}
        for i, head in enumerate(self.p_heads):
            m, o = head(x[i])
            for k in ('box', 'cls', 'angle'):
                o2m_out[k].append(m[k])
                o2o_out[k].append(o[k])
        return {'o2m': o2m_out, 'o2o': o2o_out}

    def forward_o2o(self, x):
        """推理用: 只走 o2o 头, 返回 dict 格式"""
        o2o_out = {'box': [], 'cls': [], 'angle': []}
        for i, head in enumerate(self.p_heads):
            pred = head.forward_o2o(x[i])
            for k in ('box', 'cls', 'angle'):
                o2o_out[k].append(pred[k])
        return o2o_out

    def loss(self, x, batch_bboxes, batch_labels):
        o2m_preds = {'box': [], 'cls': [], 'angle': []}
        o2o_preds = {'box': [], 'cls': [], 'angle': []}
        for i, head in enumerate(self.p_heads):
            # ★ o2m: 正常梯度回传到 backbone/FPN (与官方一致)
            m = head.forward_o2m(x[i])
            for k in ('box', 'cls', 'angle'):
                o2m_preds[k].append(m[k])
            # ★ o2o: detach FPN 特征 (与官方 one2one = forward_head(x_detach) 一致)
            o = head.forward_o2o(x[i].detach())
            for k in ('box', 'cls', 'angle'):
                o2o_preds[k].append(o[k])
        return self.loss_fn(o2m_preds, o2o_preds, batch_bboxes, batch_labels)

    def update_progressive(self, cur_epoch):
        if self.loss_fn is not None:
            self.loss_fn.update_progressive(cur_epoch)


class DualOBBConvHead(nn.Module):
    """多层双分支 OBB 卷积头 (o2m + o2o)

    ★★★ 与官方 OBB26 (继承 OBB→Detect) 一致的 head 结构:
    - box_head: Conv(ch,c2,3)+BN+SiLU → Conv(c2,c2,3)+BN+SiLU → Conv2d(c2,4,1)
    - cls_head: DWConv(ch,ch,3)+BN+SiLU → Conv(ch,c3,1)+BN+SiLU
               → DWConv(c3,c3,3)+BN+SiLU → Conv(c3,c3,1)+BN+SiLU
               → Conv2d(c3,nc,1)
    - angle_head: Conv(ch,c4,3)+BN+SiLU → Conv(c4,c4,3)+BN+SiLU → Conv2d(c4,1,1)

    Bias 初始化 (与官方 bias_init 一致):
    - box 最后一层 bias = 2.0
    - cls 最后一层 bias = log(5/nc/(640/stride)²)
    - angle 最后一层 bias = 0.0
    """

    def __init__(self, in_channels, nc, stride=8):
        super().__init__()
        self.nc = nc

        # --- 通道数 (与官方 Detect 一致) ---
        c2 = max(16, in_channels // 4, 4)   # box 中间通道 (reg_max=1 → 4)
        c3 = max(in_channels, min(nc, 100))  # cls 中间通道
        c4 = max(in_channels // 4, 1)        # angle 中间通道 (ne=1)

        # --- o2m heads (多层 Conv+BN+SiLU) ---
        self.o2m_box_head = nn.Sequential(
            _conv_bn_act(in_channels, c2, 3),
            _conv_bn_act(c2, c2, 3),
            nn.Conv2d(c2, 4, 1),
        )
        self.o2m_cls_head = nn.Sequential(
            _conv_bn_act(in_channels, in_channels, 3, g=in_channels),
            _conv_bn_act(in_channels, c3, 1),
            _conv_bn_act(c3, c3, 3, g=c3),
            _conv_bn_act(c3, c3, 1),
            nn.Conv2d(c3, nc, 1),
        )
        self.o2m_angle_head = nn.Sequential(
            _conv_bn_act(in_channels, c4, 3),
            _conv_bn_act(c4, c4, 3),
            nn.Conv2d(c4, 1, 1),
        )

        # --- o2o heads (deep copy, 与官方 one2one_cv2/cv3/cv4 一致) ---
        self.o2o_box_head = copy.deepcopy(self.o2m_box_head)
        self.o2o_cls_head = copy.deepcopy(self.o2m_cls_head)
        self.o2o_angle_head = copy.deepcopy(self.o2m_angle_head)

        # ★ Bias 初始化 (与官方 bias_init 一致)
        self._bias_init(stride)

    def _bias_init(self, stride):
        """与官方 Detect.bias_init() 一致的 bias 初始化"""
        cls_bias = math.log(5 / self.nc / (640 / stride) ** 2)
        for box_head, cls_head, angle_head in [
            (self.o2m_box_head, self.o2m_cls_head, self.o2m_angle_head),
            (self.o2o_box_head, self.o2o_cls_head, self.o2o_angle_head),
        ]:
            # box: 最后一层 Conv2d bias = 2.0 (与官方一致)
            box_head[-1].bias.data[:] = 2.0
            # cls: 最后一层 Conv2d bias = log(5/nc/(640/stride)²)
            cls_head[-1].bias.data[:self.nc] = cls_bias
            # angle: 最后一层 Conv2d bias = 0.0 (raw logits → angle ≈ 0)
            angle_head[-1].bias.data[:] = 0.0

    def forward(self, x):
        return self.forward_o2m(x), self.forward_o2o(x)

    def forward_o2m(self, x):
        return {
            'box': self.o2m_box_head(x),
            'cls': self.o2m_cls_head(x),
            'angle': self.o2m_angle_head(x),
        }

    def forward_o2o(self, x):
        return {
            'box': self.o2o_box_head(x),
            'cls': self.o2o_cls_head(x),
            'angle': self.o2o_angle_head(x),
        }
