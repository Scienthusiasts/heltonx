import torch
import torch.nn as nn
import torch.nn.functional as F

from detection.models.yolo_blocks import Conv, C3k2
from heltonx.utils.utils import init_weights
from heltonx.utils.register import MODELS


def _make_channels(base_ch, width, max_channels=1024):
    """根据 width 缩放系数和 max_channels 约束计算实际通道数

    与 ultralytics 一致: min(base, max_channels) * width
    即先对 base 通道数施加 max_channels 上限, 再乘以 width 缩放系数
    """
    ch = min(base_ch, max_channels)
    return max(round(ch * width), 8)


@MODELS.register
class YOLO26PAFPN(nn.Module):
    '''YOLO26 专用 PAFPN (基于 Ultralytics YOLO26 架构)

    与 YOLOv5PAFPN 的核心区别:
    - 使用 C3k2 替代 C3 模块
    - 所有 C3k2 均使用 c3k=True (与 YOLO11 不同, YOLO11 部分 c3k=False)
    - P5 输出层使用 attn=True (Bottleneck + PSABlock 串联)
    - P5 输出层使用 e=0.5
    - 使用 YOLO26 的缩放系数和 max_channels 约束

    结构 (PANet 双向融合, base 通道数):
        Backbone 输出: P3=512, P4=512, P5=1024
        Top-down:  P5 → Upsample → Concat(P4_bb) → C3k2→512
                   P4_td → Upsample → Concat(P3_bb) → C3k2→256  ← P3 输出
        Bottom-up: P3 → Conv(s=2) → Concat(P4_td) → C3k2→512    ← P4 输出
                   P4 → Conv(s=2) → Concat(P5_bb) → C3k2→1024   ← P5 输出 (attn)
    '''

    # 与 YOLO26Backbone 一致的缩放系数
    depth_dict = {'n': 0.50, 's': 0.50, 'm': 0.50, 'l': 1.00, 'x': 1.00}
    width_dict = {'n': 0.25, 's': 0.50, 'm': 1.00, 'l': 1.00, 'x': 1.50}
    max_channels_dict = {'n': 1024, 's': 1024, 'm': 512, 'l': 512, 'x': 512}

    def __init__(self, phi, num_extra_levels=0):
        """
        Args:
            phi (str): 模型尺寸, 'n'/'s'/'m'/'l'/'x'
            num_extra_levels (int): 额外下采样层数量, 例如 2 -> P6, P7
        """
        super().__init__()
        self.num_extra_levels = num_extra_levels

        dep_mul = self.depth_dict[phi]
        wid_mul = self.width_dict[phi]
        max_ch = self.max_channels_dict[phi]

        # 骨干输出通道 (base: P3=512, P4=512, P5=1024)
        bb_p3 = _make_channels(512, wid_mul, max_ch)
        bb_p4 = _make_channels(512, wid_mul, max_ch)
        bb_p5 = _make_channels(1024, wid_mul, max_ch)

        # FPN 输出通道 (base: P3=256, P4=512, P5=1024)
        fpn_p3 = _make_channels(256, wid_mul, max_ch)
        fpn_p4 = _make_channels(512, wid_mul, max_ch)
        fpn_p5 = _make_channels(1024, wid_mul, max_ch)

        # 深度缩放
        n = max(round(2 * dep_mul), 1)  # C3k2 repeats=2
        n_p5 = max(round(1 * dep_mul), 1)  # P5 输出层 C3k2 repeats=1

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # --- Top-down 路径 ---
        # Concat(P5_up, P4_bb) → C3k2 → fpn_p4
        self.td_p4_C3k2 = C3k2(bb_p5 + bb_p4, fpn_p4, n=n, c3k=True)
        # Concat(td_p4_up, P3_bb) → C3k2 → fpn_p3
        self.td_p3_C3k2 = C3k2(fpn_p4 + bb_p3, fpn_p3, n=n, c3k=True)

        # --- Bottom-up 路径 ---
        self.p3_downsample = Conv(fpn_p3, fpn_p3, 3, 2)
        # Concat(p3_down, td_p4) → C3k2 → fpn_p4
        self.bu_p4_C3k2 = C3k2(fpn_p3 + fpn_p4, fpn_p4, n=n, c3k=True)

        self.p4_downsample = Conv(fpn_p4, fpn_p4, 3, 2)
        # Concat(p4_down, P5_bb) → C3k2 → fpn_p5
        # YOLO26 的 P5 输出层使用 attn=True (Bottleneck + PSABlock 串联)
        self.bu_p5_C3k2 = C3k2(fpn_p4 + bb_p5, fpn_p5, n=n_p5, c3k=False, e=0.5, attn=True)

        # 额外的下采样层 (例如 P6, P7)
        if self.num_extra_levels > 0:
            self.extra_convs = nn.ModuleList([
                Conv(fpn_p5, fpn_p5, 3, 2)
                for _ in range(self.num_extra_levels)
            ])

        # 权重初始化
        for m in self.modules():
            init_weights(m, 'normal', 0, 0.01)

    def forward(self, x):
        """
        Args:
            x (List[Tensor]): [P3, P4, P5] 骨干输出
                P3: [B, bb_p3, H/8, W/8]
                P4: [B, bb_p4, H/16, W/16]
                P5: [B, bb_p5, H/32, W/32]

        Returns:
            Tuple[Tensor]: [P3', P4', P5'] 融合后的特征 (可能含 P6, P7)
        """
        c3, c4, c5 = x

        # Top-down
        td_p4 = self.td_p4_C3k2(torch.cat([self.upsample(c5), c4], dim=1))
        td_p3 = self.td_p3_C3k2(torch.cat([self.upsample(td_p4), c3], dim=1))

        # Bottom-up
        p3 = td_p3
        p4 = self.bu_p4_C3k2(torch.cat([self.p3_downsample(p3), td_p4], dim=1))
        p5 = self.bu_p5_C3k2(torch.cat([self.p4_downsample(p4), c5], dim=1))

        results = [p3, p4, p5]

        # 额外层
        if self.num_extra_levels > 0:
            last = results[-1]
            for conv in self.extra_convs:
                last = conv(last)
                results.append(last)

        return tuple(results)


# for test only
if __name__ == '__main__':
    phi = 'n'
    fpn = YOLO26PAFPN(phi)

    # phi='n': bb_P3=128, bb_P4=128, bb_P5=256
    # 根据所选 phi 构造正确的输入
    from detection.models.backbones.yolo26_backbone import YOLO26Backbone, _make_channels as bb_ch
    dep = YOLO26PAFPN.depth_dict[phi]
    wid = YOLO26PAFPN.width_dict[phi]
    mc = YOLO26PAFPN.max_channels_dict[phi]
    p3_ch = bb_ch(512, wid, mc)
    p4_ch = bb_ch(512, wid, mc)
    p5_ch = bb_ch(1024, wid, mc)
    c3 = torch.rand((2, p3_ch, 80, 80))
    c4 = torch.rand((2, p4_ch, 40, 40))
    c5 = torch.rand((2, p5_ch, 20, 20))
    outs = fpn([c3, c4, c5])
    for out in outs:
        print(out.shape)

    # phi='n': P3=[2,64,80,80], P4=[2,128,40,40], P5=[2,256,20,20]
    # phi='s': P3=[2,128,80,80], P4=[2,256,40,40], P5=[2,512,20,20]
    # phi='m': P3=[2,256,80,80], P4=[2,512,40,40], P5=[2,512,20,20]
