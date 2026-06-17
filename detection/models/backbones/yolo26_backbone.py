import torch
import torch.nn as nn

from detection.models.yolo_blocks import Conv, C3k2, SPPF, C2PSA
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
from heltonx.utils.register import MODELS


def _make_channels(base_ch, width, max_channels=1024):
    """根据 width 缩放系数和 max_channels 约束计算实际通道数

    与 ultralytics 一致: min(base, max_channels) * width
    即先对 base 通道数施加 max_channels 上限, 再乘以 width 缩放系数
    """
    ch = min(base_ch, max_channels)
    return max(round(ch * width), 8)


@MODELS.register
class YOLO26Backbone(nn.Module):
    '''YOLO26 骨干网络 (基于 Ultralytics YOLO26 架构)

    核心改进:
    - C3k2 替代 YOLOv5 的 C3 模块
    - C2PSA (Position-Sensitive Attention) 位于骨干末端
    - 新的 depth/width 缩放系数和 max_channels 约束

    网络结构:
        stem:  Conv(3,c1) + Conv(c1,c2) + C3k2(c2,c3, e=0.25)
        dark3: Conv(c3,c3) + C3k2(c3,c4, e=0.25)              → P3/8 输出
        dark4: Conv(c4,c4) + C3k2(c4,c4, c3k=True)             → P4/16 输出
        dark5: Conv(c4,c8) + C3k2(c8,c8, c3k=True) + SPPF + C2PSA → P5/32 输出

    注意: ultralytics 对 m/l/x 规模强制所有 C3k2 的 c3k=True,
         仅 n/s 规模的 stem/dark3 使用 c3k=False (Bottleneck, k=3x3).
         dark4/dark5 对所有规模均使用 c3k=True.
    '''

    # YOLO26 的缩放系数 (与 YOLOv5 不同)
    depth_dict = {'n': 0.50, 's': 0.50, 'm': 0.50, 'l': 1.00, 'x': 1.00}
    width_dict = {'n': 0.25, 's': 0.50, 'm': 1.00, 'l': 1.00, 'x': 1.50}
    max_channels_dict = {'n': 1024, 's': 1024, 'm': 512, 'l': 512, 'x': 512}

    def __init__(self, phi: str, out_layers, load_ckpt=False, froze_backbone=False):
        """
        Args:
            phi (str): 模型尺寸, 'n'/'s'/'m'/'l'/'x'
            out_layers (list): 输出层索引, 通常 [2,3,4] 对应 P3/P4/P5
            load_ckpt: 预训练权重路径
            froze_backbone: 是否冻结骨干参数
        """
        super().__init__()
        self.out_layers = out_layers

        dep_mul = self.depth_dict[phi]
        wid_mul = self.width_dict[phi]
        max_ch = self.max_channels_dict[phi]

        # 计算各层通道数 (YAML base: 64, 128, 256, 512, 1024)
        c1 = _make_channels(64, wid_mul, max_ch)   # stem conv out
        c2 = _make_channels(128, wid_mul, max_ch)   # dark2 conv out
        c3 = _make_channels(256, wid_mul, max_ch)   # dark3 P3 输出
        c4 = _make_channels(512, wid_mul, max_ch)   # dark4 P4 输出
        c8 = _make_channels(1024, wid_mul, max_ch)  # dark5 P5 输出

        # 深度缩放: C3k2 和 C2PSA 的 n 参数
        n = max(round(2 * dep_mul), 1)  # YAML 中 repeats=2
        n_psa = max(round(2 * dep_mul), 1)  # C2PSA repeats=2

        # ultralytics 对 m/l/x 规模强制所有 C3k2 的 c3k=True
        # (即使 YAML 中写 c3k=False，parse_model 也会覆盖为 True)
        # n/s: c3k=False → Bottleneck (cv1=3x3, cv2=3x3)
        # m/l/x: c3k=True → C3k 子模块 (cv1/cv2/cv3=1x1 + 内部 Bottleneck)
        c3k_forced = phi in ('m', 'l', 'x')

        # --- 网络结构 ---
        # stem: Conv(3, c1, 3, 2) + Conv(c1, c2, 3, 2) + C3k2(c2, c3, e=0.25)
        self.stem = nn.Sequential(
            Conv(3, c1, 3, 2),
            Conv(c1, c2, 3, 2),
            C3k2(c2, c3, n=n, c3k=c3k_forced, e=0.25),
        )

        # dark3: P3/8 输出
        self.dark3 = nn.Sequential(
            Conv(c3, c3, 3, 2),
            C3k2(c3, c4, n=n, c3k=c3k_forced, e=0.25),
        )

        # dark4: P4/16 输出
        self.dark4 = nn.Sequential(
            Conv(c4, c4, 3, 2),
            C3k2(c4, c4, n=n, c3k=True),
        )

        # dark5: P5/32 输出 (含 SPPF + C2PSA)
        self.dark5 = nn.Sequential(
            Conv(c4, c8, 3, 2),
            C3k2(c8, c8, n=n, c3k=True),
            SPPF(c8, c8, 5),
            C2PSA(c8, c8, n=n_psa),
        )

        # 预训练权重
        if load_ckpt:
            self = load_state_dict_with_prefix(self, load_ckpt)

        # 冻结骨干
        if froze_backbone:
            for param in self.parameters():
                param.requires_grad = False

    def _ultralytics_key_map(self):
        """返回 ultralytics 数字索引键到本模型命名键的映射

        ultralytics 的 DetectionModel 使用扁平 nn.Sequential，键名为 model.0, model.1, ...
        本模型使用命名模块: stem, dark3, dark4, dark5，每个也是 nn.Sequential。
        映射关系:
            ultralytics layer 0  -> stem.0   (Conv)
            ultralytics layer 1  -> stem.1   (Conv)
            ultralytics layer 2  -> stem.2   (C3k2)
            ultralytics layer 3  -> dark3.0  (Conv)
            ultralytics layer 4  -> dark3.1  (C3k2)
            ultralytics layer 5  -> dark4.0  (Conv)
            ultralytics layer 6  -> dark4.1  (C3k2)
            ultralytics layer 7  -> dark5.0  (Conv)
            ultralytics layer 8  -> dark5.1  (C3k2)
            ultralytics layer 9  -> dark5.2  (SPPF)
            ultralytics layer 10 -> dark5.3  (C2PSA)
        """
        return {
            '0': 'stem.0',
            '1': 'stem.1',
            '2': 'stem.2',
            '3': 'dark3.0',
            '4': 'dark3.1',
            '5': 'dark4.0',
            '6': 'dark4.1',
            '7': 'dark5.0',
            '8': 'dark5.1',
            '9': 'dark5.2',
            '10': 'dark5.3',
        }

    def forward(self, x):
        """前向传播

        Args:
            x (Tensor): 输入图像 [B, 3, H, W]

        Returns:
            List[Tensor]: 按 out_layers 顺序返回多尺度特征图
        """
        p1 = self.stem(x)
        p3 = self.dark3(p1)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        outs = [p1, p3, p4, p5]
        return [outs[i] for i in self.out_layers]


# for test only
if __name__ == '__main__':
    phi = 's'
    backbone = YOLO26Backbone(phi, out_layers=[1, 2, 3])
    x = torch.rand((2, 3, 640, 640))
    outs = backbone(x)
    for out in outs:
        print(out.shape)

    # phi='n': P3=[2,128,80,80], P4=[2,128,40,40], P5=[2,256,20,20]
    # phi='s': P3=[2,256,80,80], P4=[2,256,40,40], P5=[2,512,20,20]
    # phi='m': P3=[2,512,80,80], P4=[2,512,40,40], P5=[2,512,20,20]
    # phi='l': P3=[2,512,80,80], P4=[2,512,40,40], P5=[2,512,20,20]
    # phi='x': P3=[2,768,80,80], P4=[2,768,40,40], P5=[2,768,20,20]
