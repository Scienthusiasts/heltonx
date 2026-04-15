import torch
import torch.nn as nn
import torch.nn.functional as F

from detection.models.yolo_blocks import *
from heltonx.utils.utils import init_weights
from heltonx.utils.register import MODELS





@MODELS.register
class YOLOv5PAFPN(nn.Module):
    '''Feature Pyramid Network
    '''
    def __init__(self, phi, num_extra_levels=0):
        """
        Args:
            phi (str): 模型尺寸，如 'n', 's', 'm', 'l', 'x'
            num_extra_levels (int): 额外下采样层数量, 例如 2 -> P6, P7
        """
        super(YOLOv5PAFPN, self).__init__()
        self.num_extra_levels = num_extra_levels
        
        '''不同尺寸的基本配置'''
        depth_dict          = {'n':0.33, 's':0.33, 'm':0.67, 'l':1.00, 'x':1.33}
        width_dict          = {'n':0.25, 's':0.50, 'm':0.75, 'l':1.00, 'x':1.25}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
        base_channels       = int(wid_mul * 64)
        base_depth          = max(round(dep_mul * 3), 1)

        '''网络结构'''
        self.upsample           = nn.Upsample(scale_factor=2, mode="nearest")
        self.c5_conv            = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.t5_c4_C3           = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)
        self.conv_t4            = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.t3_C3              = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)
        self.p3_downsample_conv = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.p3_t4_C3           = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        self.p4_downsample_conv = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.p4_t5_C3           = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 额外的下采样层 (例如 P6, P7)
        if self.num_extra_levels > 0:
            self.extra_convs = nn.ModuleList([
                # 保持与 P5 相同的输出通道数进行 3x3 步长为 2 的卷积下采样
                Conv(base_channels *16, base_channels *16, 3, 2)
                for i in range(self.num_extra_levels)
            ])

        # 权重初始化
        for m in self.modules():
            init_weights(m, 'normal', 0, 0.01)


    def _upsample_cat(self, x, y):
        '''将特征图x上采样到特征图y的大小(两倍)并与y拼接
        '''
        # 按照通道维度拼接
        return torch.cat((self.upsample(x), y), dim=1)
    

    def forward(self, x):
        # 对于输入图像大小=640x640, c3.channel=512x80x80, c4.channel=1024x40x40, c5.channel=2048x20x20 (resnet50)
        c3, c4, c5 = x
        
        # 上采样融合 (Top-down)
        t5 = self.c5_conv(c5)
        t4 = self.conv_t4(self.t5_c4_C3(self._upsample_cat(t5, c4)))
        t3 = self._upsample_cat(t4, c3)
        
        # 下采样融合 (Bottom-up)
        p3 = self.t3_C3(t3)
        p4 = self.p3_t4_C3(torch.cat([self.p3_downsample_conv(p3), t4], 1))
        p5 = self.p4_t5_C3(torch.cat([self.p4_downsample_conv(p4), t5], 1))

        # 基础的三层特征
        results = [p3, p4, p5]

        # 如果需要提取额外的尺度特征，则对最顶层的特征（p5）连续进行下采样
        if self.num_extra_levels > 0:
            last = results[-1]
            for conv in self.extra_convs:
                last = conv(last)
                results.append(last)

        # 统一返回 Tuple 格式（兼容原代码返回 p3, p4, p5 的拆包形式，或者支持不定长的接收形式）
        return tuple(results)






# for test only
if __name__ == '__main__':
    phi = 's'
    # FPN
    fpn = YOLOv5PAFPN(phi, num_extra_levels=2)
    c3 = torch.rand((4, 128, 80, 80))
    c4 = torch.rand((4, 256, 40, 40))
    c5 = torch.rand((4, 512, 20, 20))
    outs = fpn([c3,c4,c5])
    for out in outs: print(out.shape)

    # n:[64,  128, 256 ]
    # s:[128, 256, 512 ]
    # m:[192, 384, 768 ]
    # l:[256, 512, 1024]
    # x:[324, 640, 1280]
