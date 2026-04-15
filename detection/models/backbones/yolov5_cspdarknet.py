import torch
import torch.nn as nn

from detection.models.yolo_blocks import *
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
# 注册机制
from heltonx.utils.register import MODELS




@MODELS.register
class YOLOv5CSPDarknet(nn.Module):
    '''YOLOv5专属Backbone
    '''
    def __init__(self, phi:str, out_layers, load_ckpt=False, froze_backbone=False):
        super().__init__()
        self.out_layers = out_layers
        '''不同尺寸的基本配置'''
        depth_dict          = {'n':0.33, 's':0.33, 'm':0.67, 'l':1.00, 'x':1.33}
        width_dict          = {'n':0.25, 's':0.50, 'm':0.75, 'l':1.00, 'x':1.25}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
        base_channels       = int(wid_mul * 64)
        base_depth          = max(round(dep_mul * 3), 1)

        '''网络组件'''
        self.stem = Conv(3, base_channels, 6, 2, 2)
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C3(base_channels * 2, base_channels * 2, base_depth),
        )
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth * 2),
        )
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            C3(base_channels * 16, base_channels * 16, base_depth),
            SPPF(base_channels * 16, base_channels * 16),
        )
        # 是否导入预训练权重
        if load_ckpt:
            # self = load_state_dict_with_prefix(self, load_ckpt)
            state_dict = torch.load(load_ckpt, map_location='cpu')
            self.load_state_dict(state_dict)
        # 是否冻结backbone
        if froze_backbone:
            for param in self.parameters():
                param.requires_grad = False



    def forward(self, x):
        p1 = self.stem(x)
        p2 = self.dark2(p1)
        p3 = self.dark3(p2)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        outs = [p1, p2, p3, p4, p5]
        return [outs[i] for i in self.out_layers]










# for test only
if __name__ == '__main__':
    '''基本配置: n s m l x'''
    phi = 's'
    loadckpt = f'ckpts/yolo/cspdarknet_{phi}_v6.1_backbone.pth'
    backbone = YOLOv5CSPDarknet(phi, out_layers=[2,3,4], load_ckpt=loadckpt, froze_backbone=False)
    # print(backbone)
    # 验证
    x = torch.rand((4, 3, 640, 640))
    outs = backbone(x)
    for out in outs: print(out.shape)

    # n:[64,  128, 256 ]
    # s:[128, 256, 512 ]
    # m:[192, 384, 768 ]
    # l:[256, 512, 1024]
    # x:[324, 640, 1280]
