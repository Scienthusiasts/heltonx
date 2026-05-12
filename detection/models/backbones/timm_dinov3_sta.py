import torch.nn as nn
import timm
import torch.nn.functional as F
from heltonx.utils.ckpts_utils import *
from heltonx.utils.utils import init_weights
# 注册机制
from heltonx.utils.register import MODELS
from heltonx.utils.utils import multi_apply



from torchvision.models.detection import faster_rcnn



class ConvBlock(nn.Module):
    """Conv + BN + ReLU 基本卷积块
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)




class BiFusion(nn.Module):
    """
    Bi-Fusion Module
        Args:
            A: [B, C1, H1, W1] (context, e.g. semantic feature)
            B: [B, C2, H2, W2] (details, e.g. low-level feature)
        Returns:
            out: [B, C, H2, W2]
    """
    def __init__(self, A_ch, B_ch, out_ch):
        super(BiFusion, self).__init__()
        # 融合卷积
        self.fuse_conv = ConvBlock(A_ch + B_ch, out_ch, 1, 1, 0)


    def forward(self, A, B):
        # 1. 将A上采样到B的空间尺寸
        A_upsampled = F.interpolate(A, size=B.shape[2:], mode='bilinear', align_corners=False)
        # 2. 拼接通道
        fused = torch.cat([A_upsampled, B], dim=1)
        # 3. 1x1卷积融合
        out = self.fuse_conv(fused)
        return out
    



class SpatialTuningAdapter(nn.Module):
    """
    Spatial Tuning Adapter (STA)
    输入:  原始图像或浅层特征
    输出:  多尺度特征列表 (包含所有的尺度)
    """
    def __init__(self, in_ch=3, layer_dims=[128, 256, 512, 1024, 2048]):
        super(SpatialTuningAdapter, self).__init__()
        self.base_conv = ConvBlock(in_ch, layer_dims[0], k=3, s=2, p=1)
        
        # 根据 layer_dims 的长度动态生成后续卷积层
        layers = [ConvBlock(layer_dims[i], layer_dims[i+1], k=3, s=2, p=1) for i in range(len(layer_dims) - 1)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.base_conv(x)
        lvl_feats = [x]  # idx 0: 第1层尺度
        
        for layer in self.layers:
            x = layer(x)
            lvl_feats.append(x) # 追加后续尺度

        # 返回所有尺度的特征，交由外部类通过 out_layers 参数去按需切片
        return lvl_feats





@MODELS.register
class DINOv3STA(nn.Module):
    """DINOv3STA DINOv3+多尺度adapter
    """
    def __init__(self, dino_name: str, sta_layer_dims: list, fuse_layer_dims: list, 
                 out_layers: list, dino_out_indices: list, dino_ckpt: str, 
                 dino_dim: int = 384, froze_dino: bool = True):
        """
        Args:
            dino_name:        timm 中dino模型名称
            sta_layer_dims:   指定sta每一层的维度
            fuse_layer_dims:  每一层bi_fusion后的维度 (长度需对应 out_layers)
            out_layers:       指定输出 STA 中哪几层多尺度特征，如 [1, 2, 3, 4] 表示输出第2到第5层
            dino_out_indices: 提取 dino 的哪几层特征用于融合 (长度需对应 out_layers)
            dino_ckpt:        加载dino预训练权重
            dino_dim:         dino模型提取特征的维度 (小模型一般为384)
            froze_dino:       是否冻结dino骨干网络
        """
        super().__init__()
        
        # 校验维度对齐
        assert len(out_layers) == len(fuse_layer_dims), "out_layers 和 fuse_layer_dims 的长度必须一致!"
        assert len(out_layers) == len(dino_out_indices), "out_layers 和 dino_out_indices 的长度必须一致!"
        
        self.out_layers = out_layers
        
        # features_only=True 直接去掉分类头, out_indices 指定提取哪些层的特征
        self.dinov3 = timm.create_model(dino_name, pretrained=False, features_only=True, out_indices=dino_out_indices)
        self.sta = SpatialTuningAdapter(3, sta_layer_dims)
        
        # 动态构建对应的融合层
        self.bi_fusions = nn.ModuleList([
            BiFusion(dino_dim, sta_layer_dims[idx], fuse_layer_dims[i]) 
            for i, idx in enumerate(self.out_layers)
        ])
        
        # 导入dino权重
        self.dinov3 = load_state_dict_with_prefix(self.dinov3, dino_ckpt, prefixes_to_try=['model.'])
        
        # 是否冻结dinov3权重
        if froze_dino:
            for param in self.dinov3.parameters():
                param.requires_grad_(False)
                
        # 初始化
        for m in self.sta.modules():
            init_weights(m, 'normal', 0, 0.01)
        for m in self.bi_fusions.modules():
            init_weights(m, 'normal', 0, 0.01)


    def forward_single(self, dino_feat, sta_feat, i):
        """单层融合前向"""
        lvl_x = self.bi_fusions[i](dino_feat, sta_feat)
        return lvl_x


    def forward(self, x):
        """前向传播"""
        dino_x = self.dinov3(x)      # 取出 dino 多层特征
        sta_all = self.sta(x)        # 取出 sta 的所有特征
        
        # 根据 out_layers 获取目标尺度的 sta 特征
        sta_selected = [sta_all[idx] for idx in self.out_layers]
        
        n = range(len(self.out_layers))
        # 执行融合 (要求 dino_x 和 sta_selected 长度一致)
        lvl_feats = multi_apply(self.forward_single, dino_x, sta_selected, n)
        
        return lvl_feats





# for test only:
if __name__ == '__main__':

    # 配置字典
    cfgs=dict(
        type="DINOv3STA",
        dino_name="vit_small_patch16_dinov3.lvd1689m",
        sta_layer_dims=[64, 128, 256, 512, 1024],
        out_layers=[1, 2, 3, 4],            # 对应输出STA中的索引1, 2, 3, 4的层 (即2~5层特征)
        dino_out_indices=[2, 5, 8, 11],     # 提取DINO中对应的层与STA融合 (与out_layers长度一致)
        dino_dim=384,                       # 设定DINO通道尺寸
        fuse_layer_dims=[128, 256, 512, 1024],   
        dino_ckpt="ckpts/vit_small_patch16_dinov3.lvd1689m.pt",
        froze_dino=True
    )
    backbone = MODELS.build_from_cfg(cfgs)

    x = torch.randn(4, 3, 640, 640)
    x = backbone(x)

    for o in x:
        print(o.shape)



