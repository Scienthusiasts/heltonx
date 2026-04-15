import torch
import torch.nn as nn
import torch.distributed as dist
from utils.utils import init_weights
# 注册机制
from heltonx.utils.register import MODELS
from detection.utils.fcos_utils import *
from detection.losses import *
from detection.utils.yolov5_utils import *




class ScaleExp(nn.Module):
    '''指数放缩可学习模块,
       通过使用指数变换，可以确保预测结果总是非负数, 同时, 学习一个放缩系数 self.scale 使得网络能够动态地调整回归值的范围.
    '''
    def __init__(self, init_value=1.0):
        super(ScaleExp,self).__init__()
        # 可学习缩放参数
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self,x):
        # 对预测的特征图的数值再进行一个指数放缩, 并且放缩的参数是可学习的
        return torch.exp(x * self.scale)



class DecoupledFCOSHead(nn.Module):
    '''解耦的FCOS预测头模块 (每个尺度独享一个，不共享参数)
    '''
    def __init__(self, in_channel, nc):
        super(DecoupledFCOSHead, self).__init__()
        self.nc = nc
        
        '''定义网络结构'''
        # 分类回归头解耦
        self.cls_head = nn.Conv2d(in_channel, self.nc, kernel_size=3, padding=1)
        self.cnt_head = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(in_channel, 4, kernel_size=3, padding=1)
        
        # 回归头上的可学习放缩系数 (因为当前头不跨尺度共享，所以只需要1个ScaleExp)
        self.scale_exp = ScaleExp(1.0)

        # 权重初始化
        for m in self.modules():
            init_weights(m, 'normal', 0, 0.01)
            
        # 对分类头的偏置专门的初始化方式(使其一开始倾向于背景)
        prior = 0.01
        nn.init.constant_(self.cls_head.bias, -math.log((1 - prior) / prior))

    def forward(self, x):
        cls_logit = self.cls_head(x)
        cnt_logit = self.cnt_head(x)
        reg_pred  = self.scale_exp(self.reg_head(x))
        
        return cls_logit, cnt_logit, reg_pred
    












@MODELS.register
class YOLOv5FCOSHead(nn.Module):
    def __init__(self, phi, nc, img_size, cls_loss:nn.Module, reg_loss:nn.Module, cnt_loss:nn.Module, assigner:nn.Module, layers_num=3):
        """
        """
        super(YOLOv5FCOSHead, self).__init__()
        '''基本配置'''
        depth_dict          = {'n': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'n': 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
        base_channels       = int(wid_mul * 64)  

        # 使用几层特征图
        self.layers_num = layers_num
        self.nc = nc
        self.img_size = img_size
        s = [4, 8, 16, 16, 16]
        
        # 包含每个尺度的head (解耦的FCOS头，一个尺度一个head, 不共享)
        # FCOS是Anchor-free的，因此移除了anchors和anchors_mask参数
        self.p_heads = nn.ModuleList([DecoupledFCOSHead(base_channels*s[i], nc) for i in range(layers_num)])
        
        '''正负样本分配与损失函数'''
        self.assigner = assigner
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.cnt_loss = cnt_loss


    def forward(self, x):
        '''前向传播
        '''
        cls_logits, cnt_logits, reg_preds = [], [], []
        for i in range(self.layers_num):
            cls_logit, cnt_logit, reg_pred = self.p_heads[i](x[i]) 
            cls_logits.append(cls_logit)
            cnt_logits.append(cnt_logit)
            reg_preds.append(reg_pred)

        return cls_logits, cnt_logits, reg_preds

    def _reshape_cat_out(self, preds_list, channels):
        """将多尺度预测列表展平并拼接
           [bs, C, H, W] -> [bs, H*W, C] -> concat -> [bs * total_anchor_num, C]
        """
        reshaped_preds = []
        for p in preds_list:
            bs, c, h, w = p.shape
            reshaped_preds.append(p.permute(0, 2, 3, 1).reshape(bs, -1, c))
        # 按照空间维度拼接，最后整体展平以便于计算loss
        return torch.cat(reshaped_preds, dim=1).reshape(-1, channels)


    def loss(self, x, batch_bboxes, batch_labels):
        '''前向传播+计算损失 (FCOS逻辑)
        '''
        # 1. 取得多尺度前向传播结果
        # 返回均为List，里面包含每层的Tensor
        cls_logits_list, cnt_logits_list, reg_preds_list = self.forward(x)
        
        # 2. FCOS 正负样本分配
        # 对应位置标记为-1的是负样本 [bs * total_anchor_num, 1] 等
        cls_targets, cnt_targets, reg_targets = self.assigner(batch_bboxes, batch_labels)
        
        # 获得正样本(bool) [bs * total_anchor_num]
        pos_mask = (cnt_targets > -1).reshape(-1)
        
        # 3. 调整预测结果的形状(将不同尺度的预测结果展平并拼在一起)
        # [bs * total_anchor_num, cls_num]
        cls_preds = self._reshape_cat_out(cls_logits_list, self.nc)
        # [bs * total_anchor_num, 1]
        cnt_preds = self._reshape_cat_out(cnt_logits_list, 1)
        # [bs * total_anchor_num, 4]
        reg_preds = self._reshape_cat_out(reg_preds_list, 4)

        # 4. 计算损失
        '''分类损失(所有样本均参与计算)'''
        num_pos = torch.sum(pos_mask).clamp_(min=1).float()
        # 生成one_hot标签
        cls_targets = (torch.arange(0, self.nc, device=cls_targets.device)[None, :] == cls_targets).float()
        cls_loss = self.cls_loss(cls_preds, cls_targets).sum() / num_pos
        
        '''centerness损失(正样本才计算)'''
        cnt_loss = self.cnt_loss(cnt_preds[pos_mask], cnt_targets[pos_mask])
        
        '''回归损失(正样本才计算)'''
        reg_preds_pos, reg_targets_pos = reg_preds[pos_mask], reg_targets[pos_mask]
        # FCOS 的 L,T 坐标系处理
        reg_preds_pos[:, :2] *= -1
        reg_targets_pos[:, :2] *= -1
        reg_loss = self.reg_loss(reg_preds_pos, reg_targets_pos)

        '''loss统一为字典格式输出'''
        losses = dict(
            cls_loss = cls_loss,
            cnt_loss = cnt_loss,
            reg_loss = reg_loss
        )
        return losses
















if __name__ == '__main__':
    # ---------------- 模拟依赖环境 ---------------- #
    # 模拟简单的占位组件，仅用于实例化网络，不参与单纯的 forward 测试
    class MockComponent(nn.Module):
        def forward(self, *args, **kwargs):
            pass

    # ---------------- 基本配置 ---------------- #
    phi = 's'
    num_classes = 80
    img_size = [640, 640]
    batch_size = 4
    
    # YOLOv5 FPN 各层输出的通道数计算逻辑 (Phi='s' 时，wid_mul=0.5 -> base=32 -> 128, 256, 512)
    # P3(stride=8) -> 128
    # P4(stride=16) -> 256
    # P5(stride=32) -> 512
    head_in_channel = {
        'n': [64,  128, 256 ],
        's': [128, 256, 512, 1024, 2048 ],
        'm': [192, 384, 768 ],
        'l': [256, 512, 1024],
        'x': [324, 640, 1280]
    }[phi]

    # ---------------- 实例化检测头 ---------------- #
    # FCOS 为 anchor-free，无需传入 anchors 和 anchors_mask
    head = YOLOv5FCOSHead(
        phi=phi,
        nc=num_classes,
        img_size=img_size,
        cls_loss=MockComponent(),
        reg_loss=MockComponent(),
        cnt_loss=MockComponent(),
        assigner=MockComponent(),
        layers_num=5
    )

    # ---------------- 构造伪造的多尺度特征图 ---------------- #
    # 输入图像大小 640x640 下对应的特征图尺寸: 80x80, 40x40, 20x20
    p3 = torch.rand((batch_size, head_in_channel[0], 80, 80))
    p4 = torch.rand((batch_size, head_in_channel[1], 40, 40))
    p5 = torch.rand((batch_size, head_in_channel[2], 20, 20))
    p6 = torch.rand((batch_size, head_in_channel[3], 10, 10))
    p7 = torch.rand((batch_size, head_in_channel[4], 5, 5))
    
    fpn_features = [p3, p4, p5, p6, p7]

    # ---------------- 前向传播测试 ---------------- #
    # 返回分别为三个尺度的 分类预测、中心度预测、回归预测 (均为 List[Tensor])
    cls_logits_list, cnt_logits_list, reg_preds_list = head(fpn_features)

    # ---------------- 验证输出维度 ---------------- #
    print(f"--- FCOS Head Forward Outputs (Batch Size: {batch_size}, Classes: {num_classes}) ---")
    
    strides = [8, 16, 32, 64, 128]
    feat_sizes = [80, 40, 20, 10, 5]
    
    for i in range(5):
        print(f"\n[Scale P{i+3}] Stride: {strides[i]}, Feature Map: {feat_sizes[i]}x{feat_sizes[i]}")
        # 分类分支期望维度: [bs, nc, h, w] -> [4, 80, h, w]
        print(f"  Classification Head (cls) : {cls_logits_list[i].shape}") 
        # 中心度分支期望维度: [bs, 1, h, w] -> [4, 1, h, w]
        print(f"  Centerness Head (cnt)     : {cnt_logits_list[i].shape}") 
        # 回归分支期望维度: [bs, 4, h, w] -> [4, 4, h, w] (FCOS 预测 l, t, r, b 4个值)
        print(f"  Regression Head (reg)     : {reg_preds_list[i].shape}")