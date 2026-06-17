import torch
import torch.nn as nn
import timm
import math
import torch.distributed as dist
from utils.utils import init_weights
# 注册机制
from heltonx.utils.register import MODELS
from heltonx.utils.utils import multi_apply
from detection.utils.fcos_utils import *
from detection.losses import *



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
        return torch.exp(x*self.scale)



@MODELS.register
class FCOSHead(nn.Module):
    '''FCOS的预测头模块(不同尺度共享同一个特征头)
    '''
    def __init__(self, nc, in_channel, cnt_loss:nn.Module, cls_loss:nn.Module, reg_loss:nn.Module, assigner:nn.Module):
        super(FCOSHead,self).__init__()
        self.nc=nc
        cls_branch=[]
        reg_branch=[]
        '''定义网络结构'''
        # 预测头之前的特征提取部分
        for _ in range(4):
            # 分类分支特征提取(cls和centerness)
            cls_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            cls_branch.append(nn.GroupNorm(32, in_channel)),
            cls_branch.append(nn.ReLU(True))
            # 回归分支特征提取(reg)
            reg_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            reg_branch.append(nn.GroupNorm(32, in_channel)),
            reg_branch.append(nn.ReLU(True))

        # 预测头之前的共享特征提取
        self.cls_conv=nn.Sequential(*cls_branch)
        self.reg_conv=nn.Sequential(*reg_branch)
        # 分类回归头解耦
        # 分类头
        self.cls_head = nn.Conv2d(in_channel, self.nc, kernel_size=3, padding=1)
        # centerness头
        self.cnt_head = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)
        # 回归头
        self.reg_head = nn.Conv2d(in_channel, 4, kernel_size=3, padding=1)
        # 回归头上的可学习放缩系数
        self.scale_exp = nn.ModuleList([ScaleExp(1) for _ in range(5)])
        '''定义损失函数'''
        self.cnt_loss = cnt_loss
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss

        # 权重初始化
        for m in self.modules():
            init_weights(m, 'normal', 0, 0.01)
        # 对分类头的偏置专门的初始化方式(目的是, 一开始网络的分类会倾向于背景, 可以从一个合理的状态开始训练):
        prior = 0.01
        nn.init.constant_(self.cls_head.bias, -math.log((1 - prior) / prior))

        '''定义样本分配策略'''
        self.assigner = assigner
    


    def forward_single(self, lvl, lvl_x):
        """单个尺度的前向传播
            Args:
                lvl:   当前处理的是第几个尺度
                lvl_x: 每个尺度的特征 [bs, c, lvl_h, lvl_w]
            Returns:
                cls_logit: [bs, nc, w, h]
                cnt_logit: [bs, 1, w, h]
                reg_pred:  [bs, 4=(l, t, r, b), w, h]
        """
        cls_conv_out = self.cls_conv(lvl_x)
        reg_conv_out = self.reg_conv(lvl_x)
        cls_logit = self.cls_head(cls_conv_out)
        cnt_logit = self.cnt_head(cls_conv_out)
        reg_pred  = self.scale_exp[lvl](self.reg_head(reg_conv_out))
        return cls_logit, cnt_logit, reg_pred


    def forward(self, x):
        """前向传播，使用 multi_apply 并行处理多尺度特征层
        """
        n = range(len(x))
        cls_logits, cnt_logits, reg_preds = multi_apply(self.forward_single, n, x)
        return cls_logits, cnt_logits, reg_preds




    def loss(self, fpn_feat, batch_bboxes, batch_labels):
        """
            Args:
                fpn_feat:
                batch_bboxes:
                batch_labels:
        """
        # head部分前向
        # [[bs, cls_num, w, h],...,[...]] [[bs, 1, w, h],...,[...]] [[bs, 4, w, h],...,[...]]
        cls_logits, cnt_logits, reg_preds = self.forward(fpn_feat)
        '''FCOS的正负样本分配'''
        # 对应位置标记为-1的是负样本 [bs * total_anchor_num, 1] [bs * total_anchor_num, 1] [bs * total_anchor_num, 4]
        cls_targets, cnt_targets, reg_targets = self.assigner(batch_bboxes, batch_labels)
        # 获得正样本(bool) [bs, total_anchor_num]
        pos_mask = (cnt_targets > -1).reshape(-1)
        '''计算损失'''
        # 调整预测结果的形状(将不同尺度的预测结果拼在一起):
        # [[bs, cls_num, 80, 80],...,[[bs, cls_num, 5, 5]]] -> [bs * total_anchor_num, cls_num]
        cls_preds = reshape_cat_out(cls_logits).reshape(-1, self.nc)
        # [[bs, 1, 80, 80],...,[[bs, 1, 5, 5]]] -> [bs * total_anchor_num, 1]
        cnt_preds = reshape_cat_out(cnt_logits).reshape(-1, 1)
        # [[bs, 4, 80, 80],...,[[bs, 4, 5, 5]]] -> [bs * total_anchor_num, 4]
        reg_preds = reshape_cat_out(reg_preds).reshape(-1, 4)
        # 计算损失:
        '''分类损失(所有样本均参与计算)'''
        # 计算batch里每张图片的正样本数量 [bs,]
        num_pos = torch.sum(pos_mask).clamp_(min=1).float()
        # 生成one_hot标签
        cls_targets  = (torch.arange(0, self.nc, device=cls_targets.device)[None,:] == cls_targets).float()
        cls_loss = self.cls_loss(cls_preds, cls_targets).sum() / torch.sum(num_pos)
        '''centerness损失(正样本才计算)'''
        if pos_mask.sum() > 0:
            cnt_loss = self.cnt_loss(cnt_preds[pos_mask], cnt_targets[pos_mask])
        else:
            cnt_loss = cnt_preds.sum() * 0
        '''回归损失(正样本才计算)'''
        if pos_mask.sum() > 0:
            # 计算GIoU loss
            reg_preds, reg_targets = reg_preds[pos_mask], reg_targets[pos_mask]
            reg_preds[:, :2]*=-1
            reg_targets[:, :2]*=-1
            reg_loss = self.reg_loss(reg_preds, reg_targets)
        else:
            reg_loss = reg_preds.sum() * 0
        '''loss以字典形式回传'''
        loss = dict(
            cls_loss = cls_loss,
            cnt_loss = cnt_loss,
            reg_loss = reg_loss
        )
        return loss  





















# for test only
if __name__ == '__main__':
    num_cls = 15
    fpn_out_channel = 256
    bs = 4
    size = [80, 40, 20, 10, 5]
    # 模拟FPN输出:
    x = [torch.rand((bs, fpn_out_channel, lvl_size, lvl_size)) for lvl_size in size]
    head = FCOSHead(num_cls, fpn_out_channel)
    cls_logits, cnt_logits, reg_preds = head(x)

    for cls, cnt, reg in zip(cls_logits, cnt_logits, reg_preds):
        print(cls.shape, cnt.shape, reg.shape)

    # torch.Size([4, 15, 80, 80]) torch.Size([4, 1, 80, 80]) torch.Size([4, 4, 80, 80])
    # torch.Size([4, 15, 40, 40]) torch.Size([4, 1, 40, 40]) torch.Size([4, 4, 40, 40])
    # torch.Size([4, 15, 20, 20]) torch.Size([4, 1, 20, 20]) torch.Size([4, 4, 20, 20])
    # torch.Size([4, 15, 10, 10]) torch.Size([4, 1, 10, 10]) torch.Size([4, 4, 10, 10])
    # torch.Size([4, 15, 5, 5]) torch.Size([4, 1, 5, 5]) torch.Size([4, 4, 5, 5])