import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from torch.nn import functional as F
import cv2
import torch.nn as nn
import os
import math
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt






def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv









def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p




class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1=c1 * 4, c2=c2, k=k, s=s, p=p, g=g, act=act)

    def forward(self, x):
        # 320, 320, 12 => 320, 320, 64
        return self.conv(
            # 640, 640, 3 => 320, 320, 12
            torch.cat(
                [
                    x[..., ::2, ::2], 
                    x[..., 1::2, ::2], 
                    x[..., ::2, 1::2], 
                    x[..., 1::2, 1::2]
                ], 1
            )
        )






class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))




class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))




class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward_chunk(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))







class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))






class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))



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
            # self.load_state_dict(torch.load(load_ckpt))
            self = load_state_dict_with_prefix(self, load_ckpt)
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




@MODELS.register
class YOLOv5PAFPN(nn.Module):
    '''Feature Pyramid Network
    '''
    def __init__(self, phi):
        super(YOLOv5PAFPN, self).__init__()
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
        # 上采样融合
        t5 = self.c5_conv(c5)
        t4 = self.conv_t4(self.t5_c4_C3(self._upsample_cat(t5, c4)))
        t3 = self._upsample_cat(t4, c3)
        # 下采样融合
        p3 = self.t3_C3(t3)
        p4 = self.p3_t4_C3(torch.cat([self.p3_downsample_conv(p3), t4], 1))
        p5 = self.p4_t5_C3(torch.cat([self.p4_downsample_conv(p4), t5], 1))

        return p3, p4, p5





@MODELS.register
class YOLOv5Head(nn.Module):
    def __init__(self, phi, nc, img_size, anchors, anchors_mask, cls_loss:nn.Module, box_loss:nn.Module, obj_loss:nn.Module, assigner:nn.Module, layers_num=3, label_smoothing=0):
        """
        """
        super(YOLOv5Head, self).__init__()
        '''基本配置'''
        depth_dict          = {'n': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'n': 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
        base_channels       = int(wid_mul * 64)  
        base_depth          = max(round(dep_mul * 3), 1)  

        # 自适应调整不同损失的权重，在COCO数据集下，默认回归损失权重0.05, 分类损失权重1 obj损失权重0.5
        self.box_ratio = 0.05
        self.obj_ratio = 1   #* (img_size[0] * img_size[1]) / (640 ** 2)
        self.cls_ratio = 0.5 #* (self.num_classes / 80)
        print(f'box_loss_ratio:{self.box_ratio} | obj_loss_ratio:{self.obj_ratio} | cls_loss_ratio:{self.cls_ratio}')

        # 使用几层特征图
        self.layers_num = layers_num
        s = [4, 8, 16]
        # 包含每个尺度的head(一个尺度一个head, 不共享)
        self.p_heads = nn.ModuleList([CoupledConvHead(i, nc, img_size, anchors, base_channels*s[i], anchors_mask, cls_loss, box_loss, obj_loss, label_smoothing) for i in range(layers_num)])
        '''正负样本分配'''
        self.assigner = assigner
        self.anchors = anchors
        self.img_size = img_size
        self.anchors_mask = anchors_mask
        self.nc = nc

    def forward(self, x):
        '''前向传播
        '''
        preds = []
        for i in range(self.layers_num):
            pred = self.p_heads[i](x[i]) 
            preds.append(pred)

        return preds

    def loss(self, x, batch_bboxes, batch_labels):
        '''前向传播+计算损失
        '''
        '''正负样本分配'''
        y_trues = [[] for _ in range(self.layers_num)]
        for bboxes, labels in zip(batch_bboxes, batch_labels):
            # coco格式转成YOLO格式(xywh -> norm(cxcywh)):
            bboxes[:, 0] += bboxes[:, 2] / 2
            bboxes[:, 1] += bboxes[:, 3] / 2
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / self.img_size[1]
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / self.img_size[0]
            # len(y_true)=3(三个尺度特征), y_true[i] = [[3(每个尺度3个anchor), h_i, w_i, 5+nc(回归和分类一起耦合预测)], ..., [...]]
            y_true = self.assigner.assgin_single(bboxes.cpu().numpy(), labels.cpu().numpy(), bbox_attrs=5+self.nc)
            # 按尺度append
            for i in range(self.layers_num):
                y_trues[i].append(y_true[i])
        # y_trues = [[bs, 3, h_i, w_i, 5+nc], ..., [...]]
        y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]

        box_loss, cls_loss, obj_loss = 0, 0, 0
        # 遍历每个尺度, 计算每个尺度的损失
        for i in range(self.layers_num):
            lvl_box_loss, lvl_cls_loss, lvl_obj_loss = self.p_heads[i].loss(x[i], y_trues[i]) 
            box_loss += lvl_box_loss
            cls_loss += lvl_cls_loss
            obj_loss += lvl_obj_loss

        '''loss统一为字典格式输出'''
        losses = dict(
            box_loss = box_loss * self.box_ratio,
            cls_loss = cls_loss * self.cls_ratio,
            obj_loss = obj_loss * self.obj_ratio
        )
        return losses




class CoupledConvHead(nn.Module):
    def __init__(self, l, cat_nums, img_size, anchors, in_channels, anchors_mask, cls_loss:nn.Module, box_loss:nn.Module, obj_loss:nn.Module, label_smoothing=0):
        '''Head
            Args:

            Returns:
                None
        '''
        super(CoupledConvHead, self).__init__()
        # 当前head提取fpn哪一层特征(0:P3 1:P4 2:P5)
        self.l = l
        self.label_smoothing = label_smoothing
        self.anchors_mask = anchors_mask
        self.anchors = anchors
        self.img_size = img_size
        self.num_classes = cat_nums
        # 不同特征层的obj损失权重不同 p3,p4,p5, 对应大目标权重更低，小目标权重更高
        self.balance = [4, 1.0, 0.4]

        '''损失函数'''
        self.cls_loss = cls_loss
        self.box_loss = box_loss
        self.obj_loss = obj_loss  

        '''网络部分, YOLOv5 head是耦合的'''
        self.head = nn.Conv2d(in_channels, len(anchors_mask) * (5 + self.num_classes), 1)
        '''初始化权重'''
        init_weights(self.head, 'normal', 0, 0.01)




    def forward(self, x):
        '''前向传播
        '''
        predict = self.head(x) 
        return predict 



    def loss(self, fpn_single_feat, y_true):
        # 前向，获得网络预测结果
        input = self.forward(fpn_single_feat)
        #  获得bs，特征层的高和宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        # stride_h = stride_w = 32、16、8 (下采样率)
        stride_h = self.img_size[0] / in_h
        stride_w = self.img_size[1] / in_w
        # 此时获得的scaled_anchors大小是相对于特征层的
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        # torch.Size([bs, 255, w, h]) -> torch.Size([bs, 3, w, h, 85]) (85 : cx, cy, w, h, obj_score, cls_score=80)
        prediction = input.view(bs, len(self.anchors_mask[self.l]), self.num_classes+5, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # cx, cy, w, h, obj_score, cls_score通过simoid限制到(0,1)之间
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.sigmoid(prediction[..., 2]) 
        h = torch.sigmoid(prediction[..., 3]) 
        conf = prediction[..., 4]
        pred_cls = prediction[..., 5:]

        # 将预测结果进行解码, 即将预测offset作用到anchors上
        # pred_boxes.shape = torch.Size([bs, 3, w, h, 4])
        pred_boxes = YOLOv5Reg2Box(bs, self.l, x, y, h, w, self.anchors_mask, scaled_anchors, in_h, in_w)

        y_true = y_true.type_as(x)
        
        box_loss, cls_loss = 0, 0
        # 如果有正样本才算回归和分类损失
        pos_samples_mask = y_true[..., 4]==1
        if torch.sum(pos_samples_mask) != 0:
            '''定位损失(直接用的giou) [bs, 3, w, h]'''
            box_loss = self.box_loss(pred_boxes, y_true[..., :4])
            giou = 1 - box_loss
            box_loss = box_loss[pos_samples_mask]
            box_loss = box_loss.mean()

            '''分类损失(只对属于正样本的grid计算梯度)'''
            # cls_gt.shape = [nums_gt, 10] (one-hot)
            cls_gt = smooth_labels(y_true[..., 5:][pos_samples_mask], self.label_smoothing, self.num_classes)
            cls_loss = self.cls_loss(pred_cls[pos_samples_mask], cls_gt)
            # obj正样本对应位置的预测值设置为这个位置的预测框与GT的giou
            tobj = torch.where(pos_samples_mask, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))
        else:
            tobj = torch.zeros_like(y_true[..., 4])
        '''目标损失(当前网格是否有目标)'''
        obj_loss = self.obj_loss(conf, tobj) * self.balance[self.l]

        return box_loss, cls_loss, obj_loss













@MODELS.register
class YOLOv5(nn.Module):
    '''完整YOLOv5网络架构
    '''
    def __init__(self, img_size, backbone:nn.Module, fpn:nn.Module, heads:nn.Module, anchors, anchors_mask, nc, nms_score_thr, nms_iou_thr, nms_agnostic, load_ckpt):
        super(YOLOv5, self).__init__()
        self.img_size = img_size
        self.nms_score_thr = nms_score_thr
        self.nms_iou_thr = nms_iou_thr
        # 类别数
        self.nc = nc
        self.anchors = np.array(anchors)
        self.anchors_mask = anchors_mask
        self.nms_agnostic = nms_agnostic
        self.nms = NMS()
        '''网络基本组件'''
        self.backbone = backbone
        self.fpn = fpn
        self.heads = heads
        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)


    def forward(self, datas, return_loss=True):
        '''一个batch的前向流程 
        Args:
            datas[0]: batch_imgs:   一个batch里的图像      [bs, 3, H, W]
            datas[1]: batch_bboxes: 一个batch里的GT框      [[gt_nums_per_img, 4=(x, y, w, h)], ..., [...]]
            datas[2]: batch_labels: 一个batch里的GT框类别  [[gt_nums_per_img], ..., [...]]
            return_loss:  只前向或计算损失
        Returns:
            losses: 所有损失组成的列表 
            p_predict:
        '''
        if return_loss:
            batch_imgs, batch_bboxes, batch_labels = datas[0], datas[1], datas[2]
            # 前向过程
            backbone_feat = self.backbone(batch_imgs)
            p = self.fpn(backbone_feat)
            # 计算损失
            losses = self.heads.loss(p, batch_bboxes, batch_labels)
            return losses
        else:
            backbone_feat = self.backbone(datas)
            p3, p4, p5 = self.fpn(backbone_feat)
            p = [p3, p4, p5]
            p_predict = self.heads(p)
            return p_predict




    def infer(self, image:torch.tensor, agnostic=False, vis_heatmap=False, save_vis_path=None):
        '''推理一张图/一帧
            Args:
                image:  读取的图像 [1, 3, H, W]
            Returns:
                boxes:       网络回归的box坐标    [obj_nums, 4=(x0, y0, x1, y1)]
                box_scores:  网络预测的box置信度  [obj_nums]
                box_classes: 网络预测的box类别    [obj_nums]
        '''
        img_size = image.shape[2:]
        with torch.no_grad():
            '''网络推理得到最原始的未解码未nms的结果'''
            # p3, p4, p5
            predicts = self.forward(image, return_loss=False)
            '''利用Head的预测结果对RPN proposals进行微调+解码(解码到原图尺寸下的绝对坐标xyxy) 获得预测框'''
            # torch.Size([1, 1200, 85])
            # torch.Size([1, 4800, 85])
            # torch.Size([1, 19200, 85])
            decode_predicts = inferDecodeBox(predicts, img_size, self.nc, self.anchors, self.anchors_mask)
            '''计算nms'''
            # torch.cat(decode_predicts, 1) : torch.Size([1, 25200, 85])
            # decode_predicts = self.nms(torch.cat(decode_predicts, 1), self.nms_score_thr, self.nms_iou_thr, self.nms_agnostic)[0]
            decode_predicts = non_max_suppression(torch.cat(decode_predicts, 1), img_size, conf_thres=self.nms_score_thr, nms_thres=self.nms_iou_thr, agnostic=agnostic)
            # 图像里没预测出目标的情况:
            if len(decode_predicts) == 0 : return [],[],[]
            box_classes = np.array(decode_predicts[0][:, 6], dtype = 'int32')
            box_scores = decode_predicts[0][:, 4] * decode_predicts[0][:, 5]
            # xyxy
            boxes = decode_predicts[0][:, :4]
            '''是否可视化obj heatmap'''
            # if vis_heatmap:vis_YOLOv5_heatmap(predicts, [W, H], self.img_size, image, box_classes, save_vis_path=save_vis_path)

            return boxes, box_scores, box_classes










def inferDecodeBox(inputs, input_shape, num_classes, anchors, anchors_mask):
    '''YOLOv5将offset作用到anchor上进行解码得到最终预测结果(推理时用)
        # Args:
            - inputs:       多尺度特征图, inputs.shape = [1, 75, 80, 80], [1, 75, 40, 40], [1, 75, 30, 30]
            - input_shape:  网络接受的输入尺寸
            - num_classes:  类别数
            - anchors:      先验框(9个)
            - anchors_mask: 用于选取当前尺度特征图用哪些尺寸的先验框
        # Returns:
            - outputs: 存储不同尺度下的预测结果 [[...], [...], [...]] outputs[i].shape = [bs, num_anchors, 5+num_cls]
    '''    
    outputs = []
    for i, input in enumerate(inputs):
        batch_size      = input.size(0)
        input_height    = input.size(2)
        input_width     = input.size(3)
        # 输入为640x640时, stride_h = stride_w = 32 16 8
        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width
        # scaled_anchors是anchors相对于特征图的尺寸
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in anchors]
        # 将预测结果reshape [1, 3*(5+num_cls), w, h] -> [1, 3, w, h, 5+num_cls]
        prediction = input.view(batch_size, len(anchors_mask[i]), 5 + num_classes, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        '''得到offset(通过sigmoid将offset的范围限制在(0,1)之间)'''
        # anchors的中心位置的调整(平移因子)(offsets)
        dx = torch.sigmoid(prediction[..., 0])  
        dy = torch.sigmoid(prediction[..., 1])
        # anchors的宽高调整参数(缩放因子)(offsets)
        dw = torch.sigmoid(prediction[..., 2]) 
        dh = torch.sigmoid(prediction[..., 3]) 
        # obj置信度, 是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # cls置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])
        # 将offset作用到anchor上进行解码得到最终预测结果
        pred_boxes = YOLOv5Reg2Box(batch_size, i, dx, dy, dh, dw, anchors_mask, scaled_anchors, input_height, input_width)

        # 将输出结果转换成原图尺寸下的绝对坐标(xyxy)
        _scale = torch.Tensor([stride_w, stride_h, stride_w, stride_h]).type_as(dx)
        # output.shape = [bs, num_anchors, 5+num_cls]
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale, conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, num_classes)), -1)
        outputs.append(output.data)
    return outputs





def YOLOv5Reg2Box(bs, l, dx, dy, dh, dw, anchors_mask, scaled_anchors, in_h, in_w):
    '''YOLOv5将offset作用到anchor上进行解码得到最终预测结果
    '''
    # 将预测结果进行解码，判断预测结果和真实值的重合程度
    
    # 生成网格，先验框中心，网格左上角(特征图尺寸下的绝对坐标) 
    # grid_x.shape, grid_y.shape = [bs, 3, w, h]
    grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(int(bs * len(anchors_mask[l])), 1, 1).view(dx.shape).type_as(dx)
    grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(int(bs * len(anchors_mask[l])), 1, 1).view(dy.shape).type_as(dx)

    # 按照网格格式生成先验框的宽高(特征图尺寸下的绝对坐标)
    # anchor_w.shape, anchor_h.shape = [bs, 3, w, h]
    scaled_anchors_l = np.array(scaled_anchors)[anchors_mask[l]] # 只根据anchors_mask取出对应层的anchor
    anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(dx)
    anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(dx)
    anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(dw.shape)
    anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(dh.shape)

    '''很重要!! 这部分是YOLOv5将对offset作用到anchor上进行解码得到最终预测结果的核心代码'''
    '''YOLOv5是对网格的左上角点和宽高做微调'''
    # NOTE:惨痛教训:这里不要写dx.data, 否则没有梯度, 导致训练时box_loss不收敛
    dx = dx * 2. - 0.5 # 将dx范围调整成(-0.5, 1.5), 意味着anchor的中心点x可以调整的范围为相邻一个网格的单位
    dy = dy * 2. - 0.5 # 将dy范围调整成(-0.5, 1.5), 意味着anchor的中心点y可以调整的范围为相邻一个网格的单位
    dw = (dw * 2) ** 2 # 将dw范围调整成(0, 4), 意味着anchor的宽可以调整的范围为其本身的4倍以内
    dh = (dh * 2) ** 2 # 将dh范围调整成(0, 4), 意味着anchor的高可以调整的范围为其本身的4倍以内
    pred_boxes = torch.zeros((bs, 3, in_h, in_w, 4), device=dx.device)
    pred_boxes[..., 0] = grid_x + dx
    pred_boxes[..., 1] = grid_y + dy
    pred_boxes[..., 2] = anchor_w * dw
    pred_boxes[..., 3] = anchor_h * dh
    # pred_boxes.shape = [bs, 3, w, h, 4]
    return pred_boxes
    








def cxcywh2xyxy(box_xy, box_wh, input_shape, image_shape):
    # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([input_shape, input_shape], axis=-1)
    return boxes





def non_max_suppression(prediction, input_shape, conf_thres=0.5, nms_thres=0.4, agnostic=False):
    '''nms
    '''    
    #   将预测结果的cxcywh格式转换成xyxy的格式。
    #   prediction = [bs, num_anchors, num_cls+5]
    box_corner         = prediction.new(prediction.shape)
    box_corner[:,:, 0] = prediction[:,:, 0] - prediction[:,:, 2] / 2
    box_corner[:,:, 1] = prediction[:,:, 1] - prediction[:,:, 3] / 2
    box_corner[:,:, 2] = prediction[:,:, 0] + prediction[:,:, 2] / 2
    box_corner[:,:, 3] = prediction[:,:, 1] + prediction[:,:, 3] / 2
    prediction[:,:,:4] = box_corner[:,:,:4]
    # 因为预测时bs=1，所以len(output) = 1
    output = []
    # batch里每张图像逐图像进行nms
    for i, image_pred in enumerate(prediction):
        # 取出每个anchor预测置信度最大的那个类别的置信度以及类别索引
        # class_conf  [num_anchors, 1]    种类置信度
        # class_pred  [num_anchors, 1]    种类
        cls_score, cls_id = torch.max(image_pred[:, 5:], 1, keepdim=True)
        '''首先筛选掉置信度小于阈值的预测 '''
        # class_conf  [num_anchors]
        conf_keep = (image_pred[:, 4] * cls_score[:, 0] >= conf_thres).squeeze()
        image_pred = image_pred[conf_keep]
        cls_score = cls_score[conf_keep]
        cls_id = cls_id[conf_keep]
        # 如果筛选之后没有目标保留，则跳过继续
        if not image_pred.size(0): continue
        # 将坐标和预测的置信度，类别拼在一起 detections  [num_anchors, 7]
        detections = torch.cat((image_pred[:, :5], cls_score.float(), cls_id.float()), 1)
        if agnostic:
            '''类别无关nms(eval时使用这个一般会掉点)'''
            result = NMSbyAll(detections, nms_thres).cpu().numpy()
        else:
            '''逐类别nms'''
            result = NMSbyCLS(detections, nms_thres).cpu().numpy()
        output.append(result)
    return output




def NMSbyCLS(predicts, nms_thres):
    '''逐类别nms'''
    cls_output = torch.tensor([])
    unique_cats = predicts[:, -1].unique()
    for cat in unique_cats:
        # 获得某一类下的所有预测结果
        detections_class = predicts[predicts[:, -1] == cat]
        # 使用官方自带的非极大抑制会速度更快一些
        final_cls_score = detections_class[:, 4] * detections_class[:, 5]
        '''接着筛选掉nms大于nms_thres的预测''' 
        keep = nms(detections_class[:, :4], final_cls_score, nms_thres)
        nms_detections = detections_class[keep]
        # 将类别nms结果记录cls_output
        cls_output = nms_detections if len(cls_output)==0 else torch.cat((cls_output, nms_detections))
    
    return cls_output





def NMSbyAll(predicts, nms_thres):
    '''类别无关的nms'''
    # 使用官方自带的非极大抑制会速度更快一些
    final_cls_score = predicts[:, 4] * predicts[:, 5]
    '''接着筛选掉nms大于nms_thres的预测''' 
    keep = nms(predicts[:, :4], final_cls_score, nms_thres)
    nms_detections = predicts[keep]
    
    return nms_detections




# 平滑标签
def smooth_labels(y_true, label_smoothing, num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes




@MODELS.register
class BCELoss(nn.Module):
    '''二分类交叉熵损失 sigmoid + bceloss
    '''
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction=reduction
        self.loss = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, pred, target):
        """
        """
        loss = self.loss(pred, target)
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()




@MODELS.register
class IoULoss(nn.Module):
    '''L2损失
    '''
    def __init__(self, iou_type, xywh=False, reduction='mean'):
        super(IoULoss, self).__init__()
        self.reduction = reduction
        self.iou_type = iou_type
        self.xywh = xywh
        self.eps = 1e-7


    def forward(self, pred, target):
        """
        """
        iou = self.bbox_iou_pairwise(pred, target)
        loss = 1. - iou
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss
        if self.reduction=='sum':
            return loss.sum()
        
    def bbox_iou_pairwise(self, box1, box2):
        """计算 box1 和 box2 的 IoU (对应位置一对一计算)
            Args:
                box1: [total_anchor_num, 4(x, y, w, h / x0, y0, x1, y1)]
                box2: [total_anchor_num, 4(x, y, w, h / x0, y0, x1, y1)]
            Returns:
                iou:  [total_anchor_num]
        """
        if self.xywh:  # (x, y, w, h) → (x1, y1, x2, y2)
            x1, y1, w1, h1 = box1.unbind(-1)
            x2, y2, w2, h2 = box2.unbind(-1)
            b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
            b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
            b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
            b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
        else:  # (x1, y1, x2, y2)
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

        # 相交区域
        inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
        inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        inter = inter_w * inter_h
        # 各自面积
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        # 并集面积
        union = area1 + area2 - inter + self.eps
        # IoU
        iou = inter / union

        # 处理 GIoU / DIoU / CIoU
        if self.iou_type in ["giou", "diou", "ciou"]:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # 包围盒宽度
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # 包围盒高度

            if self.iou_type in ["diou", "ciou"]:
                c2 = cw ** 2 + ch ** 2 + self.eps
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2) / 4
                if self.iou_type == "ciou":
                    v = (4 / math.pi**2) * (torch.atan((b2_x2 - b2_x1) / (b2_y2 - b2_y1 + self.eps)) -
                                            torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1 + self.eps)))**2
                    with torch.no_grad():
                        alpha = v / (v - iou + 1 + self.eps)
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                return iou - rho2 / c2  # DIoU
            # GIoU
            c_area = cw * ch + self.eps
            return iou - (c_area - union) / c_area
        # IoU
        return iou