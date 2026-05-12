import torch
import torch.nn as nn
import torch.distributed as dist
from utils.utils import init_weights
# 注册机制
from heltonx.utils.register import MODELS
from detection.utils.fcos_utils import *
from detection.losses import *
from detection.utils.yolov5_utils import smooth_labels





@MODELS.register
class YOLOv5Head(nn.Module):
    """YOLOv5 检测头模块 (耦合式检测头)

    Args:
        phi (str): 模型规模标识，可选 'n', 's', 'm', 'l', 'x'
        nc (int): 类别数量
        img_size (List[int]): 输入图像尺寸，如 [640, 640]
        anchors (List[List[int]]): 先验框列表
        anchors_mask (List[List[int]]): 先验框掩码
        cls_loss (nn.Module): 分类损失函数
        box_loss (nn.Module): 边界框损失函数
        obj_loss (nn.Module): 目标置信度损失函数
        assigner (nn.Module): 正负样本分配器
        bbox_coder (nn.Module): bbox 编解码器
        layers_num (int): 特征层数量，默认 3
        label_smoothing (float): 标签平滑参数，默认 0
    """
    def __init__(self, phi, nc, img_size, anchors, anchors_mask, cls_loss, box_loss, obj_loss, assigner, bbox_coder, layers_num=3, label_smoothing=0):
        super().__init__()
        '''基本配置'''
        depth_dict          = {'n': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'n': 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
        base_channels       = int(wid_mul * 64)  
        base_depth          = max(round(dep_mul * 3), 1)  

        # 自适应调整不同损失的权重（与官方一致）
        # 官方逻辑: box *= 3/nl, cls *= (nc/80)*(3/nl), obj *= (imgsz/640)^2*(3/nl)
        nl = layers_num
        self.box_ratio = 0.05 * (3 / nl)
        self.obj_ratio = 1.0 * (img_size[0] * img_size[1]) / (640 ** 2) * (3 / nl)
        self.cls_ratio = 0.5 * (nc / 80) * (3 / nl)
        print(f'box_loss_ratio:{self.box_ratio} | obj_loss_ratio:{self.obj_ratio} | cls_loss_ratio:{self.cls_ratio}')

        # 使用几层特征图
        self.layers_num = layers_num
        s = [4, 8, 16]
        # 包含每个尺度的head(一个尺度一个head, 不共享)
        self.p_heads = nn.ModuleList([CoupledConvHead(i, nc, img_size, anchors, base_channels*s[i], anchors_mask, cls_loss, box_loss, obj_loss, bbox_coder, label_smoothing) for i in range(layers_num)])
        '''正负样本分配'''
        self.assigner = assigner
        self.anchors = anchors
        self.img_size = img_size
        self.anchors_mask = anchors_mask
        self.nc = nc

    def forward(self, x):
        """前向传播

        Args:
            x (List[Tensor]): 多尺度特征图列表

        Returns:
            preds (List[Tensor]): 各层预测结果
        """
        preds = []
        for i in range(self.layers_num):
            pred = self.p_heads[i](x[i])
            preds.append(pred)

        return preds

    def loss(self, x, batch_bboxes, batch_labels):
        """计算损失

        Args:
            x (List[Tensor]): 多尺度特征图列表
            batch_bboxes (List[Tensor]): batch 内各图像的 GT 框
            batch_labels (List[Tensor]): batch 内各图像的 GT 类别

        Returns:
            losses (Dict[str, Tensor]): 各损失组成的字典
        """
        '''正负样本分配'''
        y_trues = [[] for _ in range(self.layers_num)]
        for bboxes, labels in zip(batch_bboxes, batch_labels):
            # coco格式转成YOLO格式(xywh -> norm(cxcywh)):
            print(bboxes.shape)
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
    """YOLOv5 单层耦合卷积检测头

    Args:
        l (int): 当前层索引 (0: P3, 1: P4, 2: P5)
        cat_nums (int): 类别数量
        img_size (List[int]): 输入图像尺寸
        anchors (List[List[int]]): 先验框列表
        in_channels (int): 输入通道数
        anchors_mask (List[int]): 先验框掩码
        cls_loss (nn.Module): 分类损失函数
        box_loss (nn.Module): 边界框损失函数
        obj_loss (nn.Module): 目标置信度损失函数
        bbox_coder (nn.Module): bbox 编解码器
        label_smoothing (float): 标签平滑参数，默认 0
    """
    def __init__(self, l, cat_nums, img_size, anchors, in_channels, anchors_mask, cls_loss, box_loss, obj_loss, bbox_coder, label_smoothing=0):
        super().__init__()
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
        self.bbox_coder = bbox_coder

        '''网络部分, YOLOv5 head是耦合的'''
        self.head = nn.Conv2d(in_channels, len(anchors_mask) * (5 + self.num_classes), 1)
        '''初始化权重'''
        init_weights(self.head, 'normal', 0, 0.01)




    def forward(self, x):
        """单层前向传播

        Args:
            x (Tensor): 输入特征图

        Returns:
            predict (Tensor): 预测结果
        """
        predict = self.head(x)
        return predict



    def loss(self, fpn_single_feat, y_true):
        """计算单层损失

        Args:
            fpn_single_feat (Tensor): 单层 FPN 特征图
            y_true (Tensor): 真实标签

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (box_loss, cls_loss, obj_loss)
        """
        # 前向，获得网络预测结果
        input = self.forward(fpn_single_feat)
        #  获得bs，特征层的高和宽
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        # 使用固定下采样率, 不依赖img_size, 避免非正方形图像时stride计算错误
        strides = [8, 16, 32]
        stride = strides[self.l]
        # 此时获得的scaled_anchors大小是相对于特征层的
        scaled_anchors  = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        # torch.Size([bs, 255, w, h]) -> torch.Size([bs, 3, w, h, 85]) (85 : cx, cy, w, h, obj_score, cls_score=80)
        prediction = input.view(bs, len(self.anchors_mask[self.l]), self.num_classes+5, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # cx, cy, w, h, obj_score, cls_score通过simoid限制到(0,1)之间
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.sigmoid(prediction[..., 2]) 
        h = torch.sigmoid(prediction[..., 3]) 
        conf = prediction[..., 4]
        pred_cls = prediction[..., 5:]

        # 将预测结果进行解码, 即将预测offset作用到anchors上(基于特征图尺寸的cxcywh)
        # pred_boxes.shape = torch.Size([bs, 3, w, h, 4])
        pred_boxes = self.bbox_coder.reg2box(bs, self.l, x, y, h, w, scaled_anchors, in_h, in_w)
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
        # print(self.l, self.balance[self.l], tobj.shape)

        return box_loss, cls_loss, obj_loss




















# for test only
if __name__ == '__main__':
    # 基本配置
    phi = 's'
    depth_dict          = {'n': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict          = {'n': 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
    base_channels       = int(wid_mul * 64)  # 64
    base_depth          = max(round(dep_mul * 3), 1)  # 3
    head_in_channel = {
        'n':[64,  128, 256 ],
        's':[128, 256, 512 ],
        'm':[192, 384, 768 ],
        'l':[256, 512, 1024],
        'x':[324, 640, 1280]
    }[phi]
    num_classes = 80 
    anchors_mask = [[0,1,2], [3,4,5], [6,7,8]]
    # head
    p3_head = YOLOv5Head(num_classes, base_channels * 4 , anchors_mask[0])
    p4_head = YOLOv5Head(num_classes, base_channels * 8 , anchors_mask[1])
    p5_head = YOLOv5Head(num_classes, base_channels * 16, anchors_mask[2])
    # 验证
    p3 = torch.rand((4, head_in_channel[0], 80, 80))
    p4 = torch.rand((4, head_in_channel[1], 40, 40))
    p5 = torch.rand((4, head_in_channel[2], 20, 20))
    p3_predict = p3_head(p3)
    p4_predict = p4_head(p4)
    p5_predict = p5_head(p5)
    # 
    print(p3_predict.shape) # torch.Size([4, 255, 80, 80])
    print(p4_predict.shape) # torch.Size([4, 255, 40, 40])
    print(p5_predict.shape) # torch.Size([4, 255, 20, 20])