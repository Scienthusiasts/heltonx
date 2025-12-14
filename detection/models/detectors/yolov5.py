import torch.nn as nn
from PIL import Image
from collections import Counter
import time

# 注册机制
from heltonx.utils.register import MODELS
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
from detection.utils.fcos_utils import *
from detection.losses import *
from detection.utils.nms import NMS
from detection.utils.yolov5_utils import *



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
            p = self.fpn(backbone_feat)
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
        H, W = image.shape[2:]
        with torch.no_grad():
            '''网络推理得到最原始的未解码未nms的结果'''
            # p3, p4, p5
            predicts = self.forward(image, return_loss=False)
            '''利用Head的预测结果对RPN proposals进行微调+解码(解码到原图尺寸下的绝对坐标xyxy) 获得预测框'''
            # torch.Size([1, 1200, 85])
            # torch.Size([1, 4800, 85])
            # torch.Size([1, 19200, 85])
            # decode_predicts 是归一化的cxcywh
            decode_predicts = inferDecodeBox(predicts, img_size, self.nc, self.anchors, self.anchors_mask)
            # torch.cat(decode_predicts, 1) : torch.Size([1, 25200, 85])
            decode_predicts = torch.cat(decode_predicts, 1)
            '''计算nms, 同时将归一化的cxcywh->归一化的xywh'''
            # decode_predicts = self.nms(decode_predicts, self.nms_score_thr, self.nms_iou_thr, self.nms_agnostic)[0]
            decode_predicts = non_max_suppression(decode_predicts, img_size, conf_thres=self.nms_score_thr, nms_thres=self.nms_iou_thr, agnostic=agnostic)
            # 图像里没预测出目标的情况:
            if len(decode_predicts) == 0 : return [],[],[]
            box_classes = np.array(decode_predicts[0][:, 6], dtype = 'int32')
            box_scores = decode_predicts[0][:, 4] * decode_predicts[0][:, 5]
            # norm(xywh) -> xywh
            boxes = decode_predicts[0][:, :4]
            boxes[:, [1,3]] *= H
            boxes[:, [0,2]] *= W
            '''是否可视化obj heatmap'''
            if vis_heatmap:
                vis_YOLOv5_heatmap(predicts, [W, H], self.img_size, image, box_classes, save_vis_path=save_vis_path, padding=False)

            return boxes, box_scores, box_classes








# for test only
if __name__ == '__main__':
    phi='x'
    model = YOLOv5(anchors_mask=[[6,7,8], [3,4,5], [0,1,2]], num_classes=80, phi=phi, pretrained=False, loadckpt=None)
    torch.save(model.state_dict(), f"{phi}.pt")
    # 验证 1
    # summary(model, input_size=[(3, 224, 224)])  
    # 验证 2
    x = torch.rand((8, 3, 640, 640))
    p3_predict, p4_predict, p5_predict = model(x)
    print(p3_predict.shape)    
    print(p4_predict.shape)   
    print(p5_predict.shape) 
