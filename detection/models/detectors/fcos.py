import torch
import torch.nn as nn
import timm
import torch.distributed as dist
from utils.utils import init_weights
# 注册机制
from heltonx.utils.register import MODELS
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
from detection.utils.fcos_utils import *
from detection.losses import *
from detection.utils.nms import NMS

@MODELS.register
class FCOS(nn.Module):
    '''完整FCOS网络架构
    '''
    def __init__(self, backbone:nn.Module, fpn:nn.Module, head:nn.Module, img_size, nc, load_ckpt, nms_score_thr, nms_iou_thr, nms_agnostic, bbox_coder:nn.Module):
        """
        """
        super(FCOS, self).__init__()
        self.nms = NMS()
        self.bbox_coder = bbox_coder
        # 基本参数
        self.nc = nc
        self.img_size = img_size
        self.nms_score_thr = nms_score_thr
        self.nms_iou_thr = nms_iou_thr
        self.nms_agnostic = nms_agnostic
        '''网络基本组件'''
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)


    def forward(self, datas, return_loss=True):
        '''一个batch的前向流程 (核心, 要更改训练pipeline主要改这里)
        Args:
            datas[0]: batch_imgs:   一个batch里的图像      [bs, 3, H, W]
            datas[1]: batch_bboxes: 一个batch里的GT框      [[gt_nums_per_img, 4=(x, y, w, h)], ..., [...]]
            datas[2]: batch_labels: 一个batch里的GT框类别  [[gt_nums_per_img], ..., [...]]
            return_loss:  只前向或计算损失
        Returns:
            losses: 所有损失组成的列表 
            cls_logit: 类别logits   [bs, nc, w, h]
            cnt_logit: 中心度logits [bs, 1, w, h]
            reg_pred:  回归值       [bs, 4=(l, t, r, b), w, h]
        '''
        if return_loss:
            batch_imgs, batch_bboxes, batch_labels = datas[0], datas[1], datas[2]
            # 前向过程
            backbone_feat = self.backbone(batch_imgs)
            fpn_feat = self.fpn(backbone_feat)
            # 计算损失(FCOS的正负样本分配在head部分执行)
            loss = self.head.loss(fpn_feat, batch_bboxes, batch_labels)
            return loss
        else:
            batch_imgs = datas
            backbone_feat = self.backbone(batch_imgs)
            fpn_feat = self.fpn(backbone_feat)
            cls_logits, cnt_logits, reg_preds = self.head(fpn_feat)

            return cls_logits, cnt_logits, reg_preds


    def infer(self, image:torch.tensor, vis_heatmap=False, save_vis_path=None):
        '''推理一张图/一帧
            Args:
                image:  读取的图像 [1, 3, H, W]
            # Returns:
                boxes:       网络回归的box坐标    [obj_nums, 4=(x0, y0, x1, y1)]
                box_scores:  网络预测的box置信度  [obj_nums]
                box_classes: 网络预测的box类别    [obj_nums]
        '''
        # tensor_img有padding的黑边
        with torch.no_grad():
            '''网络推理得到推理结果(dense)'''
            cls_logits, cnt_logits, reg_preds = self.forward(image, return_loss=False)
            # [bs, total_anchor_num, 4+1+1=(x0, y0, x1, y1, score, label)]
            predictions = self.bbox_coder.decode(cls_logits, cnt_logits, reg_preds)
            # 只有一个batch, 所以直接取[0]
            results = self.nms(predictions, self.nms_score_thr, self.nms_iou_thr, self.nms_agnostic)[0]
            # 图像里没预测出目标的情况:
            if len(results) == 0 : return [],[],[]
            box_classes = np.array(results[:, 5], dtype = 'int32')
            box_scores = results[:, 4]
            boxes = results[:, :4] # xyxy
            '''box坐标映射(有灰边图像里的坐标->原图的坐标)'''
            # W, H 原始图像的大小
            H, W = image.shape[2:]
            '''是否可视化obj heatmap'''
            if vis_heatmap:
                vis_FCOS_heatmap(cls_logits, cnt_logits, [W, H], self.img_size, image, save_vis_path=save_vis_path)
            
            # 只需要返回3要素:坐标, 得分, 类别
            return boxes, box_scores, box_classes







# for test only
if __name__ == '__main__':
    backbone_name = 'resnet50.a1_in1k'
    img_size = [640, 640]
    num_classes = 15
    loadckpt = False
    tta_img_size = [[640,640], [832,832], [960,960]]
    backbone = dict(
         modelType = backbone_name, 
         loadckpt = False, 
         pretrain = True, 
         froze = True,
    )
    head = dict(
        num_classes = num_classes,
        in_channel = 256,
    )

    model = FCOS(backbone_name, img_size, num_classes, loadckpt, tta_img_size, backbone, head)
    
    x = torch.rand((8, 3, 640, 640))
    cls_logits, cnt_logits, reg_preds = model(x)
    for cls, cnt, reg in zip(cls_logits, cnt_logits, reg_preds):
        print(cls.shape, cnt.shape, reg.shape)

    # torch.Size([8, 15, 80, 80]) torch.Size([8, 1, 80, 80]) torch.Size([8, 4, 80, 80])
    # torch.Size([8, 15, 40, 40]) torch.Size([8, 1, 40, 40]) torch.Size([8, 4, 40, 40])
    # torch.Size([8, 15, 20, 20]) torch.Size([8, 1, 20, 20]) torch.Size([8, 4, 20, 20])
    # torch.Size([8, 15, 10, 10]) torch.Size([8, 1, 10, 10]) torch.Size([8, 4, 10, 10])
    # torch.Size([8, 15, 5, 5]) torch.Size([8, 1, 5, 5]) torch.Size([8, 4, 5, 5])
