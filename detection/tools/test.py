# coding=utf-8
import os
import json
import torch
from torch import nn
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np
import math
from collections import Counter
from detection.utils.metrics import *
from detection.utils.utils import OpenCVDrawBox
from detection.datasets.preprocess import Transforms
from detection.utils.utils import resize_tensor_to_multiple
from heltonx.utils.register import EVALPIPELINES
from heltonx.utils.utils import to_device
from heltonx.utils.register import MODELS





def resize_to_multiple_no_keep_ratio(img, n):
    h, w = img.shape[:2]
    new_w = math.ceil(w / n) * n
    new_h = math.ceil(h / n) * n
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)




def infer_single_img(model, device, img_path, cat_names, save_vis_path, img_size=[800, 800]):
    """推理一张图

    Args:
        model: 检测模型
        device: 计算设备 (cpu/cuda)
        img_path: 图像路径
        cat_names: 类别名称列表
        save_vis_path: 可视化图像保存路径
        img_size: 固定图像大小，如 [832, 832]，默认 [800, 800]

    Returns:
        boxes:       网络回归的box坐标    [obj_nums, 4]
        box_scores:  网络预测的box置信度  [obj_nums]
        box_classes: 网络预测的box类别    [obj_nums]
    """
    # 图像均值 标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([[0.229, 0.224, 0.225]])
    transform = Transforms(img_size=img_size)

    # 读取原始图像（RGB格式）
    image = np.array(Image.open(img_path).convert('RGB'))
    H, W = image.shape[:2]

    # 模型预处理（缩放、归一化等）
    tensor_img = torch.tensor(transform.test_transform(image=image)['image'])
    # 可能调整到32的倍数
    tensor_img = resize_tensor_to_multiple(tensor_img, 32)

    # 获取实际缩放后的尺寸（H, W）
    resized_h, resized_w = tensor_img.shape[0], tensor_img.shape[1]

    # 归一化反变换（仅用于调试，此处不用于绘制）
    # resize_img = ((tensor_img.numpy() * std + mean) * 255).astype(np.uint8)

    tensor_img = tensor_img.permute(2,0,1).unsqueeze(0).to(device)

    '''每个类别都获得一个随机颜色'''
    image2color = dict()
    for cat in cat_names:
        image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]

    '''推理一张图像'''
    boxes, box_scores, box_classes = model.infer(tensor_img, vis_heatmap=True, save_vis_path='./det_res.jpg')
    #  检测出物体才继续    
    if len(boxes) == 0: 
        print(f'no objects in image: {img_path}.')
        return boxes, box_scores, box_classes

    '''画框（在原图上绘制，需要将 boxes 从缩放后尺寸映射回原图）'''
    vis_img = OpenCVDrawBox(image, boxes, box_classes, box_scores, save_vis_path,
                            image2color, cat_names, show_text=True,
                            resized_size=(resized_w, resized_h))  # 注意顺序 (width, height)
    # 保存（OpenCVDrawBox 已保存，此处无需重复保存，但保留以兼容旧逻辑）
    # cv2.imwrite(save_vis_path, vis_img)  # 若 OpenCVDrawBox 未保存可取消注释
    # 统计检测出的类别和数量
    detect_cls = dict(Counter(box_classes))
    detect_name = {}
    for key, val in detect_cls.items():
        detect_name[cat_names[key]] = val
    print(f'detect result: {detect_name}')
    return boxes, box_scores, box_classes






if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cat_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
                # "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    cat_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    nc = len(cat_names)

    img_size = [800, 800]

    '''模型配置参数'''
    # load_ckpt = 'log/yolov5_coco_train_ddp/2026-04-25-22-32-45_train_ddp/best_val_map.pt'
    # phi = 's'
    # anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]] 
    # anchors_mask=[[0,1,2], [3,4,5], [6,7,8]]
    # model_cfgs = dict(
    #     type="YOLOv5",
    #     nc=nc, 
    #     img_size=img_size, 
    #     anchors=anchors,
    #     anchors_mask=anchors_mask,
    #     load_ckpt=load_ckpt,
    #     nms_score_thr=0.25,
    #     nms_iou_thr=0.3, 
    #     nms_agnostic=False,
    #     bbox_coder=dict(
    #         type="YOLOv5BBoxCoder",
    #         img_size=img_size,
    #         anchors=anchors,
    #         anchors_mask=anchors_mask,
    #         nc=nc,
    #     ),
    #     backbone=dict(
    #         type="YOLOv5CSPDarknet",
    #         phi=phi,
    #         out_layers=[2,3,4],
    #         froze_backbone=False,
    #         load_ckpt=f'ckpts/yolo/cspdarknet_{phi}_v6.1_backbone.pth'
    #     ), 
    #     fpn=dict(
    #         type="YOLOv5PAFPN",
    #         phi=phi,
    #     ), 
    #     heads=dict(
    #         type="YOLOv5Head",
    #         phi=phi,
    #         nc=nc, 
    #         img_size=img_size, 
    #         anchors=anchors,
    #         anchors_mask=anchors_mask,
    #         label_smoothing=0,
    #         layers_num=3,
    #         bbox_coder=dict(
    #             type="YOLOv5BBoxCoder",
    #             img_size=img_size,
    #             anchors=anchors,
    #             anchors_mask=anchors_mask,
    #             nc=nc,
    #         ),
    #         cls_loss=dict(
    #             # type="FocalLoss",
    #             # reduction="mean",
    #             # gamma=2.0, 
    #             # alpha=0.25
    #             type="BCELoss",
    #             reduction="mean"
    #         ),
    #         box_loss=dict(
    #             type="IoULoss",
    #             iou_type='ciou',
    #             xywh=True,
    #             reduction="none",
    #         ),
    #         obj_loss=dict(
    #             type="BCELoss",
    #             reduction="mean"
    #         ), 
    #         assigner=dict(
    #             type="YOLOv5Assigner",
    #             img_size=img_size, 
    #             anchors=anchors,
    #             anchors_mask=anchors_mask,
    #             threshold=4,
    #         )
    #     )
    # )



    '''FCOS'''
    # load_ckpt = "log/fcos_pafpn_dinov3sta_coco_train_ddp/2025-10-24-11-55-27_train_ddp/last.pt"
    # model_cfgs = dict(
    #     type="FCOS",
    #     img_size=img_size,
    #     nc=nc, 
    #     load_ckpt=load_ckpt,
    #     nms_score_thr=0.2,
    #     nms_iou_thr=0.3, 
    #     nms_agnostic=False,
    #     bbox_coder=dict(
    #         type="FCOSBBoxCoder",
    #         strides=[8, 16, 32, 64, 128]
    #     ),
    #     backbone=dict(
    #         type="DINOv3STA",
    #         dino_name="vit_small_patch16_dinov3.lvd1689m",
    #         dino_dim=384,
    #         dino_out_indices=[5, 8, 11],
    #         sta_layer_dims=[64, 128, 128, 256, 512],
    #         fuse_layer_dims=[128, 256, 512],
    #         out_layers=[2, 3, 4],
    #         dino_ckpt="ckpts/vit_small_patch16_dinov3.lvd1689m.pt",
    #         froze_dino=True
    #     ), 
    #     fpn = dict(
    #         type="PAFPN",
    #         in_channels=[128, 256, 512],
    #         out_channel=256,
    #         num_extra_levels=2
    #     ),
    #     head=dict(
    #         type="FCOSHead",
    #         nc=nc, 
    #         in_channel=256, 
    #         cnt_loss=dict(
    #             type="BCELoss",
    #             reduction="mean"
    #         ), 
    #         cls_loss=dict(
    #             type="FocalLoss",
    #             reduction="none",
    #             gamma=2.0, 
    #             alpha=0.25
    #         ),
    #         reg_loss=dict(
    #             type="IoULoss",
    #             iou_type='giou',
    #             xywh=False,
    #             reduction="mean",
    #         ),
    #         assigner=dict(
    #             type="FCOSAssigner",
    #             img_size=img_size, 
    #             strides=[8, 16, 32, 64, 128], 
    #             limit_ranges=[[-1,64],[64,128],[128,256],[256,512],[512,999999]], 
    #             sample_radiu_ratio=1.5
    #         )
    #     )
    # )




    '''detr'''
    # load_ckpt = 'log/detr_coco_train_ddp/2026-04-26-16-06-06_train_ddp/best_val_map.pt'
    # model_cfgs = dict(
    #     type="DETR",
    #     nc=nc,
    #     img_size=img_size,
    #     load_ckpt=load_ckpt,
    #     score_thr=0.05,  # DETR 不做 NMS, 仅过滤低置信度
    #     backbone=dict(
    #         type="TIMMBackbone",
    #         model_name="resnet50.a1_in1k",
    #         pretrained=False,
    #         out_layers=[4],  # 只取最后一层 C5
    #         froze_backbone=False,
    #         load_ckpt='ckpts/backbone_resnet50.a1_in1k.pt'
    #     ),
    #     fpn=dict(
    #         type="DETRTransformer",
    #         in_channels=2048,
    #         hidden_dim=256,
    #         num_heads=8,
    #         num_encoder_layers=6,
    #         num_decoder_layers=6,
    #         num_queries=100,
    #         dim_feedforward=2048,
    #         dropout=0.1,
    #     ),
    #     head=dict(
    #         type="DETRHead",
    #         nc=nc,
    #         hidden_dim=256,
    #         num_queries=100,
    #         num_decoder_layers=6,
    #         cls_loss=dict(
    #             type="DETRFocalLoss",
    #             alpha=0.25,
    #             gamma=2.0,
    #         ),
    #         l1_loss=dict(
    #             type="DETRL1Loss",
    #         ),
    #         giou_loss=dict(
    #             type="DETRGiouLoss",
    #         ),
    #         assigner=dict(
    #             type="HungarianAssigner",
    #             cls_cost_weight=1.0,
    #             l1_cost_weight=5.0,
    #             giou_cost_weight=2.0,
    #         ),
    #     ),
    #     bbox_coder=dict(
    #         type="DETRBBoxCoder",
    #         img_size=img_size,
    #     ),
    # )


    '''模型配置参数'''
    load_ckpt = 'log/faster_rcnn_pafpn_dinov3sta_coco_train_ddp/2026-05-11-16-58-28_train_ddp/last.pt'
    model_cfgs = dict(
        type="FasterRCNN",
        nc=nc,
        img_size=img_size,
        load_ckpt=load_ckpt,
        backbone=dict(
            type="DINOv3STA",
            dino_name="vit_small_patch16_dinov3.lvd1689m",
            dino_dim=384,
            dino_out_indices=[2, 5, 8, 11],
            sta_layer_dims=[64, 128, 256, 512, 1024],
            fuse_layer_dims=[128, 256, 512, 1024],
            out_layers=[1, 2, 3, 4],
            dino_ckpt="ckpts/vit_small_patch16_dinov3.lvd1689m.pt",
            froze_dino=True
        ), 
        fpn = dict(
            type="PAFPN",
            in_channels=[128, 256, 512, 1024],
            out_channel=256,
            num_extra_levels=1
        ),
        rpn_head=dict(
            type="RPNHead",
            in_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            anchor_generator=dict(
                type="AnchorGenerator",
                strides=[4, 8, 16, 32, 64],
                ratios=[0.5, 1.0, 2.0],
                scales=[8],
            ),
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=(0., 0., 0., 0.),
                target_stds=(1., 1., 1., 1.),
            ),
            cls_loss=dict(
                type="BCELoss",
                reduction="mean",
            ),
            reg_loss=dict(
                type="SmoothL1Loss",
                beta=1.0,
                reduction="mean",
            ),
            num_samples=256,
            pos_fraction=0.5,
            nms_pre=2000,
            max_per_img=1000,
            nms_thr=0.7,
        ),
        roi_head=dict(
            type="StandardRoIHead",
            bbox_roi_extractor=dict(
                type="RoIAlign",
                output_size=(7, 7),
                use_multilevel=True,
                featmap_strides=[4, 8, 16, 32, 64],
                aligned=True,
            ),
            bbox_head=dict(
                type="BBoxFCHead",
                in_channels=256,
                roi_feat_size=7,
                num_classes=nc,
                fc_out_channels=1024,
                num_fcs=2,
                with_cls=True,
                with_reg=True,
                reg_class_agnostic=False,
            ),
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=(0., 0., 0., 0.),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            cls_loss=dict(
                type="CrossEntropyLoss",
                reduction="mean",
            ),
            reg_loss=dict(
                type="SmoothL1Loss",
                beta=1.0,
                reduction="mean",
            ),
            num_samples=512,
            pos_fraction=0.25,
            score_thr=0.3,
            max_per_img=100,
            nms_thr=0.5,
        )
    )





    model = MODELS.build_from_cfg(model_cfgs).to(device)
    model.eval()
    img_path = 'detection/demos/12.jpg'
    save_vis_path = './det_res.jpg'
    infer_single_img(model, device, img_path, cat_names, save_vis_path)