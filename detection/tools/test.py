# coding=utf-8
import os
import argparse
import torch
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
from heltonx.utils.utils import to_device, dynamic_import_class
from heltonx.utils.register import MODELS

# 需要import才能注册
from detection import *


def resize_to_multiple_no_keep_ratio(img, n):
    h, w = img.shape[:2]
    new_w = math.ceil(w / n) * n
    new_h = math.ceil(h / n) * n
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)




def infer_single_img(model, device, img_path, cat_names, save_vis_path, img_size=[800, 800], show_text=True, vis_heatmap=False):
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

    tensor_img = tensor_img.permute(2,0,1).unsqueeze(0).to(device)

    '''每个类别都获得一个随机颜色'''
    image2color = dict()
    for cat in cat_names:
        image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]

    '''推理一张图像'''
    boxes, box_scores, box_classes = model.infer(tensor_img, vis_heatmap=vis_heatmap, save_vis_path='./det_res.jpg')
    #  检测出物体才继续    
    if len(boxes) == 0: 
        print(f'no objects in image: {img_path}.')
        return boxes, box_scores, box_classes

    '''画框（在原图上绘制，需要将 boxes 从缩放后尺寸映射回原图）'''
    vis_img = OpenCVDrawBox(image, boxes, box_classes, box_scores, save_vis_path,
                            image2color, cat_names, show_text=show_text,
                            resized_size=(resized_w, resized_h))  # 注意顺序 (width, height)
    # 统计检测出的类别和数量
    detect_cls = dict(Counter(box_classes))
    detect_name = {}
    for key, val in detect_cls.items():
        detect_name[cat_names[key]] = val
    print(f'detect result: {detect_name}')
    return boxes, box_scores, box_classes




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection Inference')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--ckpt', type=str, default=None, help='权重文件路径（覆盖config中的load_ckpt）')
    parser.add_argument('--thr', type=str, default=0.05, help='推理时的得分阈值')
    parser.add_argument('--img', type=str, default='detection/demos/13.jpg', help='待推理图像路径')
    parser.add_argument('--save', type=str, default='./det_res.jpg', help='可视化结果保存路径')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 从config文件加载参数
    cargs = dynamic_import_class(args.config, get_class=False)

    # 从config中读取类别信息和图像尺寸
    cat_names = cargs.cat_names
    img_size = cargs.img_size

    # 模型配置：使用config中的model_cfgs，并覆盖load_ckpt
    model_cfgs = cargs.model_cfgs.copy()
    if 'nms_score_thr' in model_cfgs:
        model_cfgs['nms_score_thr'] = args.thr
    else:
        model_cfgs['score_thr'] = args.thr
    if args.ckpt is not None:
        model_cfgs['load_ckpt'] = args.ckpt

    nc = len(cat_names)
    print(f"模型类型: {model_cfgs['type']}, 类别数: {nc}, 图像尺寸: {img_size}")
    print(f"权重路径: {model_cfgs.get('load_ckpt', 'None')}")

    model = MODELS.build_from_cfg(model_cfgs).to(device)
    model.eval()
    infer_single_img(model, device, args.img, cat_names, args.save, img_size=img_size, show_text=True)
    # /mnt/yht/env/yht_pretrain/bin/python -m detection.tools.test --config detection/configs/detr_coco_ddp.py --ckpt log/detection/detr_coco_train_ddp/2026-06-05-10-17-14_train_ddp/last.pt
    # /mnt/yht/env/yht_pretrain/bin/python -m detection.tools.test --config detection/configs/yolo26_coco_ddp.py --ckpt log/yolo26x_coco_train_ddp/2026-06-15-08-37-56_train_ddp/best_val_map.pt
