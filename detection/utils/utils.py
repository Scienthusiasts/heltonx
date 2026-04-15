import numpy as np
import torch
from torchvision.ops import nms
from torch.nn import functional as F
import cv2
import torch.nn as nn
import os
import math
import json
from tqdm import tqdm
import matplotlib.pyplot as plt








def resize_tensor_to_multiple(img: torch.Tensor, n: int) -> torch.Tensor:
    """
    将输入Tensor的 H 和 W 调整为最接近的、能被 n 整除的尺寸。
        Args:
            img (Tensor): 输入图像, 形状为 [H, W, C]
            n (int): 目标倍数 (例如 16, 32 等)
        Returns:
            Tensor: 调整后的图像
    """
    H, W, C = img.shape
    # 计算最接近且可被 n 整除的尺寸
    new_H = int(round(H / n) * n)
    new_W = int(round(W / n) * n)
    new_H = max(n, new_H)
    new_W = max(n, new_W)
    # 调整形状为 [1, C, H, W]
    img = img.permute(2, 0, 1).unsqueeze(0)
    # 进行双线性插值
    resized_img = F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=False)
    # 还原形状为 [H, W, C]
    resized_img = resized_img.squeeze(0).permute(1, 2, 0)
    return resized_img 



def map_boxes_to_origin_size(boxes, orig_size, target_size):
    """
    将基于 [S, S] (resize+padding 后) 图像预测的 boxes 
    映射回原始图像 [H, W] 尺寸的坐标系 (NumPy 版本)。
    
    Args:
        boxes (np.ndarray): [n, 4] 预测框坐标，基于 [S, S]。
        orig_size (tuple): 原图尺寸 [H, W]。
        target_size (int): 网络输入尺寸 S (正方形)。
        
    Returns:
        boxes_orig (np.ndarray): [n, 4] 映射回原图坐标的 boxes。
    """
    H, W = orig_size
    S = target_size

    # 计算缩放比例
    r = S / max(H, W)
    
    # 计算 padding（相对 [S, S]）
    if H > W:
        # 高更长 ⇒ 左右pad
        new_w = int(W * r)
        new_h = int(H * r)
        pad_w = (S - new_w) / 2
        pad_h = 0
    else:
        # 宽更长 ⇒ 上下pad
        new_w = int(W * r)
        new_h = int(H * r)
        pad_w = 0
        pad_h = (S - new_h) / 2

    # 去除 padding 偏移
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h

    # 防止负数
    boxes = np.clip(boxes, 0, None)

    # 映射回原始比例
    boxes /= r

    # 限制在原图尺寸范围内
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)

    return boxes


def OpenCVDrawBox(image, boxes, classes, scores, save_vis_path, image2color, class_names, resize_size, show_text=True):
    '''plt画框
        Args:
            :param image:         原始图像(Image格式)
            :param boxes:         网络预测的box坐标
            :param classes:       网络预测的box类别
            :param scores:        网络预测的box置信度
            :param save_res_path: 可视化结果保存路径

        Returns:
            None
    '''
    H, W = image.shape[:2]
    max_len = max(W, H)
    w = int(W * resize_size[0] / max_len)
    h = int(H * resize_size[1] / max_len)
    boxes[:, [0,2]] *= w / W
    boxes[:, [1,3]] *= h / H

    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    # 框的粗细
    thickness = max(1, int(image.shape[0] * 0.003))
    for box, cls, score in zip(boxes, classes, scores):
        # if class_names[cls] not in  ['pedestrian', 'people', 'person']:continue
        x0, y0, x1, y1 = round(box[0]), round(box[1]), round(box[2]), round(box[3])
        color = np.array(image2color[class_names[cls]])
        color = tuple([int(c*255) for c in color])
        # color = (0,0,255)
        # text = 'target_1'
        # text = '{} {:.2f}'.format(text, score)
        text = '{} {:.2f}'.format(class_names[cls], score)
        # obj的框
        cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=thickness)
        # 文本的框
        if show_text:
            cv2.rectangle(image, (x0-1, y0-30), (x0+len(text)*12, y0), color, thickness=-1)
            # 文本
            cv2.putText(image, text, (x0, y0-6), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255,255,255), thickness=2)
    # 保存
    if save_vis_path!=None:
        return image
    image = cv2.resize(image, (W, H))
    return image