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
import matplotlib.patches as patches
import colorsys






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





def OpenCVDrawBox(image, boxes, classes, scores, save_vis_path, image2color, class_names, show_text=True, resized_size=None):
    '''
    基于 Matplotlib 在原图上绘制检测框（自动根据图像尺寸调整线条粗细和文本大小）
    Args:
        image:         原始图像 (numpy.ndarray, HxWx3, 可以是 RGB 或 BGR，函数内部会统一为 RGB)
        boxes:         检测框坐标 [N,4] (x1,y1,x2,y2) 浮点数或整数，坐标基于缩放后图像尺寸
        classes:       类别索引
        scores:        置信度
        save_vis_path: 保存路径，为 None 时仅返回图像
        image2color:   类别到颜色映射 {name: [R,G,B] 0~1}  (RGB)
        class_names:   类别索引 -> 名称列表
        show_text:     是否显示文本
        resized_size:  可选，模型输入缩放后的尺寸 (width, height)。若提供，则将 boxes 从该尺寸映射回原始图像尺寸
    Returns:
        绘制后的图像 (numpy.ndarray, RGB 格式)
    '''
    H, W = image.shape[:2]

    # 如果提供了缩放后的尺寸，则将 boxes 映射到原始图像坐标
    if resized_size is not None:
        resized_w, resized_h = resized_size
        scale_x = W / resized_w
        scale_y = H / resized_h
        boxes = boxes.copy().astype(float)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    # 自适应参数（基于原始图像尺寸）
    img_diag = (H ** 2 + W ** 2) ** 0.5
    linewidth = max(0.5, min(2, img_diag * 0.002))  # 框线粗细（点）
    fontsize = max(6, min(12, H / 50.0))           # 字体大小（点）

    # 创建图形，去除白边
    dpi = 100
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(image)  # 期望 RGB
    ax.axis('off')

    # 辅助函数：根据边框颜色生成更亮的相似文本颜色
    def get_lighter_color(rgb_color, factor=1.5, min_v=0.6):
        h, l, s = colorsys.rgb_to_hls(rgb_color[0], rgb_color[1], rgb_color[2])
        l = min(1.0, l * factor)
        l = max(min_v, l)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return (r, g, b)

    # 绘制所有检测框
    for box, cls, score in zip(boxes, classes, scores):
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        # 确保坐标在图像内
        x0 = max(0, min(x0, W))
        y0 = max(0, min(y0, H))
        x1 = max(0, min(x1, W))
        y1 = max(0, min(y1, H))
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            continue

        color = np.array(image2color[class_names[cls]])  # RGB 0~1
        color = tuple(color)

        # 矩形框（无填充）
        rect = patches.Rectangle((x0, y0), width, height,
                                 linewidth=linewidth, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        if show_text:
            text_str = f'{class_names[cls]} {score:.2f}'
            text_x = x0
            text_y = y0 - 1   # 默认显示在框上方

            if text_y < fontsize:
                text_y = y1 + fontsize

            text_color = get_lighter_color(color, factor=1.3, min_v=0.7)

            bbox_props = dict(
                facecolor=(0, 0, 0, 0.6),
                edgecolor='none',
                pad=1.5
            )

            ax.text(text_x, text_y, text_str,
                    fontsize=fontsize,
                    color=text_color,
                    bbox=bbox_props,
                    verticalalignment='top', horizontalalignment='left')

    # 将画布内容转为 numpy 数组（RGBA -> RGB）
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_plot = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    if save_vis_path is not None:
        plt.imsave(save_vis_path, img_plot)

    return img_plot