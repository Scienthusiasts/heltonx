# coding=utf-8
"""OBB 检测推理脚本

参考 detection/tools/test.py, 针对旋转框检测 (xywhr 格式) 适配:
- 推理输出 xywhr (5维) 而非 xyxy (4维)
- 可视化绘制旋转框 (用 cv2.minAreaRect + 绘制四边形)
"""

import os
import argparse
import math
import colorsys
import torch
import numpy as np
from PIL import Image, ImageFile
from detectionobb.datasets.preprocess_obb import OBBTransforms
from detectionobb.utils.eval_utils_obb import map_rboxes_to_origin_size
from heltonx.utils.register import MODELS
from heltonx.utils.utils import dynamic_import_class
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_tensor_to_multiple(img, n):
    """将 [H, W, C] tensor 的 H 和 W 调整为最接近 n 的倍数"""
    H, W, C = img.shape
    new_H = int(round(H / n) * n)
    new_W = int(round(W / n) * n)
    new_H = max(n, new_H)
    new_W = max(n, new_W)
    img = img.permute(2, 0, 1).unsqueeze(0)
    resized_img = torch.nn.functional.interpolate(
        img, size=(new_H, new_W), mode='bilinear', align_corners=False)
    resized_img = resized_img.squeeze(0).permute(1, 2, 0)
    return resized_img


def generate_class_colors(cat_names):
    """为每个类别生成确定性颜色 (HSV 均匀分布, 与 dota_dataset.py 风格一致)"""
    colors = {}
    for i, cat in enumerate(cat_names):
        hue = i / len(cat_names)
        r, g, b = [int(c * 255) for c in colorsys.hls_to_rgb(hue, 0.5, 0.7)]
        colors[cat] = (r / 255, g / 255, b / 255)  # matplotlib 期望 [0,1]
    return colors


def draw_obb_on_image(image, boxes, classes, scores, cat_names, colors,
                      save_path, show_text=True, thickness=2):
    """使用 matplotlib 绘制旋转框 (与 dota_dataset.py _vis_dota_batch 一致)

    Args:
        image (np.ndarray): [H, W, 3] 原图 (RGB)
        boxes (np.ndarray): [N, 5] xywhr
        classes (np.ndarray): [N] 类别索引
        scores (np.ndarray): [N] 置信度
        cat_names (list): 类别名称列表
        colors (dict): 类别→matplotlib颜色元组
        save_path (str): 保存路径
        show_text (bool): 是否显示标签
        thickness (int): 框线宽度
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    ax.axis('off')

    # xywhr → 4 顶点坐标
    corners = xywhr2xyxyxyxy_np(boxes)  # [N, 4, 2]

    for i in range(len(boxes)):
        cls_idx = int(classes[i])
        color = colors.get(cat_names[cls_idx], (0, 1, 0))  # 默认绿色

        # 绘制旋转框 (与 dota_dataset.py 一致)
        poly = Polygon(corners[i], fill=False, edgecolor=color, linewidth=thickness)
        ax.add_patch(poly)

        if show_text:
            name = cat_names[cls_idx]
            label = f'{name} {scores[i]:.2f}'
            ax.text(corners[i, 0, 0], corners[i, 0, 1] - 5, label,
                    color='white', fontsize=8,
                    bbox=dict(facecolor=color, alpha=0.6, pad=2, edgecolor='none'))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def xywhr2xyxyxyxy_np(xywhr):
    """numpy 版: 将 xywhr [N, 5] 转为 4 顶点坐标 [N, 4, 2]

    xywhr: (cx, cy, w, h, angle_rad)
    """
    cx, cy, w, h, angle = xywhr[:, 0], xywhr[:, 1], xywhr[:, 2], xywhr[:, 3], xywhr[:, 4]
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # 四个角点相对于中心的偏移
    hw, hh = w / 2, h / 2
    dx = np.stack([hw, -hw, -hw, hw], axis=-1)  # [N, 4]
    dy = np.stack([hh, hh, -hh, -hh], axis=-1)  # [N, 4]

    # 旋转
    x = cx[:, None] + dx * cos_a[:, None] - dy * sin_a[:, None]
    y = cy[:, None] + dx * sin_a[:, None] + dy * cos_a[:, None]

    corners = np.stack([x, y], axis=-1)  # [N, 4, 2]
    return corners


def infer_single_img(model, device, img_path, cat_names, save_vis_path,
                     img_size=[1024, 1024], show_text=True, score_thr=0.05):
    """推理一张图并保存可视化结果

    Args:
        model: OBB 检测模型
        device: 计算设备
        img_path: 图像路径
        cat_names: 类别名称列表
        save_vis_path: 可视化保存路径
        img_size: 网络输入尺寸
        show_text: 是否显示标签文字
        score_thr: 置信度阈值 (覆盖模型默认值)

    Returns:
        boxes, scores, classes: 推理结果
    """
    tf = OBBTransforms(img_size)
    colors = generate_class_colors(cat_names)

    # 读取原始图像
    image = np.array(Image.open(img_path).convert('RGB'))
    H, W = image.shape[:2]

    # 预处理: resize + pad + normalize
    transformed = tf.valid_transform(image=image)
    tensor_img = torch.tensor(transformed['image'])  # [H, W, C] normalized

    # 可能需要调整到 stride 倍数
    tensor_img = resize_tensor_to_multiple(tensor_img, 32)

    # 获取实际缩放后尺寸
    resized_h, resized_w = tensor_img.shape[0], tensor_img.shape[1]
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 推理
    with torch.no_grad():
        boxes, scores, classes = model.infer(tensor_img)

    if len(boxes) == 0:
        print(f'No objects detected in: {img_path}')
        # 保存空白原图
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        ax.axis('off')
        plt.tight_layout()
        fig.savefig(save_vis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return boxes, scores, classes

    # 映射回原图尺寸
    boxes = map_rboxes_to_origin_size(boxes, (H, W), resized_h)

    # 绘制可视化 (matplotlib, dota_dataset.py 风格)
    draw_obb_on_image(image, boxes, classes, scores,
                      cat_names, colors, save_vis_path, show_text=show_text)

    # 统计检测结果
    from collections import Counter
    detect_cls = dict(Counter(classes.tolist()))
    detect_name = {cat_names[k]: v for k, v in detect_cls.items()}
    print(f'Detect result: {detect_name}')
    print(f'Saved visualization to: {save_vis_path}')

    return boxes, scores, classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OBB Detection Inference')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--ckpt', type=str, default=None, help='权重文件路径 (覆盖config中的load_ckpt)')
    parser.add_argument('--thr', type=float, default=0.05, help='推理时的得分阈值')
    parser.add_argument('--img', type=str, default='detectionobb/demo/dota1.0/P0903__1024__295___0.png', help='待推理图像路径')
    parser.add_argument('--save', type=str, default='detobb_res.jpg', help='可视化结果保存路径')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 从 config 文件加载参数
    cargs = dynamic_import_class(args.config, get_class=False)

    cat_names = cargs.cat_names
    img_size = cargs.img_size

    # 模型配置
    model_cfgs = cargs.model_cfgs.copy()
    # ★ 修复: 传入 checkpoint 路径
    if args.ckpt is not None:
        model_cfgs['load_ckpt'] = args.ckpt
    model_cfgs['score_thr'] = args.thr


    nc = len(cat_names)
    print(f"Model: {model_cfgs['type']}, NC: {nc}, ImgSize: {img_size}")
    print(f"Ckpt: {model_cfgs.get('load_ckpt', 'None')}")

    # 需要 import 才能注册
    from detectionobb import *

    model = MODELS.build_from_cfg(model_cfgs).to(device)
    model.eval()
    infer_single_img(model, device, args.img, cat_names, args.save,
                     img_size=img_size, show_text=True, score_thr=args.thr)

    # Usage:
    # /mnt/yht/env/yht_pretrain/bin/python -m detectionobb.tools.test --config detectionobb/configs/yolo26obb_dota_ddp.py --ckpt log/yolo26s_obb_dota_obb_train_ddp/2026-06-29-14-00-45_train_ddp/last.pt
