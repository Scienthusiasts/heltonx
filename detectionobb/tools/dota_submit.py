# coding=utf-8
"""DOTA 测试集推理 + 提交文件生成

使用训练好的模型对 DOTA 测试集进行推理，生成可直接提交官方服务器的 zip 文件。

用法:
    python -m detectionobb.tools.dota_submit \\
        --config detectionobb/configs/yolo26obb_dota_ddp.py \\
        --ckpt log/yolo26s_obb_dota_obb_train_ddp/xxx/last.pt \\
        --test_dir /path/to/dota/test \\
        --output submit.zip \\
        --thr 0.01
"""
import os
import sys
import shutil
import zipfile
import argparse
import math

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from detectionobb import *
from detectionobb.utils.obb_ops import xywhr2xyxyxyxy, regularize_rboxes
from heltonx.utils.register import MODELS
from heltonx.utils.utils import dynamic_import_class

# 图像归一化参数 (与 OBBTransforms 一致)
IMG_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
IMG_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def map_rboxes_to_origin_size(rboxes, orig_size, target_size):
    """将 [S, S] 图像上的 xywhr 预测映射回原始图像 [H, W] 尺寸"""
    if len(rboxes) == 0:
        return rboxes
    H, W = orig_size
    S = target_size
    r = S / max(H, W)
    if H > W:
        new_w = int(W * r)
        pad_w = (S - new_w) / 2
        pad_h = 0
    else:
        new_h = int(H * r)
        pad_w = 0
        pad_h = (S - new_h) / 2
    rboxes = rboxes.copy().astype(float)
    rboxes[:, 0] -= pad_w
    rboxes[:, 1] -= pad_h
    rboxes[:, 0:4] /= r
    rboxes[:, 0] = np.clip(rboxes[:, 0], 0, W)
    rboxes[:, 1] = np.clip(rboxes[:, 1], 0, H)
    return rboxes


def infer_single_img(model, device, img_path, img_size, score_thr):
    """对单张图片推理, 返回 (boxes, scores, classes) in patch 坐标

    图片已是 1024×1024 patch, 仅需 normalize + GPU 传输.
    """
    # 读图 (CPU 不可避免)
    img_np = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
    H, W = img_np.shape[:2]

    # normalize + GPU 传输一步到位: (x/255 - mean) / std
    mean = IMG_MEAN.view(1, 3, 1, 1).to(device, non_blocking=True)
    std = IMG_STD.view(1, 3, 1, 1).to(device, non_blocking=True)
    img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
    img = (img / 255.0 - mean) / std

    with torch.no_grad():
        boxes, scores, classes = model.infer(img)

    if len(boxes) == 0:
        return np.zeros((0, 5)), np.zeros(0), np.zeros(0, dtype=np.int32)

    # regularize + 过滤
    boxes_t = torch.from_numpy(boxes).float()
    boxes_t = regularize_rboxes(boxes_t).numpy()
    keep = scores >= score_thr
    return boxes_t[keep], scores[keep], classes[keep]


def main():
    parser = argparse.ArgumentParser(description='DOTA Test Inference & Submission')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--ckpt', type=str, default=None, help='权重文件路径')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='测试集目录, 需含 images/ 子目录')
    parser.add_argument('--output', type=str, default='submit.zip', help='输出zip文件路径')
    parser.add_argument('--thr', type=float, default=0.01, help='推理得分阈值')
    parser.add_argument('--gap', type=int, default=200, help='切分重叠像素')
    parser.add_argument('--subsize', type=int, default=1024, help='切分尺寸')
    parser.add_argument('--scale', type=float, default=1.0, help='测试缩放比例')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载配置 & 模型
    cargs = dynamic_import_class(args.config, get_class=False)
    cat_names = cargs.cat_names
    img_size = cargs.img_size
    nc = len(cat_names)

    model_cfgs = cargs.model_cfgs.copy()
    if args.ckpt is not None:
        model_cfgs['load_ckpt'] = args.ckpt
    model_cfgs['score_thr'] = args.thr

    print(f"Model: {model_cfgs['type']}, NC: {nc}, ImgSize: {img_size}")
    print(f"Ckpt: {model_cfgs.get('load_ckpt', 'None')}")
    print(f"Test dir: {args.test_dir}")

    model = MODELS.build_from_cfg(model_cfgs).to(device)
    model.eval()

    # === Step 1: 切分测试集大图 ===
    temp_dir = os.path.join(os.path.dirname(args.output), '_dota_submit_tmp')
    split_dir = os.path.join(temp_dir, 'split')
    result_raw_dir = os.path.join(temp_dir, 'result_raw')
    result_merged_dir = os.path.join(temp_dir, 'result_merged')

    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(result_raw_dir, exist_ok=True)
    os.makedirs(result_merged_dir, exist_ok=True)

    test_images_dir = os.path.join(args.test_dir, 'images')
    if not os.path.isdir(test_images_dir):
        print(f'Error: {test_images_dir} 不存在, 测试集目录需含 images/ 子目录')
        sys.exit(1)

    # 使用 ImgSplit 切分
    print('\n=== Step 1: Splitting test images ===')
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../DOTA_devkit'))
    from ImgSplit import splitbase

    # 检测实际图片扩展名
    detected_ext = '.png'
    for f in os.listdir(test_images_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
            detected_ext = ext
            break
    print(f'Detected image extension: {detected_ext}')

    # 列出所有图片文件
    img_files = [f for f in os.listdir(test_images_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

    # 判断是否需要切分: 读取第一张图检查尺寸
    first_img_path = os.path.join(test_images_dir, img_files[0])
    first_pil = Image.open(first_img_path)
    img_w, img_h = first_pil.size
    first_pil.close()
    need_split = max(img_h, img_w) > args.subsize

    if need_split:
        # ImgSplit 要求每张图有对应 label 文件. 为每张测试图创建空标注.
        label_dir = os.path.join(args.test_dir, 'labelTxt')
        os.makedirs(label_dir, exist_ok=True)
        for f in tqdm(img_files, desc='Creating placeholder labels'):
            name_no_ext = os.path.splitext(f)[0]
            label_path = os.path.join(label_dir, name_no_ext + '.txt')
            if not os.path.exists(label_path):
                open(label_path, 'w').close()

        split = splitbase(args.test_dir, split_dir, gap=args.gap,
                          subsize=args.subsize, thresh=0.7, ext=detected_ext)
        split.splitdata(args.scale)
        # 使用切分后的图像目录
        infer_images_dir = os.path.join(split_dir, 'images')
    else:
        print(f'Images are already ≤{args.subsize}px, skipping split')
        # 直接使用原图目录
        infer_images_dir = test_images_dir

    # === Step 2: 推理所有图像 ===
    print('\n=== Step 2: Running inference ===')
    # 收集所有待推理图像
    patch_images = []
    for root, dirs, files in os.walk(infer_images_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                patch_images.append(os.path.join(root, f))
    patch_images.sort()

    # 类别→结果行列表
    class_dets = {name: [] for name in cat_names}

    for img_path in tqdm(patch_images, desc='Inferring patches'):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        boxes, scores, classes = infer_single_img(
            model, device, img_path, img_size, args.thr)

        if len(boxes) == 0:
            continue

        # xywhr → 8 顶点 polygon
        corners = xywhr2xyxyxyxy(torch.from_numpy(boxes)).numpy()
        for i in range(len(boxes)):
            cls_idx = int(classes[i])
            cls_name = cat_names[cls_idx]
            poly = corners[i].reshape(-1)
            poly_str = ' '.join(f'{c:.2f}' for c in poly)
            class_dets[cls_name].append(
                f'{img_name} {scores[i]:.4f} {poly_str}')

    # === Step 3: 写入 Task1 格式结果 ===
    print('\n=== Step 3: Writing detection results ===')
    for cls_name, dets in tqdm(class_dets.items(), desc='Writing Task1 files'):
        if not dets:
            continue
        save_path = os.path.join(result_raw_dir, f'Task1_{cls_name}.txt')
        with open(save_path, 'w') as f:
            f.write('\n'.join(dets))

    # === Step 4: 合并切分结果 (ResultMerge) ===
    print('\n=== Step 4: Merging patch results ===')
    from ResultMerge import mergebypoly
    mergebypoly(result_raw_dir, result_merged_dir)

    # === Step 5: 打包 zip ===
    print('\n=== Step 5: Creating submission zip ===')
    merge_files = [f for f in os.listdir(result_merged_dir)
                   if f.startswith('Task1_') and f.endswith('.txt')]
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in tqdm(merge_files, desc='Zipping results'):
            fpath = os.path.join(result_merged_dir, fname)
            zf.write(fpath, fname)

    # 清理临时文件
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f'\nDone! Submission saved to: {args.output}')
    print(f'Zip contents: {len(zipfile.ZipFile(args.output).namelist())} files')


if __name__ == '__main__':
    main()



'''
python -m detectionobb.tools.dota_submit \
    --config detectionobb/configs/yolo26obb_dota_ddp.py \
    --ckpt log/yolo26l_obb_dota_obb_train_ddp/2026-07-01-10-42-24_train_ddp/best_val_map.pt \
    --test_dir /mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/test \
    --output log/yolo26l_obb_dota_obb_train_ddp/2026-07-01-10-42-24_train_ddp/submit.zip \
    --thr 0.05
'''