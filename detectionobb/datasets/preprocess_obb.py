"""OBB 专用数据预处理/增强

与 detection/datasets/preprocess.py 的区别:
- 支持 8 顶点格式的旋转框
- albumentations 不支持 polygon 格式, 因此几何变换 (翻转/旋转90°/resize/pad)
  需手动同步到 polygon 顶点坐标
- Mosaic/Mixup 时通过 4 顶点坐标变换, 再用 cv2.minAreaRect 重新计算 xywhr
  (非均匀缩放会改变旋转角度, 必须通过顶点变换再重算)
"""

import cv2
import albumentations as A
import numpy as np
import random
import torch
from detectionobb.utils.obb_ops import xyxyxyxy2xywhr, xywhr2xyxyxyxy


class OBBTransforms:
    """OBB 数据预处理/数据增强

    内部格式: 旋转框以 8 顶点 (x1,y1,x2,y2,x3,y3,x4,y4) 形式存储
    albumentations 不支持 polygon bbox, 因此:
    - 颜色/噪声类变换只影响图像, 不需要同步框坐标
    - 几何类变换 (翻转/旋转90°/resize/pad) 手动同步到 polygon 顶点
    """

    def __init__(self, img_size):
        self.img_mean = (0.48145466, 0.4578275, 0.40821073)
        self.img_std = (0.26862954, 0.26130258, 0.27577711)
        self.img_size = img_size
        max_size = max(img_size[0], img_size[1])
        self.pad_value = [128, 128, 128]

        # 训练时增强: 只含颜色/噪声变换 (几何变换已手动同步到 polygon 顶点)
        self.train_transform = A.Compose([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CLAHE(p=0.1),
            A.GaussNoise(var_limit=(0.05, 0.09), p=0.4),
            A.ToGray(p=0.01),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
        ])

        # 验证时增强: resize + pad + normalize (不用 bbox_params, 手动同步)
        self.valid_transform = A.Compose([
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(img_size[0], img_size[1], border_mode=cv2.BORDER_CONSTANT, value=[128, 128, 128]),
            A.Normalize(mean=self.img_mean, std=self.img_std),
        ])

    def apply_train_transform(self, image, rboxes, labels):
        """训练时增强: 手动施加几何变换到 polygon, albumentations 只做颜色/噪声

        流程:
        1. 先手动随机施加几何变换 (HFlip/VFlip/Rotate90) 到图像+corners
        2. 再用 albumentations 做颜色/噪声变换 (只影响图像)
        """
        if len(rboxes) == 0:
            trans = self.train_transform(image=image)
            return trans['image'], np.zeros((0, 5), dtype=np.float32), np.array([], dtype=np.int64)

        # xywhr → 4 顶点
        corners = xywhr2xyxyxyxy(torch.from_numpy(rboxes)).numpy()  # [N, 4, 2]
        h, w = image.shape[:2]

        # 手动施加几何变换
        # HorizontalFlip
        if random.random() < 0.5:
            image = image[:, ::-1, :]
            corners[:, :, 0] = w - 1 - corners[:, :, 0]

        # VerticalFlip
        if random.random() < 0.5:
            image = image[::-1, :, :]
            corners[:, :, 1] = h - 1 - corners[:, :, 1]

        # RandomRotate90 (0/90/180/270)
        # np.rot90(img, k) = 逆时针 k*90°
        # 逆时针90°坐标映射: (x,y) → (y, h-1-x), 新尺寸 (w, h)
        k = random.randint(0, 3)  # 旋转 k*90°
        if k > 0:
            image = np.rot90(image, k)
            for _ in range(k):
                new_x = corners[:, :, 1]
                new_y = h - 1 - corners[:, :, 0]
                corners = np.stack([new_x, new_y], axis=-1)
                h, w = w, h  # 90° 后尺寸互换

        # albumentations 只做颜色/噪声变换 (不含几何)
        trans = self.train_transform(image=image)
        image_out = trans['image']

        # 4 顶点 → xywhr (cv2.minAreaRect 重算角度)
        polygons = corners.reshape(-1, 8)
        rboxes_out = xyxyxyxy2xywhr(torch.from_numpy(polygons.astype(np.float32))).numpy()

        # 过滤退化框: w < 1 或 h < 1 的旋转框会导致 probiou NaN
        if len(rboxes_out) > 0:
            keep = (rboxes_out[:, 2] > 1) & (rboxes_out[:, 3] > 1)
            rboxes_out = rboxes_out[keep]
            labels = labels[keep]

        return image_out, rboxes_out, labels

    def apply_valid_transform(self, image, rboxes, labels):
        """验证时增强: resize + pad + normalize, 手动同步到 corners

        流程:
        1. 计算缩放因子 + pad 偏移
        2. 对 corners 施加缩放+偏移
        3. 用 albumentations 变换图像 (resize+pad+normalize)
        """
        if len(rboxes) == 0:
            trans = self.valid_transform(image=image)
            return trans['image'], np.zeros((0, 5), dtype=np.float32), np.array([], dtype=np.int64)

        h, w = image.shape[:2]
        max_size = max(self.img_size)
        target_h, target_w = self.img_size

        # LongestMaxSize: 按最长边缩放, 保持比例
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # PadIfNeeded: 居中填充到 target_h × target_w
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2

        # corners: 缩放 + 填充偏移
        corners = xywhr2xyxyxyxy(torch.from_numpy(rboxes)).numpy()  # [N, 4, 2]
        corners[:, :, 0] = corners[:, :, 0] * scale + pad_left
        corners[:, :, 1] = corners[:, :, 1] * scale + pad_top

        # albumentations 变换图像
        trans = self.valid_transform(image=image)
        image_out = trans['image']

        # 4 顶点 → xywhr
        polygons = corners.reshape(-1, 8)
        rboxes_out = xyxyxyxy2xywhr(torch.from_numpy(polygons.astype(np.float32))).numpy()

        # 裁剪: 中心在图像内 + 尺度足够
        keep = ((rboxes_out[:, 0] > 0) & (rboxes_out[:, 0] < target_w) &
                (rboxes_out[:, 1] > 0) & (rboxes_out[:, 1] < target_h) &
                (rboxes_out[:, 2] > 1) & (rboxes_out[:, 3] > 1))
        rboxes_out = rboxes_out[keep]
        labels_out = labels[keep]

        return image_out, rboxes_out, labels_out

    def train_aug(self, image, rboxes, labels):
        """训练时增强 (几何变换 + 颜色变换, 在原始分辨率上)"""
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            return image, rboxes, labels
        return self.apply_train_transform(image, rboxes, labels)

    def normal_aug(self, image, rboxes, labels):
        """验证时预处理 (resize + pad + normalize)"""
        return self.apply_valid_transform(image, rboxes, labels)

    def mosaic4(self, images, rboxes_list, labels_list, jitter=0.2, scale=0.5):
        """Mosaic 数据增强 (4 图拼接), 旋转框版本

        关键修正:
        - 非均匀缩放 (scale_w != scale_h) 会改变旋转角度
        - 必须通过 4 顶点坐标进行缩放+平移, 再用 xyxyxyxy2xywhr 重算角度
        - 不能直接对 xywhr 的 cx/cy/w/h 分别乘不同缩放因子

        Args:
            images: list of 4 images (原始分辨率, 已 train_aug)
            rboxes_list: list of 4 rboxes [N_i, 5] xywhr (原始像素坐标)
            labels_list: list of 4 labels
            jitter: 长宽缩放抖动范围
            scale: 尺度缩放最小值
        """
        H, W = self.img_size
        cx = int(random.uniform(0.3, 0.7) * W)
        cy = int(random.uniform(0.3, 0.7) * H)
        mosaic_img = np.ones((H, W, 3), dtype=np.uint8) * 128

        all_rboxes = []
        all_labels = []

        for i in range(4):
            rboxes = np.array(rboxes_list[i], dtype=np.float32)
            labels = np.array(labels_list[i], dtype=np.int64)
            h, w, _ = images[i].shape
            s = random.uniform(scale, 1)
            scale_w = random.uniform(1 - jitter, 1 + jitter) * s
            scale_h = random.uniform(1 - jitter, 1 + jitter) * s
            new_w, new_h = int(w * scale_w), int(h * scale_h)
            images[i] = cv2.resize(images[i], (new_w, new_h))

            if len(rboxes) > 0:
                # ★ 关键修正: 通过 4 顶点坐标做非均匀缩放+平移
                corners = xywhr2xyxyxyxy(torch.from_numpy(rboxes)).numpy()  # [N, 4, 2]

                # 非均匀缩放 (scale_w, scale_h 可能不同 → 角度会变)
                corners[:, :, 0] *= scale_w
                corners[:, :, 1] *= scale_h

                # 计算 quadrant 偏移
                if i == 0:
                    off_x, off_y = cx - new_w, cy - new_h
                elif i == 1:
                    off_x, off_y = cx, cy - new_h
                elif i == 2:
                    off_x, off_y = cx - new_w, cy
                else:
                    off_x, off_y = cx, cy

                # 平移到 mosaic 坐标系
                corners[:, :, 0] += off_x
                corners[:, :, 1] += off_y

                # 4 顶点 → xywhr (cv2.minAreaRect 重算角度)
                polygons = corners.reshape(-1, 8)
                rboxes = xyxyxyxy2xywhr(torch.from_numpy(polygons.astype(np.float32))).numpy()

                # 裁剪: 中心在图像内 + 尺度足够
                keep = ((rboxes[:, 0] > 0) & (rboxes[:, 0] < W) &
                        (rboxes[:, 1] > 0) & (rboxes[:, 1] < H) &
                        (rboxes[:, 2] > 4) & (rboxes[:, 3] > 4))
                rboxes = rboxes[keep]
                labels = labels[keep]

                all_rboxes.append(rboxes)
                all_labels.append(labels)

            # 放置图像
            if i == 0:
                mosaic_img[max(cy - new_h, 0):cy, max(cx - new_w, 0):cx, :] = \
                    images[i][max(0, new_h - cy):, max(0, new_w - cx):, :]
            elif i == 1:
                mosaic_img[max(cy - new_h, 0):cy, cx:min(W, cx + new_w), :] = \
                    images[i][max(0, new_h - cy):, :min(new_w, W - cx), :]
            elif i == 2:
                mosaic_img[cy:min(H, cy + new_h), max(cx - new_w, 0):cx, :] = \
                    images[i][:min(new_h, H - cy), max(0, new_w - cx):, :]
            else:
                mosaic_img[cy:min(H, cy + new_h), cx:min(W, cx + new_w), :] = \
                    images[i][:min(new_h, H - cy), :min(new_w, W - cx), :]

        if len(all_rboxes) > 0:
            all_rboxes = np.concatenate(all_rboxes, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_rboxes = np.zeros((0, 5), dtype=np.float32)
            all_labels = np.array([], dtype=np.int64)

        return mosaic_img, all_rboxes, all_labels

    def mixUp(self, images, rboxes_list, labels_list):
        """MixUp 数据增强, 旋转框版本"""
        r = np.random.beta(32.0, 32.0)
        mixup_image = (images[0] * r + images[1] * (1 - r)).astype(np.uint8)
        rboxes_arrs = [np.array(rb, dtype=np.float32).reshape(-1, 5) for rb in rboxes_list]
        mixup_rboxes = np.concatenate(rboxes_arrs, axis=0) if any(len(a) > 0 for a in rboxes_arrs) else np.zeros((0, 5), dtype=np.float32)
        mixup_labels = np.concatenate(
            [np.array(l, dtype=np.int64) for l in labels_list], axis=0)
        return mixup_image, mixup_rboxes, mixup_labels
