import cv2
import albumentations as A
import numpy as np
import random






class Transforms():
    '''数据预处理/数据增强(基于albumentations库)
       https://albumentations.ai/docs/api_reference/full_reference/
    '''
    def __init__(self, img_size, box_format='coco'):
        '''
            imgSize:    网络接受的输入图像尺寸
            box_format: 'yolo':norm(cxcywh), 'coco':xywh
        '''
        # OpenAI-CLIP:
        self.img_mean = (0.48145466, 0.4578275, 0.40821073)
        self.img_std = (0.26862954, 0.26130258, 0.27577711)
        # DINOv3 LVD-1689M:
        # self.img_mean = (0.485, 0.456, 0.406)
        # self.img_std = (0.229, 0.224, 0.225)
        # DINOv3 SAT-493M:
        # self.img_mean = (0.430, 0.411, 0.296)
        # self.img_std = (0.213, 0.156, 0.143)

        self.img_size = img_size
        max_size = max(img_size[0], img_size[1])
        self.pad_value = [128,128,128]
        self.CoarseDropout = A.Compose([
                # 随机掩码 (第一个针对分类任务效果好些)
                A.CoarseDropout(max_holes=60, max_height=15, max_width=15, min_holes=30, min_height=5, min_width=5, fill_value=128, p=0.5),
        ])
        # 训练时增强
        self.train_transform = A.Compose([
                # A.BBoxSafeRandomCrop(p=0.5, erosion_rate=0.0, min_crop_height=0.3, min_crop_width=0.3),
                # 随机翻转
                A.HorizontalFlip(p=0.5),
                # NOTE:下面这两个只能在DOTA上用:
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # 参数：随机色调、饱和度、值变化
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
                # 随机对比度增强
                A.CLAHE(p=0.1),
                # 高斯噪声
                A.GaussNoise(var_limit=(0.05, 0.09), p=0.4),     
                # 随机转为灰度图
                A.ToGray(p=0.01),
                A.OneOf([
                    # 使用随机大小的内核将运动模糊应用于输入图像
                    A.MotionBlur(p=0.2),   
                    # 中值滤波
                    A.MedianBlur(blur_limit=3, p=0.1),    
                    # 使用随机大小的内核模糊输入图像
                    A.Blur(blur_limit=3, p=0.1),  
                ], p=0.2),
            ],
            bbox_params=A.BboxParams(format=box_format, min_area=0, min_visibility=0.0, label_fields=['category_ids']),
        )
        # 验证时增强
        self.valid_transform = A.Compose([
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=max_size),
                # 较短的边做padding
                A.PadIfNeeded(img_size[0], img_size[1], border_mode=cv2.BORDER_CONSTANT, value=[128,128,128]),
                A.Normalize(mean=self.img_mean, std=self.img_std),
            ],
            bbox_params=A.BboxParams(format=box_format, min_area=0, min_visibility=0.0, label_fields=['category_ids']),
        )
        # 测试时增强(不padding黑边)
        self.test_transform = A.Compose([
                # 最长边限制为imgSize
                A.LongestMaxSize(max_size=max_size),
                # 较短的边做padding
                # A.PadIfNeeded(img_size[0], img_size[1], border_mode=cv2.BORDER_CONSTANT, value=[128,128,128]),
                A.Normalize(mean=self.img_mean, std=self.img_std),
        ])


    def mosaic4(self, images, bboxes, labels, jitter=0.2, scale=.5):
        """mosaic数据增强, 将四张图像拼在一起
            Args:
                images: list(img1, img2, img3, img4)
                bboxes: list(bboxes1, bboxes2, bboxes3, bboxes4)
                labels: list(labels1, labels2, labels3, labels4)
                jitter: 长宽缩放尺寸的范围
                scale:  尺度缩放的最小值
        """
        H, W = self.img_size
        # 随机选取放置图像的中心位置
        cx = int(random.uniform(0.3, 0.7) * W)
        cy = int(random.uniform(0.3, 0.7) * H)
        # OpenCV 使用 (H, W, C) 格式
        mosaic_img = np.ones((H, W, 3), dtype=np.uint8) * 128
        for i in range(4):
            bboxes[i] = np.array(bboxes[i])
            labels[i] = np.array(labels[i])
            h, w, _ = images[i].shape
            # 对图像进行缩放并且进行长和宽的扭曲
            scale = random.uniform(scale, 1)
            scale_w = random.uniform(1-jitter,1+jitter) * scale
            scale_h = random.uniform(1-jitter,1+jitter) * scale
            new_w, new_h = int(w * scale_w), int(h * scale_h)
            # 对图像进行缩放
            images[i] = cv2.resize(images[i], (new_w, new_h))
            # 对box进行缩放
            bboxes[i][:, [0,2]] *= scale_w
            bboxes[i][:, [1,3]] *= scale_h
            # 图像mosaic到一张图像上:
            # numpy数组索引: [y_start:y_end, x_start:x_end, :]
            if i==0: 
                mosaic_img[max(cy-new_h, 0):cy, max(cx-new_w, 0):cx, :] = images[i][max(0, new_h-cy):, max(0, new_w-cx):, :]
                # 对图像进行平移
                bboxes[i][:,0] += (cx-new_w)
                bboxes[i][:,1] += (cy-new_h)
            if i==1:
                mosaic_img[max(cy-new_h, 0):cy, cx:min(W, cx+new_w), :] = images[i][max(0, new_h-cy):, :min(new_w, W-cx), :]
                # 对图像进行平移
                bboxes[i][:,0] += cx
                bboxes[i][:,1] += (cy-new_h)
            if i==2: 
                mosaic_img[cy:min(H, cy+new_h), max(cx-new_w, 0):cx, :] = images[i][:min(new_h, H-cy), max(0, new_w-cx):, :]
                # 对图像进行平移
                bboxes[i][:,0] += (cx-new_w)
                bboxes[i][:,1] += cy
            if i==3: 
                # 对图像进行平移
                bboxes[i][:,0] += cx
                bboxes[i][:,1] += cy
                mosaic_img[cy:min(H, cy+new_h), cx:min(W, cx+new_w), :] = images[i][:min(new_h, H-cy), :min(new_w, W-cx), :]
            # 和边界处理 + 舍弃太小的框
            bboxes[i][:,2] += bboxes[i][:,0]
            bboxes[i][:,3] += bboxes[i][:,1]
            bboxes[i][:, [0,2]] = np.clip(bboxes[i][:, [0,2]], 0, W)
            bboxes[i][:, [1,3]] = np.clip(bboxes[i][:, [1,3]], 0, H)
            bboxes[i][:,2] -= bboxes[i][:,0]
            bboxes[i][:,3] -= bboxes[i][:,1]
            keep = np.where(np.logical_and(bboxes[i][:,2]>4, bboxes[i][:,3]>4))[0]
            bboxes[i] = bboxes[i][keep]
            labels[i] = labels[i][keep]

        labels = np.concatenate(labels, axis=0)
        bboxes = [np.array(b).reshape(-1, 4) if len(b) > 0 else np.zeros((0, 4)) for b in bboxes]
        bboxes = np.concatenate(bboxes, axis=0)

        return mosaic_img, bboxes, labels



    def mixUp(self, images, bboxes, labels):
        """mixUp数据增强: https://arxiv.org/pdf/1710.09412.pdf.
            Args:
                images: list(img1, img2)
                bboxes: list(bboxes1, bboxes2)
                labels: list(labels1, labels2)
        """
        # mixup 两张图像所占比例(分布在0.5附近)
        r = np.random.beta(32.0, 32.0)
        mixup_image = (images[0] * r + images[1] * (1 - r))
        # 标签和框应该拼接而不是相加
        mixup_labels = np.concatenate(labels, axis=0)
        # 确保 bbox 数组维度一致: 空 bbox 统一为 (0, 4) 避免 concatenate 报错
        bboxes = [np.array(b).reshape(-1, 4) if len(b) > 0 else np.zeros((0, 4)) for b in bboxes]
        mixup_boxes = np.concatenate(bboxes, axis=0)
        return mixup_image, mixup_boxes, mixup_labels
