import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from functools import partial
import matplotlib.pyplot as plt
import os
import torch.distributed as dist
import random
import cv2
# 自定义
from heltonx.utils.register import DATASETS
from heltonx.utils.wrappers import DDPSafeDataset
from detection.datasets.preprocess import Transforms
# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True







@DATASETS.register
class BaseDetDataset(Dataset, DDPSafeDataset):

    def __init__(self, img_size, format_mode, mode=True, mosaic_p=0, mixup_p=0):
        """__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径
            img_size:    图像尺寸, 如[800, 800]
            format_mode: 数据集标注格式, coco / yolo
            mode:        train / valid
            mosaic_p:    self.mode=='train'时采用, 对数据采取mosaic增强的概率
            mixup_p:     self.mode=='train'时采用, 对数据采取mixup增强的概率
        """      
        self.mode = mode
        self.img_size = img_size
        self.tf = Transforms(img_size, format_mode)
        self.mosaic_p=mosaic_p
        self.mixup_p=mixup_p
        # 数据集大小
        self.dataset_num = 0


    def __len__(self):
        '''重载data.Dataset父类方法, 返回数据集大小
        '''
        return self.dataset_num
    

    def get_data_by_index(self, index):
        pass


    def augment(self, image, boxes, labels):
        '''所有数据增强+预处理操作(顺序不能乱!)
        '''   
        if self.mode == 'train':
            # 基本的数据增强
            image, boxes, labels = self.train_aug(image, boxes, labels)
            # mosaic数据增强
            image, boxes, labels = self.mosaic4(image, boxes, labels, p=self.mosaic_p)
        # 数据预处理(pad成统一size, 归一化)
        image, boxes, labels = self.normal_aug(image, boxes, labels)
        if self.mode == 'train':
            # mixup数据增强
            image, boxes, labels = self.mixUp(image, boxes, labels, p=self.mixup_p)

        return image, boxes, labels

        
    def train_aug(self, image, boxes, labels):
        """基于albumentations库的训练时数据增强
        """
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            print(f"[WARN] Skipping augmentation due to invalid image shape: {image.shape}")
            return image, boxes, labels
        # albumentation的图像维度得是[W,H,C]
        train_trans = self.tf.train_transform(image=image, bboxes=boxes, category_ids=labels)
        image, boxes, labels = train_trans['image'], train_trans['bboxes'], train_trans['category_ids']
        coarse_drop_trans = self.tf.CoarseDropout(image=image)
        image = coarse_drop_trans['image']
        # 这里的box是coco格式(xywh)
        return image, boxes, labels
        


    def normal_aug(self, image, boxes, labels):
        """基于albumentations库的基础数据预处理
        """
        normal_trans = self.tf.valid_transform(image=image, bboxes=boxes, category_ids=labels)
        image, boxes, labels = normal_trans['image'], normal_trans['bboxes'], normal_trans['category_ids']
        # 这里的box是coco格式(xywh)
        return image, boxes, labels
    



    def mixUp(self, image1, boxes1, labels1, p=0.5):
        """mixUp数据增强: https://arxiv.org/pdf/1710.09412.pdf.
          (需要两张图像大小一致, 因此必须在基础数据预处理之后)
        """
        if (np.random.rand() >= p):
            return image1, boxes1, labels1
        
        index2 = np.random.randint(self.dataset_num)
        image2, boxes2, labels2 = self.get_data_by_index(index2)
        image2, boxes2, labels2 = self.train_aug(image2, boxes2, labels2)
        image2, boxes2, labels2 = self.normal_aug(image2, boxes2, labels2)
        assert image2.shape[0] > 0 and image2.shape[1] > 0, f"Invalid shape after mixup: {image2.shape}"
        images = [image1, image2]
        bboxes = [boxes1, boxes2]
        labels = [labels1, labels2]
        mixup_image, mixup_boxes, mixup_labels = self.tf.mixUp(images, bboxes, labels)
        return mixup_image, mixup_boxes, mixup_labels






    def mosaic4(self, image1, boxes1, labels1, jitter=0.2, scale=.5, p=0.5):
        """mosaic数据增强, 将四张图像拼在一起
        """
        if (np.random.rand() >= p):
            return image1, boxes1, labels1
        
        # 随机选取其他3张图像的索引
        indexs = np.random.randint(self.dataset_num, size=3)
        # 读取其余3张图像, 对图像进行数据增强
        image2, boxes2, labels2 = self.get_data_by_index(indexs[0])
        image3, boxes3, labels3 = self.get_data_by_index(indexs[1])
        image4, boxes4, labels4 = self.get_data_by_index(indexs[2])
        image2, boxes2, labels2 = self.train_aug(image2, boxes2, labels2)
        image3, boxes3, labels3 = self.train_aug(image3, boxes3, labels3)
        image4, boxes4, labels4 = self.train_aug(image4, boxes4, labels4)
        H, W = self.img_size
        images = [image1, image2, image3, image4]
        bboxes = [boxes1, boxes2, boxes3, boxes4]
        labels = [labels1, labels2, labels3, labels4]
        # mosaic数据增强
        mosaic_img, bboxes, labels = self.tf.mosaic4(images, bboxes, labels, jitter, scale)
        return mosaic_img, bboxes, labels







    # DataLoader中collate_fn参数使用
    # 由于检测数据集每张图像上的目标数量不一
    # 因此需要自定义的如何组织一个batch里输出的内容
    @staticmethod
    def dataset_collate(batch):
        images = []
        bboxes = []
        labels = []
        for img, box, label in batch:
            images.append(img)
            bboxes.append(box)
            labels.append(label)

        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
        labels = [torch.from_numpy(ann).type(torch.LongTensor) for ann in labels]
        return images, bboxes, labels





