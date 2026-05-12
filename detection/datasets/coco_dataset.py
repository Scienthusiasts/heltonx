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
from tqdm import tqdm
from pycocotools.coco import COCO
# 自定义
from heltonx.utils.register import DATASETS
from heltonx.utils.utils import seed_everything, worker_init_fn, natural_key
from detection.datasets.preprocess import Transforms
from detection.datasets.base_dataset import BaseDetDataset
# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True







@DATASETS.register
class COCODataset(BaseDetDataset):

    def __init__(self, nc, cat_names, ann_json_path, img_dir, img_size, mode, mosaic_p=0, mixup_p=0, map=None):
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径
        Args:
            nc:            数据集类别数
            ann_json_path: COCO annotation 文件路径
            img_dir:       图像的根目录
            img_size:      网络要求输入的图像尺寸
            mode:          train / valid
            map:           categories字段映射(COCO有80个类, 但是id却有到90)
        '''     
        self.mode = mode
        self.nc = nc
        self.cat_names = cat_names
        self.img_size = img_size
        self.tf = Transforms(img_size, 'coco')
        self.img_dir = img_dir
        self.ann_json_path = ann_json_path
        self.mosaic_p=mosaic_p
        self.mixup_p=mixup_p
        self.map = map
        self.inv_map = {v: k for k, v in self.map.items()} if self.map else None
                
        '''使用通用 DDP 加载(只在rank0计算过滤结果, 各rank独立初始化COCO API)'''
        # 各 rank 独立初始化 COCO API（读取本地 annotation JSON，无状态共享需求）
        self.coco = COCO(self.ann_json_path)

        def _load_data():
            # 获取数据集中所有图像对应的imgId
            img_inds = self.coco.getImgIds()
            # 过滤掉那些没有框的图像（仅在 rank 0 执行，结果广播给其他 rank）
            filter_img_inds = self.filter_img_by_id(self.coco, img_inds)
            # 数据集大小
            dataset_num = len(filter_img_inds)
            return dict(img_inds=img_inds, filter_img_inds=filter_img_inds, dataset_num=dataset_num)
        data = self.ddp_safe_load(_load_data)
        self.img_inds = data['img_inds']
        self.filter_img_inds = data['filter_img_inds']
        self.dataset_num = data['dataset_num']
        

    def __getitem__(self, index):
        '''重载data.Dataset父类方法, 获取数据集中数据内容
           这里通过pycocotools来读取图像和标签
           Returns:
               image:    [1, H, W]
               boxes:    list([[x, y, w, h], ..., [...]], ..., [...])
               labels:   ([int, ...], ..., [...])
               img_id:   list[img_id, ...] 评估时会用到, 其他时候用不到
               raw_size: 原图(未resize)的尺寸 评估时会用到, 其他时候用不到
        '''   
        img_id = self.filter_img_inds[index]
        imgInfo = self.coco.loadImgs(img_id)[0]
        raw_size = [imgInfo['height'], imgInfo['width']]
        # 通过index获得图像, 图像的框, 以及框的标签
        image, boxes, labels = self.get_data_by_index(index)
        # 数据预处理与增强
        image, boxes, labels = self.augment(image, boxes, labels)
        # id映射
        if self.map != None:
            labels = [self.map[i] for i in labels]
        labels = np.array(labels)
        boxes = np.array(boxes, dtype=np.float32)
        
        # box是未归一化的xywh
        return image.transpose(2,0,1), boxes, labels, img_id, raw_size




    def get_data_by_index(self, index):
        '''通过index获得图像, 图像的框, 以及框的标签
        Args:
            - index:  数据集里数据的索引
        Returns:
            - image: 
            - box:      
            - label: 
        '''          
        # 通过imgId获取图像信息imgInfo: 例:{'id': 12465, 'license': 1, 'height': 375, 'width': 500, 'file_name': '2011_003115.jpg'}
        imgId = self.filter_img_inds[index]
        imgInfo = self.coco.loadImgs(imgId)[0]
        # 载入图像 (通过imgInfo获取图像名，得到图像路径)               
        image = Image.open(os.path.join(self.img_dir, imgInfo['file_name']))
        image = np.array(image.convert('RGB'))
        # 得到图像里包含的BBox的所有id
        imgAnnIds = self.coco.getAnnIds(imgIds=imgId)   
        # 通过BBox的id找到对应的BBox信息
        anns = self.coco.loadAnns(imgAnnIds) 
        # 获取BBox的坐标和类别
        labels, boxes = [], []
        for ann in anns:
            # 过滤掉稠密聚集的标注框
            if ann['iscrowd'] == 1: continue
            # if ann['ignore'] == 1: continue
            labelName = ann['category_id']
            labels.append(labelName)
            boxes.append(ann['bbox'])
        labels = np.array(labels)
        boxes = np.array(boxes)

        return image, boxes, labels
    


    def filter_img_by_id(self, coco, img_inds):
        '''过滤掉没有有效标注的图像

        优化：一次性加载全部 anns/imgInfos 构建 {imgId: ann_list} 和 {imgId: img_info} 映射，
        避免对每张图像重复调用 getAnnIds / loadImgs（118K → 1 次调用）。
        '''
        print('filtering no objects images...')

        # 一次性加载全部 anns，构建 {imgId: ann_list} 映射
        all_ann_ids = coco.getAnnIds()
        all_anns = coco.loadAnns(all_ann_ids)
        img_id_to_anns = {}
        for ann in tqdm(all_anns, desc='building img->ann map'):
            img_id = ann['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)

        # 一次性加载全部 image info（触发 coco.imgs 填充整个 dict）
        # 使用传入的 img_inds（即全部 imgId），避免重复调用 coco.getImgIds()
        coco.loadImgs(img_inds)  # 内部会填充 coco.imgs dict
        ignore_names = {'000000200365.jpg', '000000550395.jpg', '9999985_00000_d_0000020.jpg'}

        filter_img_inds = []
        for img_id in tqdm(img_inds, desc='filtering imgs'):
            anns = img_id_to_anns.get(img_id, [])
            if len(anns) == 0:
                continue
            # 跳过所有 ann 均为 iscrowd 的图像
            if all(ann.get('iscrowd', 0) for ann in anns):
                continue
            # 特殊文件名过滤（存在 w/h=0 的 bbox）
            img_info = coco.imgs.get(img_id)
            if img_info and img_info['file_name'] in ignore_names:
                continue
            filter_img_inds.append(img_id)

        return filter_img_inds





    # DataLoader中collate_fn参数使用
    # 由于检测数据集每张图像上的目标数量不一
    # 因此需要自定义的如何组织一个batch里输出的内容
    @staticmethod
    def dataset_collate(batch):
        images = []
        bboxes = []
        labels = []
        img_ids = []
        raw_sizes = []
        for img, box, label, img_id, raw_size in batch:
            images.append(img)
            bboxes.append(box)
            labels.append(label)
            img_ids.append(img_id)
            raw_sizes.append(raw_size)

        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
        labels = [torch.from_numpy(ann).type(torch.LongTensor) for ann in labels]
        return images, bboxes, labels, img_ids, raw_sizes
    



    def _vis_cocodataset_batch(self, epoch, step, batch):
        '''可视化训练集一个batch
        '''
        # COCO
        images, boxes, labels, img_ids, raw_size = batch[0], batch[1], batch[2], batch[3], batch[4]
        print(raw_size)
        # 图像均值 标准差
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([[0.229, 0.224, 0.225]]) 
        plt.figure(figsize = (12,12))
        # 可视化一个batch
        for idx, imgBoxLabel in enumerate(zip(images, boxes, labels)):
            img, boxes, labels = imgBoxLabel
            img, boxes, labels = img.cpu().numpy(), boxes.cpu().numpy(), labels.cpu().numpy()
            ax = plt.subplot(4,4,idx+1)
            img = img.transpose((1,2,0))
            # 由于在数据预处理时我们对数据进行了标准归一化，可视化的时候需要将其还原
            img = np.clip(img * std + mean, 0, 1)
            for box, label in zip(boxes, labels):
                x, y, w, h = round(box[0]), round(box[1]), round(box[2]), round(box[3])
                # 绘制框
                ax.add_patch(plt.Rectangle((x, y), w, h, color='blue', fill=False, linewidth=1))
                # 绘制类别
                ax.text(x, y, self.cat_names[label], bbox={'facecolor':'white', 'alpha':0.5})

            plt.imshow(img)
            plt.axis("off")
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
        plt.savefig(f'./epoch{epoch}_step{step}.jpg', dpi=300)








if __name__ == '__main__':
    from torch.utils.data import DataLoader
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
    cat_maps = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 
        24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29, 35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38, 44:39, 46:40, 
        47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49, 56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59, 
        67:60, 70:61, 72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73, 85:74, 86:75, 87:76, 88:77, 89:78, 90:79}

    # 配置字典
    img_dir = r'/mnt/yht/data/COCO/train2017'
    ann_json_path=r'/mnt/yht/data/COCO/annotations/instances_train2017.json'
    cfg = dict(
        dataset_cfg = dict(
            type="COCODataset",
            nc=80, 
            cat_names=cat_names,
            ann_json_path=ann_json_path, 
            img_dir=img_dir,
            img_size=[800, 800], 
            mode='train', 
            mosaic_p=0.5, 
            mixup_p=0.0,
            map=cat_maps
        ),
        bs=16,
        seed=42,
        shuffle=True,
    )
    dataset_cfg = cfg["dataset_cfg"]
    seed_everything(cfg["seed"])
    train_dataset = DATASETS.build_from_cfg(dataset_cfg)
    print(f'数据集大小:{train_dataset.__len__()}')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=cfg["bs"], shuffle=cfg["shuffle"], num_workers=8, collate_fn=train_dataset.dataset_collate, worker_init_fn=partial(worker_init_fn, seed=cfg["seed"]))
    # 输出数据格式
    for epoch in range(1, 10):
        for step, batch in enumerate(train_data_loader):
            print(f"epoch:{epoch}, batch:{step}")
            if step % 100 == 0:
                # 可视化一个batch里的图像
                train_dataset._vis_cocodataset_batch(epoch, step, batch)