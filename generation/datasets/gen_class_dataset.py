import numpy as np
import torch
from PIL import Image, ImageFile
import torch.utils.data.dataset as data
from torch.utils.data import DataLoader
from torchvision.io import read_image
from functools import partial
import matplotlib.pyplot as plt
import os
import torch.distributed as dist
# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 自定义
from heltonx.utils.register import DATASETS
from heltonx.utils.utils import seed_everything, worker_init_fn
from generation.datasets.preprocess import Transforms








@DATASETS.register
class GenClassDataset(data.Dataset):      
    '''有监督分类/条件生成任务对应的数据集读取方式
    '''
    def __init__(self, img_dir, img_size, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225]):    
        '''
        Args:
            img_dir:  图像数据集的根目录 (需按类别子文件夹存放，如 img_dir/cls_name/xxx.jpg)
            img_size: 网络要求输入的图像尺寸
            img_mean: 归一化的图像均值
            img_std:  归一化的图像标准差
        '''      
        self.img_mean = img_mean
        self.img_std = img_std
        self.img_dir = img_dir
        self.transform = Transforms(img_size, self.img_mean, self.img_std)
        # 支持的图像扩展名
        IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        # ==========================================
        # 1. 解析类别文件夹，建立 Class -> ID 映射
        # ==========================================
        self.classes = sorted([d.name for d in os.scandir(img_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        
        # ==========================================
        # 2. 遍历所有图像并绑定对应的类别 ID
        # ==========================================
        self.samples = [] # 存储 (img_path, class_id) 元组
        for root, _, files in os.walk(self.img_dir):
            cls_name = os.path.basename(root)
            # 如果当前所在文件夹属于我们解析出的类别
            if cls_name in self.class_to_idx:
                for fname in files:
                    if fname.lower().endswith(IMG_EXTS):
                        path = os.path.join(root, fname)
                        self.samples.append((path, self.class_to_idx[cls_name]))
                        
        # 记录数据集大小
        self.dataSize = len(self.samples)

        # 打印数据集信息
        use_ddp = dist.is_initialized()
        if not use_ddp or (use_ddp and dist.get_rank() == 0):
            print(f'📄 dataset info: 发现 {len(self.classes)} 个类别, 图像总数:{self.dataSize}')


    def __getitem__(self, item):  
        '''获取单张图像和其对应的类别 ID
        '''   
        img_path, label = self.samples[item]
        
        # 读取图片
        img = read_image(img_path).permute(1, 2, 0).numpy()
        # 数据增强
        img = self.albumAug(img)         
        
        # 返回图像和整形标签
        return img.transpose(2, 0, 1), label
    

    def albumAug(self, img):
        """基于albumentations库的基础数据预处理
        """
        trans = self.transform.transform(image=img)          
        img = trans['image']   
        return img


    def __len__(self):
        return self.dataSize


    def dataset_collate(self, batch):
        """
        组装一个 Batch，返回 [batch_imgs, batch_labels]
        """
        images = []
        labels = []
        for img, label in batch:
            images.append(img)
            labels.append(label)
            
        # np -> tensor [B, C, H, W]
        images_tensor = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        # list -> tensor [B,]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return [images_tensor, labels_tensor]
    

    def _vis_GenDataset_batch(self, epoch, step, batch_data):
        '''可视化训练集一个batch，并在上方显示类别 ID
        '''
        batch_imgs, batch_labels = batch_data[0], batch_data[1]
        
        # 图像均值 标准差
        mean = np.array(self.img_mean) 
        std = np.array(self.img_std).reshape(1, -1) 

        B = batch_imgs.shape[0]
        # 动态计算网格大小，避免 batch size 不为 64 时报错
        grid_n = int(np.ceil(np.sqrt(B)))

        fig = plt.figure(figsize=(grid_n * 2.5, grid_n * 2.5), facecolor='white')
        
        for idx in range(B):
            img = batch_imgs[idx].numpy().transpose((1, 2, 0))
            # 反归一化并截断异常值
            img = (img * std + mean).clip(0, 1)
            
            ax = plt.subplot(grid_n, grid_n, idx + 1)
            ax.imshow(img)
            ax.axis("off")
            
            # 显示对应的类别 ID
            label_id = batch_labels[idx].item()
            ax.set_title(f"Class ID: {label_id}", fontsize=10, pad=4, fontweight='bold')

        plt.tight_layout()
        os.makedirs('./vis_img', exist_ok=True)
        save_name = f'./vis_img/epoch{epoch}_step{step}_.jpg'
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"👀 类别条件 Batch 可视化已保存至: {save_name}")









# for test only
if __name__ == '__main__':

    # 配置字典
    img_dir = r'/mnt/yht/data/imagenet-1k-256x256/images/train'
    cfg = {
        "dataset_cfg": {
            "type": "GenClassDataset",
            "img_dir": img_dir,
            "img_size": [256, 256]
        },
        "bs": 64,
        "seed": 42,
        "shuffle": True
    }

    dataset_cfg = cfg["dataset_cfg"]
    seed_everything(cfg["seed"])
    train_dataset = DATASETS.build_from_cfg(dataset_cfg)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=cfg["bs"], shuffle=cfg["shuffle"], num_workers=8, collate_fn=train_dataset.dataset_collate, worker_init_fn=partial(worker_init_fn, seed=cfg["seed"]))
    # 输出数据格式
    for epoch in range(1, 10):
        for step, batch in enumerate(train_data_loader):
            batch_imgs, batch_labels = batch[0], batch[1]
            print(batch_imgs.shape, batch_labels.shape)
            # print(epoch, step)
            if step %100 == 0:
                # 可视化一个batch里的图像
                train_dataset._vis_GenDataset_batch(epoch, step, batch)