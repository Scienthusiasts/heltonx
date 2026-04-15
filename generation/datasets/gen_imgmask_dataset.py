import numpy as np
import torch
import torch.utils.data.dataset as data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt
import os
import cv2  # 新增 cv2 用于多边形绘制
from torchvision.io import read_image
from matplotlib.gridspec import GridSpec
import torch.distributed as dist
from heltonx.utils.register import DATASETS
from heltonx.utils.utils import seed_everything, worker_init_fn
from generation.datasets.preprocess import Transforms




@DATASETS.register
class ImageMaskDataset(data.Dataset):      
    def __init__(self, img_dir, label_dir, class_names, img_size, mask_size, 
                 img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225], 
                 ori_img_size=(1024, 1024), filter_empty=False):    
        '''
        Args:
            img_dir:      图像数据集目录
            label_dir:    OBB 标签数据集目录
            class_names:  类别列表
            img_size:     图像 resize 的目标尺寸 [H, W]
            mask_size:    掩膜 resize 的目标尺寸 [h, w]
        '''      
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.mask_size = mask_size
        self.ori_img_size = ori_img_size
        self.img_mean = np.array(img_mean)
        self.img_std = np.array(img_std)
        
        self.classes = class_names
        self.num_classes = len(self.classes)
        self.class2idx = {cls_name: idx + 1 for idx, cls_name in enumerate(self.classes)}
        
        # 假设这里有您的 Transforms (只针对图像进行颜色增强和基础 resize)
        self.transform = Transforms(img_size, img_mean, img_std)
        
        IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        # 1. 获取所有图像和标签文件
        all_imgs = {os.path.splitext(f)[0]: os.path.join(root, f) 
                    for root, _, files in os.walk(self.img_dir) 
                    for f in files if f.lower().endswith(IMG_EXTS)}
        
        all_labels = {os.path.splitext(f)[0]: os.path.join(root, f) 
                      for root, _, files in os.walk(self.label_dir) 
                      for f in files if f.lower().endswith('.txt')}

        # 2. 取交集，保证图像和标签一一对应
        common_fnames = set(all_imgs.keys()).intersection(set(all_labels.keys()))
        # 🌟 修复漏洞: 将集合转换为列表，并强制按字母顺序进行排序, 否則即使有seed每次讀取的圖像都是隨機的
        sorted_common_fnames = sorted(list(common_fnames))
        
        self.data_list = []
        for fname in sorted_common_fnames:
            txt_path = all_labels[fname]
            # 3. 过滤空标签
            if filter_empty:
                if self._has_valid_object(txt_path):
                    self.data_list.append((fname, all_imgs[fname], txt_path))
            else:
                self.data_list.append((fname, all_imgs[fname], txt_path))
        self.dataSize = len(self.data_list)
        
        use_ddp = dist.is_initialized()
        if not use_ddp or (use_ddp and dist.get_rank() == 0):
            print(f'匹配到的样本数:{len(common_fnames)}, 过滤空标注后有效文件数:{self.dataSize}')

    def _has_valid_object(self, txt_path):
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9 and parts[8] in self.class2idx:
                        return True 
        except Exception:
            pass
        return False

    def __getitem__(self, item):  
        fname, img_path, txt_path = self.data_list[item]
        ori_H, ori_W = self.ori_img_size
        
        # ==================== 1. 读取并处理图像 ====================
        # 使用 torchvision 读取并转为 [H, W, C]
        img = read_image(img_path).permute(1, 2, 0).numpy()

        # ==================== 2. 读取并处理掩膜 ====================
        mask_H, mask_W = self.mask_size
        obj_masks_ori = np.zeros((self.num_classes, ori_H, ori_W), dtype=np.uint8)

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue
            cls_name = parts[8]
            if cls_name not in self.class2idx: continue
                
            channel_idx = self.class2idx[cls_name] - 1 
            try:
                pts = [float(p) for p in parts[:8]]
            except ValueError: continue 
                
            pts_x, pts_y = np.array(pts[0::2]), np.array(pts[1::2])
            poly_pts = np.stack([pts_x, pts_y], axis=1).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(obj_masks_ori[channel_idx], [poly_pts], color=255)

        # 缩小到 mask_size 并保留软边缘
        obj_masks_resized = np.zeros((self.num_classes, mask_H, mask_W), dtype=np.float32)
        for i in range(self.num_classes):
            if np.any(obj_masks_ori[i]):
                obj_masks_resized[i] = cv2.resize(
                    obj_masks_ori[i], (mask_W, mask_H), interpolation=cv2.INTER_AREA
                )

        fg_max = np.max(obj_masks_resized, axis=0)
        bg_mask = np.expand_dims(255.0 - fg_max, axis=0)
        
        masks = np.concatenate([bg_mask, obj_masks_resized], axis=0) # [16, h, w]
        masks = masks.transpose(1, 2, 0) # [H, W, C]
        trans = self.transform.mask_transform(image=img, mask=masks) 
       
        img = trans['image'].transpose(2, 0, 1)         
        masks = trans['mask'].transpose(2, 0, 1)

        return img, masks, fname
    
    
    def __len__(self):
        return self.dataSize


    def dataset_collate(self, batch):
        imgs, masks, fnames = [], [], []
        for img, mask, fname in batch:
            imgs.append(img)
            masks.append(mask)
            fnames.append(fname)
            
        imgs_tensor = torch.from_numpy(np.stack(imgs, axis=0)).type(torch.FloatTensor)
        # 'area' (区域插值)能有效緩解下采樣帶來的摩爾紋
        imgs_tensor = F.interpolate(input=imgs_tensor, size=self.img_size, mode='area')

        masks = np.stack(masks, axis=0) 
        # [0, 255] -> [-1, 1] (符合扩散模型要求)
        masks_tensor = torch.from_numpy(masks).type(torch.FloatTensor) / 255. * 2. - 1.

        return [imgs_tensor, masks_tensor, fnames]


    def _vis_GenDataset_merge_batch(self, epoch, step, batch_data):
        '''左右分格可视化：左侧显示原图 nxn，右侧显示 Mask nxn
        '''
        batch_imgs, batch_masks, fnames = batch_data[0], batch_data[1], batch_data[2]
        
        imgs_np = batch_imgs.numpy()     # [B, 3, img_H, img_W]
        masks_np = batch_masks.numpy()   # [B, 16, mask_H, mask_W], 值在 [-1, 1] 之间
        B = imgs_np.shape[0]

        # 确定 nxn 网格大小
        grid_n = int(np.ceil(np.sqrt(B)))
        
        # 创建大画板
        fig = plt.figure(figsize=(grid_n * 5, grid_n * 2.5), facecolor='white')
        # 使用 GridSpec 将画板一分为二，左边画图，右边画 Mask
        gs = GridSpec(1, 2, figure=fig, wspace=0.1)
        
        # 在左右两边分别创建细分的网格
        gs_left = gs[0].subgridspec(grid_n, grid_n, wspace=0.05, hspace=0.2)
        gs_right = gs[1].subgridspec(grid_n, grid_n, wspace=0.05, hspace=0.2)
        
        cmap = plt.get_cmap('tab20')
        
        for idx in range(B):
            row = idx // grid_n
            col = idx % grid_n
            fname = fnames[idx] if fnames is not None else f"{idx}"

            # ================= 绘制左侧：原始图像 =================
            ax_img = fig.add_subplot(gs_left[row, col])
            img = imgs_np[idx].transpose(1, 2, 0)
            # 反归一化
            img = (img * self.img_std + self.img_mean).clip(0, 1)
            ax_img.imshow(img)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title(fname, fontsize=9, pad=3)
            for spine in ax_img.spines.values():
                spine.set_edgecolor('lightgray')
                spine.set_linewidth(1.0)

            # ================= 绘制右侧：合并的掩膜 =================
            ax_mask = fig.add_subplot(gs_right[row, col])
            mask_tensor = masks_np[idx] 
            
            # [-1, 1] 的掩膜恢复到 [0, 1] 强度以便可视化
            fg_masks = mask_tensor[1:] 
            cls_idx = np.argmax(fg_masks, axis=0) + 1
            intensity = (np.max(fg_masks, axis=0) + 1) / 2.0 
            
            rgba = cmap((cls_idx - 1) / 14.0)
            rgb = rgba[..., :3]
            final_vis = rgb * intensity[..., np.newaxis]
            
            ax_mask.imshow(final_vis)
            ax_mask.set_xticks([])
            ax_mask.set_yticks([])
            ax_mask.set_title(fname, fontsize=9, pad=3)
            for spine in ax_mask.spines.values():
                spine.set_edgecolor('lightgray')
                spine.set_linewidth(1.0)

        # # 添加全局大标题区分左右
        # fig.text(0.25, 0.98, "Condition Images", ha='center', fontsize=16, fontweight='bold')
        # fig.text(0.75, 0.98, "Generated/Target Masks", ha='center', fontsize=16, fontweight='bold')

        os.makedirs('./vis_mask', exist_ok=True)
        save_name = f'./vis_mask/paired_vis_epoch{epoch}_step{step}.jpg'
        plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"👀 图像-掩膜 配对可视化已保存至: {save_name}")





        

if __name__ == '__main__':

    # DOTA v1.0 官方规定的 15 个类别
    DOTA_CLASSES = (
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 
        'basketball-court', 'storage-tank',  'soccer-ball-field', 
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    )
    
    # 请替换为您本地实际存在的 DOTA 图像和标签文件夹路径
    img_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/images'
    label_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/1.5' 
    
    cfg = {
        "dataset_cfg": {
            "type": "ImageMaskDataset",
            "class_names": DOTA_CLASSES,
            "img_dir": img_dir,
            "label_dir": label_dir,
            "img_size": [256, 256],       # 网络所需的真实图像输入尺寸
            "mask_size": [32, 32],        # LDM/DiT 在潜空间生成的掩膜尺寸 (例如 256/8)
            "ori_img_size": [1024, 1024], # 原 DOTA 切片物理尺寸
            "img_mean": [0.5, 0.5, 0.5],  
            "img_std": [0.5, 0.5, 0.5]
        },
        "bs": 16, # 建议设小一点(如16)，方便用网格对比查看左右配对结果
        "seed": 42,
        "shuffle": True
    }

    dataset_cfg = cfg["dataset_cfg"]
    seed_everything(cfg["seed"])
    
    # 构建数据集
    train_dataset = ImageMaskDataset(**{k: v for k, v in dataset_cfg.items() if k != "type"})
    
    train_data_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=cfg["bs"], 
        shuffle=cfg["shuffle"], 
        num_workers=4, 
        collate_fn=train_dataset.dataset_collate, 
        worker_init_fn=partial(worker_init_fn, seed=cfg["seed"])
    )
    
    # 测试读取与配对可视化
    for epoch in range(1, 2):
        for step, batch in enumerate(train_data_loader):
            # 获取图像、掩膜和文件名
            batch_imgs, batch_masks, batch_fnames = batch[0], batch[1], batch[2]
            
            print(f"Epoch: {epoch}, Step: {step}")
            print(f"  ➤ 图像 Shape: {batch_imgs.shape}")  # 预期: [16, 3, 256, 256]
            print(f"  ➤ 掩膜 Shape: {batch_masks.shape}") # 预期: [16, 16, 32, 32]
            
            # 每隔 10 个 step 进行一次可视化排查
            if step % 10 == 0:
                # 传入完整的 batch 列表
                train_dataset._vis_GenDataset_merge_batch(epoch, step, batch)
                # break # 如果只想看第一批就取消注释