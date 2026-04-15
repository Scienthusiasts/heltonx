import numpy as np
import torch
import torch.utils.data.dataset as data
from torch.utils.data import DataLoader
from functools import partial
import matplotlib.pyplot as plt
import os
import cv2  # 新增 cv2 用于多边形绘制
import torch.distributed as dist
from heltonx.utils.register import DATASETS
from heltonx.utils.utils import seed_everything, worker_init_fn
from generation.datasets.preprocess import Transforms




@DATASETS.register
class DOTAOBBMaskDataset(data.Dataset):      
    def __init__(self, label_dir, class_names, img_size, ori_img_size=(1024, 1024)):    
        self.label_dir = label_dir
        self.img_size = img_size
        self.ori_img_size = ori_img_size
        
        self.classes = class_names
        self.num_classes = len(self.classes)
        self.class2idx = {cls_name: idx + 1 for idx, cls_name in enumerate(self.classes)}
        self.transform = Transforms(img_size)
        
        # 1. 递归获取所有 txt 文件
        all_txt_paths = [
            os.path.join(root, fname)
            for root, _, files in os.walk(self.label_dir)
            for fname in files
            if fname.lower().endswith('.txt')
        ]
        
        # 2. 核心修改：过滤掉空的、或不包含任何有效类别目标的标注文件
        self.txt_path_list = []
        for txt_path in all_txt_paths:
            if self._has_valid_object(txt_path):
                self.txt_path_list.append(txt_path)
        
        self.dataSize = len(self.txt_path_list)
        
        # 打印过滤前后的数据集信息
        use_ddp = dist.is_initialized()
        if not use_ddp or (use_ddp and dist.get_rank() == 0):
            print(f'📄 dataset info: 原始标签文件数:{len(all_txt_paths)}, 过滤空标注后有效文件数:{self.dataSize}')


    def _has_valid_object(self, txt_path):
        """快速探测文件，只要存在至少一行合法目标即视为有效文件"""
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # DOTA 格式至少包含 9 个参数 (8 个坐标 + 1 个类别)
                    if len(parts) >= 9:
                        cls_name = parts[8]
                        if cls_name in self.class2idx:
                            return True # 发现有效目标，立即返回
        except Exception:
            pass
        return False

    

    def __getitem__(self, item):  
        txt_path = self.txt_path_list[item]
        # 提取不包含后缀的文件名
        fname = os.path.splitext(os.path.basename(txt_path))[0]
        
        out_H, out_W = self.img_size
        ori_H, ori_W = self.ori_img_size

        # 在原图尺寸上绘制掩膜，以保留绝对的几何精度
        obj_masks_ori = np.zeros((self.num_classes, ori_H, ori_W), dtype=np.uint8)

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls_name = parts[8]
            if cls_name not in self.class2idx:
                continue
                
            channel_idx = self.class2idx[cls_name] - 1 
            try:
                pts = [float(p) for p in parts[:8]]
            except ValueError:
                continue 
                
            # 直接使用原始坐标，不进行缩放
            pts_x = np.array(pts[0::2])
            pts_y = np.array(pts[1::2])
            poly_pts = np.stack([pts_x, pts_y], axis=1).astype(np.int32)
            poly_pts = poly_pts.reshape((-1, 1, 2))
            
            # 在高分辨率画布上绘制硬边缘
            cv2.fillPoly(obj_masks_ori[channel_idx], [poly_pts], color=255)

        # 使用 INTER_AREA 进行下采样，生成极度精确的“软边缘”中间值
        obj_masks_resized = np.zeros((self.num_classes, out_H, out_W), dtype=np.float32)
        for i in range(self.num_classes):
            if np.any(obj_masks_ori[i]): # 只有当前通道有目标时才计算，节省算力
                obj_masks_resized[i] = cv2.resize(
                    obj_masks_ori[i], (out_W, out_H), interpolation=cv2.INTER_AREA
                )

        # 计算背景通道：利用 255 减去所有前景通道的最大值
        fg_max = np.max(obj_masks_resized, axis=0)
        bg_mask = 255.0 - fg_max
        bg_mask = np.expand_dims(bg_mask, axis=0)

        # 拼接：背景(1) + 前景(15) = 16 通道
        masks = np.concatenate([bg_mask, obj_masks_resized], axis=0)
        # 圖像增強
        masks = masks.transpose(1,2,0) # [H, W, C]
        trans = self.transform.mask_transform(image=masks)          
        final_masks = trans['image'].transpose(2,0,1)

        # 返回 Mask 和 对应的文件名
        return final_masks, fname
    

    def __len__(self):
        return self.dataSize


    def dataset_collate(self, batch):
        '''将一批数据组合，拆分 Mask 和 文件名
        '''
        masks = []
        fnames = []
        for mask, fname in batch:
            masks.append(mask)
            fnames.append(fname)
            
        masks = np.stack(masks, axis=0) 
        # [0, 255] -> [-1, 1]
        masks_tensor = torch.from_numpy(masks).type(torch.FloatTensor) / 255. * 2. - 1.

        return [masks_tensor, fnames]
    



    def _vis_GenDataset_merge_batch(self, epoch, step, batch_masks, fnames=None):
        '''使用 Alpha 融合可视化带有“软边缘”的掩膜，并显示文件名，增加清晰的图像边界
        '''
        imgs = batch_masks.numpy()
        B = imgs.shape[0]

        # 1. 显式设置整个大画板的背景色为白色
        fig = plt.figure(figsize=(12, 12), facecolor='white')
        grid_size = int(np.ceil(np.sqrt(B)))
        cmap = plt.get_cmap('tab20')
       
        for idx in range(B):
            mask_tensor = imgs[idx] # [16, H, W]
            # 提取前景掩膜 (忽略索引0的背景)  [15, H, W]
            fg_masks = mask_tensor[1:] 

            # 获取每个像素所属的类别索引 (1~15)
            cls_idx = np.argmax(fg_masks, axis=0) + 1
            # 获取该像素的掩膜强度 (0.0 ~ 1.0)  
            intensity = (np.max(fg_masks, axis=0) + 1) / 2.

            # 映射到 RGB 颜色，并乘以强度 (无目标处变为纯黑)
            rgba = cmap((cls_idx - 1) / 14.0)
            rgb = rgba[..., :3]
            final_vis = rgb * intensity[..., np.newaxis]
            ax = plt.subplot(grid_size, grid_size, idx + 1)
            ax.imshow(final_vis)

            # 隐藏坐标刻度和文字，但不使用 axis("off") 这样就能保留物理边框
            ax.set_xticks([])
            ax.set_yticks([])

            # 如果传入了 fnames，则设置标题为文件名
            if fnames is not None and idx < len(fnames):
                ax.set_title(fnames[idx], fontsize=9, pad=5)

        # 稍微调大 wspace 和 hspace，利用白色的 Figure 背景形成物理上的“白边”隔离
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.05, hspace=0.05)
        # 确保保存目录存在
        os.makedirs('./vis_mask', exist_ok=True)
        save_name = f'./vis_mask/mask_vis_epoch{epoch}_step{step}.jpg'
        # 保存时强制指定背景为白色 (facecolor='white')
        plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"👀 软边缘掩膜可视化已保存至: {save_name}")






    def _vis_GenDataset_batch(self, epoch, step, batch_masks, fnames=None):
        '''将每个 Batch 的掩膜按列（图像）和行（类别）展平可视化，空类别不显示，采用灰度色调
        '''
        imgs = batch_masks.numpy() 
        B = imgs.shape[0]
        num_classes = self.num_classes # 15 个前景类别
        
        # 动态计算大画板尺寸：每张子图分配约 2.5x2.5 英寸的空间
        fig, axes = plt.subplots(nrows=num_classes, ncols=B, figsize=(B * 2.5, num_classes * 2.5), facecolor='white')
        
        # 保证 axes 是二维数组 (应对极少数情况下 B=1 导致 axes 降维的问题)
        if B == 1 and num_classes == 1:
            axes = np.array([[axes]])
        elif B == 1:
            axes = axes[:, None]
        elif num_classes == 1:
            axes = axes[None, :]

        for col in range(B):
            mask_tensor = imgs[col] # [16, H, W]
            fname = fnames[col] if (fnames is not None and col < len(fnames)) else f"Image_{col}"
            
            for row in range(num_classes):
                ax = axes[row, col]
                cls_idx = row + 1 # 忽略通道 0 的背景，1~15 代表具体类别
                mask = mask_tensor[cls_idx] # [H, W]
                
                # 隐藏默认的坐标刻度和数字
                ax.set_xticks([])
                ax.set_yticks([])
                
                # 检查该类别下是否有目标 (最大值大于 0 即可)
                if np.max(mask) == 0:
                    # 【核心修改】：没有任何目标，隐藏四周的边框，使其在画板上完全“隐形”
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                else:
                    # 【核心修改】：有目标，使用 'gray' 绘制黑白灰色调 (0 为纯黑，255 为纯白)
                    ax.imshow(mask, cmap='gray', vmin=0, vmax=255)
                    # 保留边框
                    for spine in ax.spines.values():
                        spine.set_edgecolor('lightgray')
                        spine.set_linewidth(1.0)
                        spine.set_visible(True)
                
                # 第一行：在最顶部的子图上方显示文件名 (即使该子图被隐藏，标题依然会悬浮保留)
                if row == 0:
                    ax.set_title(fname, fontsize=12, pad=15)
                    
                # 第一列：在最左侧的子图旁边显示类别名称
                if col == 0:
                    cls_name = self.classes[row]
                    # rotation=0 让文字水平显示，便于阅读
                    ax.set_ylabel(cls_name, fontsize=12, rotation=90, labelpad=20, ha='right', va='center')
                    
        # 调整子图间距，为文字留出空间
        plt.subplots_adjust(left=0.15, bottom=0.02, right=0.98, top=0.95, wspace=0.1, hspace=0.1)
        
        # 确保保存目录存在
        os.makedirs('./vis_mask', exist_ok=True)
        save_name = f'./vis_mask/mask_vis_epoch{epoch}_step{step}.jpg'
        
        # 降低一点 dpi 防止由于子图过多(15 x B)导致内存溢出
        plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"👀 按行列展开的灰度掩膜可视化已保存至: {save_name}")








if __name__ == '__main__':

    # DOTA v1.0 官方规定的 15 个类别
    DOTA_CLASSES = (
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 
        'basketball-court', 'storage-tank',  'soccer-ball-field', 
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    )
    # 请替换为您本地实际存在的 DOTA labelTxt 文件夹路径
    label_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/1.5' 
    
    cfg = {
        "dataset_cfg": {
            "type": "DOTAOBBMaskDataset",
            "class_names": DOTA_CLASSES,
            "label_dir": label_dir,
            "img_size": [32, 32],        # 网络所需掩膜尺寸
            "ori_img_size": [1024, 1024] # 原 DOTA 切片为 1024
        },
        "bs": 64,
        "seed": 42,
        "shuffle": True
    }

    dataset_cfg = cfg["dataset_cfg"]
    seed_everything(cfg["seed"])
    
    # 构建数据集
    train_dataset = DOTAOBBMaskDataset(**{k: v for k, v in dataset_cfg.items() if k != "type"})
    
    train_data_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=cfg["bs"], 
        shuffle=cfg["shuffle"], 
        num_workers=4, 
        collate_fn=train_dataset.dataset_collate, 
        worker_init_fn=partial(worker_init_fn, seed=cfg["seed"])
    )
    
    # 测试读取与可视化
    for epoch in range(1, 2):
        for step, batch in enumerate(train_data_loader):
            batch_masks, batch_fnames = batch[0], batch[1]
            print(f"Epoch: {epoch}, Step: {step}, Batch Shape: {batch_masks.shape}")
            # 预期形状: [16, 16, 256, 256] 
            
            if step % 10==0:
                # train_dataset._vis_GenDataset_batch(epoch, step, batch_masks)
                train_dataset._vis_GenDataset_merge_batch(epoch, step, batch_masks)
                # break