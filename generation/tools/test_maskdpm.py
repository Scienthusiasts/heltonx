# coding=utf-8
import os
import torch
from torch import nn
import cv2
from tqdm import tqdm
import random
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np
from generation.datasets.preprocess import Transforms
# 需要import才能注册
from generation import * 
from heltonx.utils.register import MODELS



def gen_samples_by_one_mask(
    model, 
    txt_path, 
    class_names,                     # 必须传入：类别名称列表
    mask_size,                       # 必须传入：生成的 Mask 尺寸 (例如 [32, 32])
    ori_img_size=(1024, 1024),       # 原 DOTA 图像切片尺寸
    img_mean=[0.485, 0.456, 0.406],  # 图像反归一化均值
    img_std=[0.229, 0.224, 0.225],   # 图像反归一化标准差
    bs=25, 
    log_dir="./logs", 
    name="one_mask_gen",
    use_random_rotate=False,         # 是否开启随机旋转开关
    angle_range=(-180, 180)          # 旋转角度范围 (默认 -180° 到 180°)
):
    """
    给定一个 txt 标注文件，为 Batch 中的每一个样本单独采样旋转角度并生成条件 Mask，
    据此生成一批图像。完全解耦 Dataset 对象，依靠外部参数独立运行。
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. 内部构建类别索引和尺寸变量
    num_classes = len(class_names)
    class2idx = {cls_name: idx + 1 for idx, cls_name in enumerate(class_names)}
    
    mask_H, mask_W = mask_size[0], mask_size[1]
    ori_H, ori_W = ori_img_size[0], ori_img_size[1]
    
    mean = np.array(img_mean)
    std = np.array(img_std).reshape(1, -1)

    # ==========================================
    # 2. 从 TXT 读取并绘制基础的、未旋转的前景 Mask
    # ==========================================
    if not os.path.exists(txt_path):
        print(f"❌ 错误: 找不到标注文件 {txt_path}")
        return

    obj_masks_ori = np.zeros((num_classes, ori_H, ori_W), dtype=np.uint8)

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9: continue
        cls_name = parts[8]
        if cls_name not in class2idx: continue
        
        channel_idx = class2idx[cls_name] - 1 
        try:
            pts = [float(p) for p in parts[:8]]
        except ValueError: continue 
        
        pts_x = np.array(pts[0::2])
        pts_y = np.array(pts[1::2])
        poly_pts = np.stack([pts_x, pts_y], axis=1).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(obj_masks_ori[channel_idx], [poly_pts], color=255)

    # 缩放到潜空间尺寸并保留软边缘 (这是基础的前景模板，只需计算一次)
    base_obj_masks_resized = np.zeros((num_classes, mask_H, mask_W), dtype=np.float32)
    for i in range(num_classes):
        if np.any(obj_masks_ori[i]): 
            base_obj_masks_resized[i] = cv2.resize(
                obj_masks_ori[i], (mask_W, mask_H), interpolation=cv2.INTER_AREA
            )

    # ==========================================
    # 3. 为 Batch 里的每个样本独立旋转并构建 16 通道 Mask
    # ==========================================
    batch_masks = []
    center = (mask_W / 2.0, mask_H / 2.0)

    for b in range(bs):
        # 深度拷贝一份基础模板，防止相互污染
        curr_obj_masks = base_obj_masks_resized.copy()
        
        # 🌟 独立旋转逻辑：为这一个样本单独抽取角度
        if use_random_rotate:
            angle = random.uniform(angle_range[0], angle_range[1])
            M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            
            for i in range(num_classes):
                if np.any(curr_obj_masks[i]):
                    curr_obj_masks[i] = cv2.warpAffine(
                        curr_obj_masks[i], M, (mask_W, mask_H), 
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )

        # 计算这一张旋转后的 Mask 的背景
        fg_max = np.max(curr_obj_masks, axis=0)
        bg_mask = np.expand_dims(255.0 - fg_max, axis=0)
        
        # 拼接并归一化到 [-1, 1]
        one_mask = np.concatenate([bg_mask, curr_obj_masks], axis=0)
        one_mask_tensor = torch.from_numpy(one_mask).type(torch.FloatTensor) / 255. * 2. - 1.
        batch_masks.append(one_mask_tensor)

    # 堆叠形成最终的 Batch: [bs, 16, h, w]
    mask_conditions = torch.stack(batch_masks, dim=0).to(device)

    if use_random_rotate:
        print(f"🔄 随机旋转已开启，已为 {bs} 个样本分配独立角度({angle_range[0]}°~{angle_range[1]}°)")

    # ==========================================
    # 4. 执行条件生成 (推理)
    # ==========================================
    batch_data = [None, mask_conditions]
    
    with torch.no_grad():
        samples = model(batch_data=batch_data, return_loss=False, bs=bs)
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()

    # ==========================================
    # 5. 可视化生成结果
    # ==========================================
    grid_size = int(np.ceil(np.sqrt(bs)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12), facecolor='white')
    
    if bs == 1:
        axes = np.array([axes])
        
    for i, ax in enumerate(axes.flat):
        if i < bs:
            gen_img_norm = samples[i].transpose((1, 2, 0))
            gen_img = (gen_img_norm * std + mean).clip(0, 1)
            
            ax.imshow(gen_img)
            ax.axis("off")
        else:
            ax.axis("off")
            
    plt.tight_layout()
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 独立旋转的 {bs} 张样本已生成并保存至: {save_path}")





if __name__ == '__main__':

    # 假设这是您主文件里的常量
    DOTA_CLASSES = (
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 
        'basketball-court', 'storage-tank',  'soccer-ball-field', 
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log_dir = "./"
    img_size = [256, 256]
    bs = 64
    load_ckpt = 'log/controlldm_dit_DOTA_train_ddp/2026-04-04-00-36-25_train_ddp/last.pt'
    latent_dim = 16
    dim = 128

    model_cfgs = dict(
        type="ControlLDM",
        vae=dict(
            type='HFVAE',
            weight_dir='ckpts/hugging_face/vae-kl-f8-d16',
            latent_dim=latent_dim,
            down_scale=8,
        ),
        img_size=img_size,
        batch_size=bs,
        load_ckpt=load_ckpt,
        schedule_name="linear_beta_schedule",
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        loss_type='huber',
        # CFG configs
        cfg_drop_prob=0.15,  # 训练时丢弃 Mask 条件的概率 (CFG Dropout)
        cfg_scale=2,       # 推理时条件引导的强度 (CFG Scale)
        denoise_model=dict(
            # type="UNet",
            # input_dim=latent_dim + len(DOTA_CLASSES)+1,
            # output_dim=latent_dim,
            # # 配置 encoder / decoder 每一层的通道数
            # layer_dims=[dim*1, dim*1, dim*2, dim*4],

            type="DiT",
            in_channels=latent_dim + len(DOTA_CLASSES) + 1, 
            out_channels=latent_dim,
            depth=12, 
            hidden_size=768, 
            patch_size=2, 
            num_heads=12, 
            learn_sigma=False, 
            use_condition=False,
        )
    )

    # 图像均值 标准差
    mean = np.array([0.5, 0.5, 0.5]) 
    std = np.array([[0.5, 0.5, 0.5]]) 

    model = MODELS.build_from_cfg(model_cfgs).to(device)
    # 直接调用，不再依赖 runner 或 dataset
    gen_samples_by_one_mask(
        model=model, 
        txt_path="/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/1.5/trainval/annfiles/P0194__1024__0___0.txt", 
        class_names=DOTA_CLASSES,
        mask_size=[32, 32],      # 潜空间尺寸 (通常是原图 256 / 下采样 8 = 32)
        img_mean=mean,  # 图像反归一化均值
        img_std=std,   # 图像反归一化标准差
        bs=bs, 
        log_dir=log_dir,
        name="test_mask_generation",
        use_random_rotate=True,         
        angle_range=(-60, 60)          
    )


