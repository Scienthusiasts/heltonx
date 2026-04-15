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






def gen_samples_by_class_id(
    model, 
    class_id,                        # 🌟 核心修改：接收指定的类别 ID
    class_names,                     # 必须传入：类别名称列表
    img_mean=[0.485, 0.456, 0.406],  # 图像反归一化均值
    img_std=[0.229, 0.224, 0.225],   # 图像反归一化标准差
    bs=25, 
    log_dir="./logs", 
    name="class_conditioned_gen"
):
    """
    给定一个类别 ID，生成该类别的批量图像。
    完全解耦 Dataset 对象，依靠外部参数独立运行。
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. 校验并获取类别信息
    num_classes = len(class_names)
    if class_id < 0 or class_id >= num_classes:
        print(f"❌ 错误: class_id {class_id} 超出范围 (0 ~ {num_classes-1})")
        return
        
    target_class_name = class_names[class_id]
    
    mean = np.array(img_mean)
    std = np.array(img_std).reshape(1, -1)

    # ==========================================
    # 2. 构建类别条件张量 [bs,]
    # ==========================================
    # 创建一个充满 target class_id 的一维 tensor
    labels = torch.full((bs,), class_id, dtype=torch.long, device=device)

    # ==========================================
    # 3. 执行条件生成 (推理)
    # ==========================================
    batch_data = [None, labels]
    
    with torch.no_grad():
        # 调用 ClassLDM 的采样逻辑
        samples = model(batch_data=batch_data, return_loss=False, bs=bs)
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()

    # ==========================================
    # 4. 可视化生成结果
    # ==========================================
    grid_size = int(np.ceil(np.sqrt(bs)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12), facecolor='white')
    
    # 防止因 bs 较小导致 axes 不是二维数组
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
            
    # 增加主标题以标明当前生成的类别
    fig.suptitle(f"Generated Class: {target_class_name} (ID: {class_id})", fontsize=18, fontweight='bold', y=0.98)
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.93) # 留出主标题的空间
    
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, f"{name}_cls{class_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 类别 [{target_class_name}] 的 {bs} 张样本已生成并保存至: {save_path}")


if __name__ == '__main__':


    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    log_dir = "./"
    img_size = [256, 256]
    bs = 64  
    load_ckpt = 'log/classldm_dit_IN1K_train_ddp/2026-04-10-21-27-33_train_ddp/last.pt'
    latent_dim = 16
    cls_name = [str(i) for i in range(1000)] 

    classes = sorted(cls_name)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    
    # 🌟 核心修正：配置必须与上一轮编写的 ClassLDM 严格对齐
    model_cfgs = dict(
        type="ClassLDM",         # 使用基于类别的 LDM 类名
        vae=dict(
            type='HFVAE',
            weight_dir='ckpts/hugging_face/vae-kl-f8-d16',
            latent_dim=latent_dim,
            down_scale=8,
        ),
        img_size=img_size,
        batch_size=bs,
        num_classes=1000, # 必须传入总类别数，供 CFG 使用
        load_ckpt=load_ckpt,
        schedule_name="linear_beta_schedule",
        timesteps=len(cls_name),
        beta_start=0.0001,
        beta_end=0.02,
        loss_type='huber',
        # CFG configs
        cfg_drop_prob=0.15,  
        cfg_scale=3.0,       # 建议调到 3.0~5.0，增强类别的条件引导
        denoise_model=dict(
            type="DiT",
            in_channels=latent_dim,
            out_channels=latent_dim,
            depth=12, 
            hidden_size=768, 
            patch_size=2, 
            num_heads=12, 
            learn_sigma=False, 
            use_condition=True,
            class_dropout_prob=0.0
        )
    )

    # 图像均值 标准差
    mean = np.array([0.5, 0.5, 0.5]) 
    std = np.array([[0.5, 0.5, 0.5]]) 

    model = MODELS.build_from_cfg(model_cfgs).to(device)
    
    # 假设我们想测试生成 'plane' (索引为 0)
    target_id = 0
    
    # 直接调用，生成指定类别的图像
    gen_samples_by_class_id(
        model=model, 
        class_id=class_to_idx[str(target_id)],  # 传入目标类别 ID
        class_names=classes,
        img_mean=mean,       # 图像反归一化均值
        img_std=std,         # 图像反归一化标准差
        bs=bs, 
        log_dir=log_dir,
        name="test_class_generation"
    )