# coding=utf-8
import os
import torch
from torch import nn
import cv2
from tqdm import tqdm
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np


from generation.datasets.preprocess import Transforms
# 需要import才能注册
from generation import * 
from heltonx.utils.register import MODELS






def gen_batch_sample_ddpm(model, bs, log_dir):

    # 图像均值 标准差
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([[0.229, 0.224, 0.225]]) 

    # model.eval()
    # 图像生成
    samples = model(bs=bs, return_loss=False)

    # 可视化
    generate_images = samples[-1]
    B, C, H, W = generate_images.shape
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))  # Create an 8x8 grid of subplots
    for i, ax in enumerate(axes.flat):
        gen_img_norm = generate_images[i].reshape(C, H, W).transpose((1,2,0))
        # figtest = reverse_transform(torch.from_numpy(generate_image))
        gen_img = gen_img_norm * std + mean
        ax.imshow(gen_img) 
        ax.axis("off")  

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"gen_samples1.png"), dpi=300)




def gen_one_sample_ddpm(model, bs, log_dir, timsteps=1000, vis_step=10, ):

    # 图像均值 标准差
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([[0.229, 0.224, 0.225]]) 

    model.eval()
    # 图像生成
    samples = model(bs=bs, return_loss=False)

    # 可视化
    generate_images = samples
    B, C, H, W = generate_images[0].shape
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))  # Create an 8x8 grid of subplots
    steps = np.linspace(0, timsteps-1, vis_step*vis_step)
    for i, ax in enumerate(axes.flat):
        gen_img_norm = generate_images[round(steps[i])][0].reshape(C, H, W).transpose((1,2,0))
        # figtest = reverse_transform(torch.from_numpy(generate_image))
        gen_img = gen_img_norm * std + mean
        ax.imshow(gen_img) 
        ax.axis("off")  

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"sample_process.png"), dpi=300)




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = "./"
    img_size = [128, 128]
    bs = 25

    '''FlickrBreeds'''
    # load_ckpt = 'log/ddpm_unet_FlickrBreeds_train_ddp/2025-10-07-19-53-15_train_ddp/last.pt'
    # model_cfgs = dict(
    #     type="DDPM",
    #     img_size=img_size,
    #     batch_size=bs,
    #     load_ckpt=load_ckpt,
    #     schedule_name="linear_beta_schedule",
    #     timesteps=1000,
    #     beta_start=0.0001,
    #     beta_end=0.02,
    #     loss_type='huber',
    #     denoise_model=dict(
    #         type="UNet",
    #         dim=img_size[0],
    #         channels=3,
    #         dim_mults=(1, 2, 4,)
    #     )
    # )

    ''''''
    load_ckpt = 'log/ddpm_unet_DIOR_train_ddp/2025-10-15-01-28-12_train_ddp/last.pt'
    dim = 128
    model_cfgs = dict(
        type="DDPM",
        img_size=img_size,
        batch_size=bs,
        load_ckpt=load_ckpt,
        schedule_name="linear_beta_schedule",
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        loss_type='huber',
        denoise_model=dict(
            type="UNet",
            input_dim=3,
            output_dim=3,
            # 配置 encoder / decoder 每一层的通道数
            layer_dims=[dim*1, dim*1, dim*2, dim*4],
        )
    )

    model = MODELS.build_from_cfg(model_cfgs).to(device)
    model.eval()

    gen_batch_sample_ddpm(model, bs, log_dir)
    # gen_one_sample_ddpm(model, bs, log_dir)


