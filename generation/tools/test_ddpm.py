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






def gen_batch_sample_ddpm(model, bs, log_dir, mean, std):
    model.eval()
    # 图像生成
    samples = model(bs=bs, return_loss=False)
    # 可视化
    generate_images = samples
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))  # Create an 8x8 grid of subplots
    for i, ax in enumerate(axes.flat):
        # [H, W, C]
        gen_img_norm = generate_images[i].transpose((1,2,0))
        gen_img = gen_img_norm * std + mean
        ax.imshow(gen_img) 
        ax.axis("off")  

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"gen_samples1.png"), dpi=300)




def gen_one_sample_ddpm(model, bs, log_dir, mean, std, timsteps=1000, vis_step=10):

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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    log_dir = "./"
    img_size = [128, 128]
    bs = 64

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

    '''DIOR'''
    # load_ckpt = 'log/ddpm_unet_DIOR_train_ddp/2025-10-15-01-28-12_train_ddp/last.pt'
    # dim = 128
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
    #         input_dim=3,
    #         output_dim=3,
    #         # 配置 encoder / decoder 每一层的通道数
    #         layer_dims=[dim*1, dim*1, dim*2, dim*4],
    #     )
    # )

    '''Celeba'''
    # load_ckpt = 'log/ddpm_unet_Celeba_train_ddp/2026-01-22-11-55-31_train_ddp/best_ssim.pt'
    # dim = 32
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
    #         input_dim=3,
    #         output_dim=3,
    #         # 配置 encoder / decoder 每一层的通道数
    #         layer_dims=[dim*1, dim*1, dim*2, dim*2, dim*4, dim*8],
    #     )
    # )

    '''GCC'''
    # load_ckpt = 'log/ddpm_unet_GCC_train_ddp/2026-03-20-18-08-39_train_ddp/last.pt'
    # dim = 128
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
    #         input_dim=3,
    #         output_dim=3,
    #         # 配置 encoder / decoder 每一层的通道数
    #         layer_dims=[dim*1, dim*1, dim*2, dim*4],
    #     )
    # )

    load_ckpt = 'log/ldm_unet_face_train_ddp/2026-03-24-01-39-38_train_ddp/last.pt'
    latent_dim = 16
    dim=128
    model_cfgs = dict(
        type="LDM",
        vae=dict(
            type='HFVAE',
            weight_dir='ckpts/hugging_face/vae-kl-f8-d16',
            latent_dim=latent_dim,
            down_scale=8,
        ),
        img_size=[256,256],
        batch_size=bs,
        load_ckpt=load_ckpt,
        schedule_name="linear_beta_schedule",
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        loss_type='huber',
        denoise_model=dict(
            type="UNet",
            input_dim=latent_dim,
            output_dim=latent_dim,
            # 配置 encoder / decoder 每一层的通道数
            layer_dims=[dim*1, dim*1, dim*2, dim*4],
        )
    )

    # 图像均值 标准差
    mean = np.array([0.5, 0.5, 0.5]) 
    std = np.array([[0.5, 0.5, 0.5]]) 

    model = MODELS.build_from_cfg(model_cfgs).to(device)
    gen_batch_sample_ddpm(model, bs, log_dir, mean, std)
    # gen_one_sample_ddpm(model, bs, log_dir, mean, std)


