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
    mean = [0.48145466, 0.4578275, 0.40821073]
    std =  [0.26862954, 0.26130258, 0.27577711]

    # model.eval()
    # 图像生成
    samples = model(bs=bs, return_loss=False)
    # 可视化
    generate_images = samples
    B, C, H, W = generate_images.shape
    size = int(bs**0.5)
    fig, axes = plt.subplots(size, size, figsize=(10, 10))  # Create an 8x8 grid of subplots
    for i, ax in enumerate(axes.flat):
        gen_img_norm = generate_images[i].reshape(C, H, W).transpose((1,2,0))
        # figtest = reverse_transform(torch.from_numpy(generate_image))
        gen_img = gen_img_norm * std + mean
        ax.imshow(gen_img) 
        ax.axis("off")  

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"gen_samples_vae.png"), dpi=200)






if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = "./"
    img_size = [256, 256]
    bs = 64
    dim = 64
    '''模型配置参数'''
    model_cfgs = dict(
        type='VAE',   
        input_dim=3,
        layer_dims=[dim, dim*2, dim*4, dim*6, dim*8, dim],  
        latent_dim=dim*4,
        img_size=img_size,
        kld_weight=0.0002,
        load_ckpt = 'log/vae_Celeba_train/2025-12-25-00-34-32_train/epoch_995.pt',
    )
    model = MODELS.build_from_cfg(model_cfgs).to(device)
    model.eval()

    gen_batch_sample_ddpm(model, bs, log_dir)




