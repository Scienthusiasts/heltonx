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






def gen_batch_sample(model, bs, log_dir):

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
    fig, axes = plt.subplots(size, size, figsize=(10, 10))  
    for i, ax in enumerate(axes.flat):
        gen_img_norm = generate_images[i].reshape(C, H, W).transpose((1,2,0))
        # figtest = reverse_transform(torch.from_numpy(generate_image))
        gen_img = gen_img_norm * std + mean
        ax.imshow(gen_img) 
        ax.axis("off")  

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"gen_samples_vae.png"), dpi=200)




def reconstruct_one_sample(device, model, img_path, img_size, log_dir):
    # 图像均值 标准差
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([[0.229, 0.224, 0.225]]) 
    transform = Transforms(img_size=img_size)
    image = np.array(Image.open(img_path).convert('RGB'))
    tensor_img = torch.tensor(transform.transform(image=image)['image'])
    tensor_img = tensor_img.permute(2,0,1).unsqueeze(0).to(device)

    # 图像生成
    rec_img = model.reconstruct(img=tensor_img).squeeze(0).permute(1,2,0)
    rec_img = ((rec_img.cpu().numpy() * std + mean) * 255).astype(np.uint8)
    # 可视化
    fig, axes = plt.subplots(1,2, figsize=(10, 5))  
    axes[0].imshow(image) 
    axes[0].axis("off")  
    axes[1].imshow(rec_img) 
    axes[1].axis("off")  

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"reconstruct_vae.png"), dpi=200)




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


    # 图像生成
    # gen_batch_sample(model, bs, log_dir)

    # 图像复原
    img_path = r'/mnt/yht/data/celeba_256/train/14110.jpg'
    reconstruct_one_sample(device, model, img_path, img_size, log_dir)




