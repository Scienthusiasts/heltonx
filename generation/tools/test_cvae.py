# coding=utf-8
import os
import torch
from torch import nn
import cv2
from tqdm import tqdm
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from generation.datasets.preprocess import Transforms
# 需要import才能注册
from generation import * 
from heltonx.utils.register import MODELS






def gen_batch_sample(model, bs, tokenizer_cfg_dir, log_dir, prompt, max_tokens_len=768):
    '''prompt相关'''
    # 加载训练好的 HuggingFace 格式的 tokenizer，用于把文本转成 token ids
    # tokenizer 内部包含词表 / 特殊 token / 编码规则等元数据
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg_dir)
    # 进行tokenize + 截断
    input_ids = tokenizer(prompt).input_ids
    input_ids = input_ids[:768]
    # padding补齐
    input_ids += [tokenizer.pad_token_id] * (max_tokens_len - len(input_ids))
    prompts_tokens = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).repeat(bs, 1)
    
    '''图像相关'''
    # 图像均值 标准差
    mean = [0.48145466, 0.4578275, 0.40821073]
    std =  [0.26862954, 0.26130258, 0.27577711]

    '''图像生成'''
    samples = model(bs=bs, c_tokens=prompts_tokens, return_loss=False)

    '''可视化'''
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
    plt.savefig(os.path.join(log_dir, f"gen_samples_cvae.png"), dpi=200)






if __name__ == '__main__':
    load_ckpt = 'log/cvae_Celeba_train_ddp/2026-01-21-01-45-07_train_ddp/last.pt'
    # load_ckpt = 'log/cvae_Celeba_train/2025-12-27-23-06-19_train/last.pt'
    tokenizer_cfg_dir = '/mnt/yht/code/HeltonPretrain/llm/tokenizer_configs/minimind2'
    prompt = '性别: [男性] | 族裔: [黑种人] | 年龄段: [中年] | 脸型: [瓜子脸] | 表情: [严肃] | 头发长短: [短发] | 发型: (寸头) | 发色: [棕色] | 穿戴配饰: (无) | 虹膜颜色: [棕色] | 胡须样式: [短胡茬] | 妆容: [未化妆] | 拍摄视角: [正脸] | 人物背景: [深色背景]'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = "./"
    img_size = [256, 256]
    bs = 64
    dim = 64
    '''模型配置参数'''
    model_cfgs = dict(
        type='CVAE',   
        img_size=img_size,
        input_dim=3,
        layer_dims=[dim, dim*2, dim*4, dim*6, dim*8, dim],  
        kld_weight=0.0002,
        latent_dim=dim*4,
        # 条件相关参数:
        condition_emb_dim=dim*4,
        vocab_size=6400,
        z_drop_prob=0.5, 
        z_drop_ratio=0.75,
        c_proj_model = dict(
        type='LightBERT',   
        emb_dim=dim*4, 
        n_layers=4, 
        heads=8, 
        max_len=192, 
        dropout=0.0
        ),
        load_ckpt=load_ckpt
    )
    model = MODELS.build_from_cfg(model_cfgs).to(device)
    model.eval()

    gen_batch_sample(model, bs, tokenizer_cfg_dir, log_dir, prompt, max_tokens_len=192)




