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






def gen_batch_sample_ddpm(model, bs, tokenizer_cfg_dir, log_dir, prompt, max_tokens_len=768):
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
    load_ckpt = 'log/cvae_Celeba_train/2025-12-28-01-16-31_train/last.pt'
    # load_ckpt = 'log/cvae_Celeba_train/2025-12-27-23-06-19_train/last.pt'
    tokenizer_cfg_dir = '/mnt/yht/code/HeltonPretrain/llm/tokenizer_configs/minimind2'
    prompt = '是一位男性，肤色属于白种人，从面部轮廓和皱纹来看，他大概处于中年阶段。脸型偏方，下巴线条分明，整体轮廓显得硬朗。他的眼神专注而严肃，嘴唇微抿，似乎正沉浸在演唱或讲话中，没有明显的微笑或其他情绪波动。头发较长，略显凌乱，颜色是深棕色带些灰白，自然垂在额前和耳际。他留着浓密的胡须，胡子覆盖了整个下巴和脸颊，呈络腮胡样式，增添了几分粗犷气质。头上戴着一顶浅灰色牛仔帽，帽檐压得较低，帽身上还点缀着几颗红色小铆钉，为整体造型添了一抹西部风格。他的眼睛虹膜是深褐色，在光线照射下显得沉稳有力。拍摄视角是正面略微偏侧，能清晰看到他的面部表情和帽子细节。背景模糊不清，似乎是户外环境，隐约可见浅色天空或幕布，暗示可能是在一个露天演出或活动现场。人物没有明显化妆痕迹，妆容自然，符合现场表演的真实状态。整体画面传递出一种质朴、坚毅又略带沧桑的氛围。'
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
        max_len=768, 
        dropout=0.0
        ),
        load_ckpt=load_ckpt
    )
    model = MODELS.build_from_cfg(model_cfgs).to(device)
    model.eval()

    gen_batch_sample_ddpm(model, bs, tokenizer_cfg_dir, log_dir, prompt, max_tokens_len=768)




