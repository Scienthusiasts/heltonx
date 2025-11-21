# coding=utf-8
import os
import json
import torch
from torch import nn
import cv2
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np

from pretrain.utils.metrics import *
from heltonx.utils.utils import natural_key
from pretrain.datasets.preprocess import Transforms
# 需要import才能注册
from pretrain import * 
from heltonx.utils.register import MODELS




def fuse_image_with_heatmap(image, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """将单通道热力图与原图融合，返回融合后的彩色图像。
        Args:
            image:    np.ndarray 原图 (H, W, 3), dtype=np.uint8
            heatmap:  np.ndarray 单通道热力图 (h, w), dtype=np.uint8
            alpha:    float 热力图权重，取值 [0,1]
            colormap: int OpenCV colormap类型, 如 cv2.COLORMAP_JET, cv2.COLORMAP_HOT
        Return:
            fused_imgs: np.ndarray 融合后的彩色图像 (H, W, 3), dtype=np.uint8
    """
    # 转伪彩色
    heatmap_color = cv2.applyColorMap(255 - heatmap, colormap)
    # heatmap缩放到原图尺寸
    heatmap_resized = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
    # 加权融合
    fused_imgs = cv2.addWeighted(image, 1 - alpha, heatmap_resized, alpha, 0)
    return fused_imgs





def draw_fig(image, pred_score, pred_class, pred_conf, cat_names, log_dir, top_k=10):
    """plt可视化
    """
    # 排序类别（按置信度从大到小）
    sorted_idx = np.argsort(pred_score)[::-1]
    sorted_scores = pred_score[sorted_idx][:top_k]
    sorted_names = [cat_names[i] for i in sorted_idx[:top_k]]

    # 颜色设置：最大概率绿色，其余橙色
    colors = ['green'] + ['orange'] * (len(sorted_names)-1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # 左边：原图 + 预测文字
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image")
    axes[0].text(
        5, 20, f"Pred: {pred_class} ({pred_conf:.4f})",
        fontsize=12, color="white", weight="bold",
        bbox=dict(facecolor="black", alpha=0.6, pad=3)
    )
    # 右边：横向条形图
    y_pos = np.arange(len(sorted_names))
    axes[1].barh(y_pos, sorted_scores, color=colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(sorted_names, fontsize=10)
    axes[1].invert_yaxis()  # 让最高概率在最上面
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel("Confidence")
    axes[1].set_title(f"Top {top_k} Class Probabilities")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/infer_result.png", dpi=300)



def infer_single_img(device, model, img_size, img_path, cat_names, log_dir, top_k=10):
    '''推理一张图片(经典图像分类任务)
    '''
    # 图像增强(预处理)实例
    transform = Transforms(img_size=img_size)
    model.eval()
    # 读取图像, 并进行预处理
    image = np.array(Image.open(img_path).convert('RGB'))
    tensor_img = torch.tensor(transform.valid_transform(image=image)['image'])
    tensor_img = tensor_img.permute(2,0,1).unsqueeze(0).to(device)
    # 推理
    with torch.no_grad():
        pred_logits = model(tensor_img)
        pred_label = torch.argmax(pred_logits, dim=1).cpu().numpy().item() 
        pred_score = pred_logits.softmax(dim=-1).cpu().numpy().squeeze()
    pred_class = cat_names[pred_label]
    pred_conf = float(pred_score[pred_label]) 
    # 可视化
    draw_fig(image, pred_score, pred_class, pred_conf, cat_names, log_dir, top_k)





def infer_single_img_protonet(device, model, img_size, img_path, cat_names, log_dir, top_k=10):
    '''推理一张图片(经典图像分类任务)
    '''
    # 图像均值 标准差
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([[0.229, 0.224, 0.225]]) 

    # 图像增强(预处理)实例
    transform = Transforms(img_size=img_size)
    model.eval()
    # 读取图像, 并进行预处理
    image = np.array(Image.open(img_path).convert('RGB'))
    tensor_img = torch.tensor(transform.valid_transform(image=image)['image'])
    resize_img = ((tensor_img.numpy() * std + mean) * 255).astype(np.uint8)
    tensor_img = tensor_img.permute(2,0,1).unsqueeze(0).to(device)
    # 推理
    with torch.no_grad():
        pred_logits, heatmap = model.forward_with_protoheatmap(tensor_img)
        print(heatmap.shape)
        pred_label = torch.argmax(pred_logits.softmax(dim=-1), dim=1).cpu().numpy().item() 
        pred_score = pred_logits.softmax(dim=-1).cpu().numpy().squeeze()
    pred_class = cat_names[pred_label]
    pred_conf = float(pred_score[pred_label]) 
    # 预测的类别对应的prototype特征图
    pred_label_heatmap = (heatmap.squeeze(0)[pred_label].cpu().numpy() * 255).astype(np.uint8)
    resize_img = fuse_image_with_heatmap(resize_img, pred_label_heatmap, alpha=0.6)
    # 可视化
    draw_fig(resize_img, pred_score, pred_class, pred_conf, cat_names, log_dir, top_k)





def infer_single_img_fgclip(device, model, img_size, img_path, captions, log_dir):
    '''推理一张图片(经典图像分类任务)
    '''
    # 图像均值 标准差
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([[0.229, 0.224, 0.225]]) 

    # 图像增强(预处理)实例
    transform = Transforms(img_size=img_size)
    model.eval()
    # 读取图像, 并进行预处理
    image = np.array(Image.open(img_path).convert('RGB'))
    tensor_img = torch.tensor(transform.valid_transform(image=image)['image'])
    resize_img = ((tensor_img.numpy() * std + mean) * 255).astype(np.uint8)
    tensor_img = tensor_img.permute(2,0,1).unsqueeze(0).to(device)
    # 推理
    heatmap, patch_size = model.forward_dense_heatmap(device, tensor_img, captions)
    heatmap = heatmap.reshape(patch_size, patch_size).cpu().numpy()
    # 可视化
    plt.imshow(heatmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{log_dir}/infer_result.png", dpi=200)










if __name__ == '__main__':
    def f1():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_dir = "./"
        model_cfgs = {
            "type": "ProtoNet",
            "load_ckpt": r'/mnt/yht/code/HeltonPretrain/log/protonet_dinov3vits_train/2025-09-25-00-34-55_train/last.pt',
            "backbone":{
                "type": "TIMMBackbone",
                "model_name": "vit_small_patch16_dinov3.lvd1689m",
                "pretrained": r'/mnt/yht/code/HeltonPretrain/ckpts/vit_small_patch16_dinov3.lvd1689m.pt',
                "out_layers": [11],
                "froze_backbone": True,
                "load_ckpt": None
            },
            "head":{
                "type": "ProtoHead",
                "layers_dim":[384, 256, 256], 
                "nc":37,
                "cls_loss": {
                    "type": "MultiClassBCELoss"
                }
            }
        }
        model = MODELS.build_from_cfg(model_cfgs).to(device)
        model.eval()
        img_dir = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/valid'
        all_entries = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)) and not d.startswith('.')]
        cat_names = sorted(all_entries, key=natural_key)
        img_size = [256, 256] # [512, 512]
        img_path = rf"{img_dir}/Maine_Coon/Maine_Coon_67.jpg"
        infer_single_img_protonet(device, model, img_size, img_path, cat_names, log_dir)

    def f2():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_size = [224, 224]
        pretrain_path = r'/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/models--qihoo360--fg-clip-base/snapshots/f30d2b82ba939fd54ca732426f99f4d6c3c92387'
        # large
        # img_size = [336, 336]
        # pretrain_path = r'/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/models--qihoo360--fg-clip-large/snapshots/19c2df7667052518ade09341652562b89b1332da'
        log_dir = "./"
        model_cfgs = {
            "type": "Qihoo360FGCLIP",
            "pretrain_path": pretrain_path
        }
        model = MODELS.build_from_cfg(model_cfgs).to(device)
        model.eval()

        img_dir = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/valid'
        img_path = rf"{img_dir}/Ragdoll/Ragdoll_259.jpg"
        captions = ["the deep blue eyes of the cat in the picture."]
        infer_single_img_fgclip(device, model, img_size, img_path, captions, log_dir)


    f2()