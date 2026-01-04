import torch.nn as nn
import timm
import math
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from heltonx.utils.ckpts_utils import *
from pretrain.datasets.preprocess import Transforms
# 注册机制
from heltonx.utils.register import MODELS



@MODELS.register
class DINOv3(nn.Module):
    """Learning Transferable Visual Models From Natural Language Supervision(CLIP): https://arxiv.org/abs/2103.00020
    """
    def __init__(self, weight_dir):
        """初始化
            Args:
                img_size:      输入图像尺寸
                pretrain_path: CLIP的权重路径
        """
        super(DINOv3, self).__init__()
        self.model = AutoModel.from_pretrained(weight_dir)


    def forward(self, x, type='image', *args, **kwargs):
        '''前向, 调用fgclip图像和文本编码器(API接口, 外部调用此方法)
        '''
        if type == 'image':
            embs = self._forward_img(x)
        if type == 'image_dense':
            embs = self._forward_img_dense(x)
        return embs


    def _forward_img(self, x):
        '''前向
            Args:
                x: [B, 3, H, W]
            Returns:
                img_embs: [B, dim=1280]
        '''
        with torch.no_grad():
            embeddings = self.dinov3(x) 
            pooled_output = embeddings.pooler_output
        return pooled_output


    def _forward_img_dense(self, x):
        '''前向
            Args:
                x: [B, 3, H, W]
            Returns:
                feature_map: [B, dim=1280, H, W]
        '''
        B, C, H, W = x.shape
        with torch.no_grad():
            embeddings = self.model(x)
            # print(embeddings)
            # 从索引5开始是为了忽略之前的special_tokens
            x = embeddings.last_hidden_state
            special_tokens, feature_map = self.split_vit_output(x, H//16, W//16)
        return feature_map


        
    def split_vit_output(self, x, h=16, w=16, num_extra_tokens=5):
        """
        将ViT输出 (BS, h*w+5, 1280) 拆分成 特殊tokens + 特征图
        参数:
            x: torch.Tensor, 形状 (BS, h*w+5, 1280)
            h, w: patch 网格大小 (默认16x16)
            num_extra_tokens: 特殊tokens数量 

        返回:
            special_tokens: (BS, num_extra_tokens, C)
            feature_map: (BS, C, H, W)
        """
        B, N, C = x.shape
        assert N == h * w + num_extra_tokens, f"输入序列长度 {N} 不等于 {h*w + num_extra_tokens}"
        # 前 num_extra_tokens 是 cls token(1) + register tokens(4)
        special_tokens = x[:, :num_extra_tokens, :]  # (BS, 65, 1280)
        # 后面是 patch tokens，reshape 成 feature map
        patch_tokens = x[:, num_extra_tokens:, :]  # (BS, 256, 1280)
        feature_map = patch_tokens.transpose(1, 2).reshape(B, C, h, w)  # (BS, 1280, 16, 16)
        print(feature_map.shape)
        return special_tokens, feature_map



    def cosine_similarity_map(self, x, row: int, col: int) -> torch.Tensor:
        """
        给定特征图和一个像素位置，计算该像素 embedding 与所有位置 embedding 的余弦相似度。
        参数:
            fmap: torch.Tensor, 形状 [B, C, H, W]，特征图
            row: int, 行索引 (0 <= row < H)
            col: int, 列索引 (0 <= col < W)

        返回:
            sim_map: torch.Tensor, 形状 [B, H, W]，相似度特征图
        """
        B, C, H, W = x.shape
        assert 0 <= row < H and 0 <= col < W, f"row={row}, col={col} 超出范围 H={H}, W={W}"
        # 取出目标位置的 embedding，形状 [B, C]
        target = x[:, :, row, col]  
        sim = F.cosine_similarity(target.view(B, C, 1), x.view(B, C, -1)).view(B, H, W)
        return sim





# for test only:
if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np


    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = [1024,1024]
    # 加载模型
    # model = DINOv3(weight_dir=r"ckpts\hugging_face\vit_large_patch16_dinov3_sat493m").to(device)
    # model = DINOv3(weight_dir=r"ckpts\hugging_face\vit_small_patch16_dinov3_lvd1689m").to(device)
    # model = DINOv3(weight_dir=r"ckpts\hugging_face\vit_base_patch16_dinov3_lvd1689m").to(device)
    model = DINOv3(weight_dir=r"ckpts\hugging_face\vit_large_patch16_dinov3_lvd1689m").to(device)
    # print(model)
    # 加载图像
    img_path = r"F:\Desktop\master\datasets\RemoteSensing\DOTA-1.0-1.5_ss_size-1024_gap-200\trainval\images\P2751__1024__824___824.png"
    image = np.array(Image.open(img_path).convert('RGB'))
    # 图像预处理
    transform = Transforms(img_size)
    tensor_img = torch.tensor(transform.valid_transform(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
    # 获取特征图
    feature_map = model.forward(tensor_img, type='image_dense')
    # 计算指定位置的注意力热力图
    row, col = 34, 7
    heatmap = model.cosine_similarity_map(feature_map, row, col).squeeze(0).cpu().numpy()
    print(heatmap.min(),  heatmap.max())
    # 创建子图布局，左侧显示原始图像，右侧显示热力图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 左侧：显示原始图像
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    # 右侧：显示热力图
    heatmap_display = axes[1].imshow(heatmap, vmin=0.0, vmax=1.0, cmap='jet')
    axes[1].set_title("Heatmap", fontsize=14, fontweight="bold")
    axes[1].text(
        col, row, "+",
        color="red", fontsize=14, fontweight="bold",
        ha="center", va="center"
    )
    axes[1].axis("off")
    # 为热力图添加颜色条
    # plt.colorbar(heatmap_display, ax=axes[1], fraction=0.046, pad=0.04)
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"infer_result.png", dpi=200, bbox_inches='tight')
    plt.show()


