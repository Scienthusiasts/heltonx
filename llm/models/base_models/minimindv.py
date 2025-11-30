import math
import torch
import os
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
# PreTrainedModel负责权重管理和训练; GenerationMixin负责推理/生成逻辑
from transformers import PreTrainedModel, GenerationMixin
from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from heltonx.utils.wrappers import NoSaveWrapper
from .blocks import *
from llm.models.base_model_configs import MiniMindVConfig
# 注册机制
from heltonx.utils.register import MODELS
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix
# 继承自LLM
from .minimind import MiniMindForCausalLM




class VisionProj(nn.Module):
    """视觉特征映射到文本tokens embeddings空间的桥接器
    """
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size),
            # nn.LayerNorm(self.hidden_size, eps=1e-05),
            # RMSNorm(self.hidden_size, eps=1e-05),
            # nn.SiLU(),
            # nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self, x):
        return self.vision_proj(x)




# 继承自语言模型
@MODELS.register
class MiniMindForCausalVLM(MiniMindForCausalLM):

    def __init__(self, config:dict, vision_encoder:nn.Module=None, load_ckpt=None):
        # 将 config 字典转换成 MiniMindConfig 实例（HuggingFace 风格）
        llm_config = MiniMindVConfig(**config)
        # 调用父类 MiniMindForCausalLM 的构造函数，构建 LLM 主干
        super(MiniMindForCausalVLM, self).__init__(config, llm_config)
        v_hidden_size = config.get('v_hidden_size')
        hidden_size = config.get('hidden_size')

        # 视觉模块(直接导入预训练模型)
        if vision_encoder:
            self.vision_encoder = NoSaveWrapper(vision_encoder).eval()
            # 冻结 vision_encoder 的所有参数
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # 将视觉特征映射到文本 token embedding 空间的投影层
        self.vision_proj = VisionProj(v_hidden_size, hidden_size)

        # 导入权重
        if load_ckpt:
            load_state_dict_with_prefix(self, load_ckpt)
            

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        '''桥接器的前向, 将图像编码器输出特征通过桥接器映射到文本tokens embeddings空间
            image_tensors: 形状 [B,3,H,W] 的 pixel_values
            返回:  CLIP vision encoder 的 patch embeddings (去掉 CLS token)
        '''
        with torch.no_grad():
            img_embedding = vision_model.module(image_tensors)
        return img_embedding


    def find_indices(self, tokens, image_ids):
        """查找 input_ids 中出现的 image token pattern（连续的 image_ids）
           返回每个 batch image token 的 (start_idx, end_idx) 列表
        """
        image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
        len_image_ids = len(image_ids)
        # 当输入太短时不可能匹配
        if len_image_ids > tokens.size(1):
            return None
        # unfold 将序列展开成滑动窗口视图：形状 (B, seq-len, window-size)
        tokens_view = tokens.unfold(1, len_image_ids, 1)
        # 找到窗口内所有 token 为image token的地方
        matches = (tokens_view == image_ids_tensor).all(dim=2)
        # 构造输出：每个 batch 对应一个 {图片i: list[(start,end)], ...}
        return {
            batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                        matches[batch_idx].nonzero(as_tuple=True)[0]]
            for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
        } or None
        
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        """推理图像生成视觉token embeddings, 并将视觉embeddings插入到对应占位image tokens 的 hidden_states 位置
        """
        # 查找 tokens 中需要替换的 image token 位置(占位tokens替换为真实的图像tokens)
        image_indices = self.find_indices(tokens, self.llm_config.image_ids)
        # 如果存在视觉输入且存在匹配的 image_ids
        if vision_tensors is not None and image_indices:
            # [BS, tokens_len, embed_dim]
            vision_proj = self.vision_proj(vision_tensors)
            # 若输入多图，则按 batch 堆叠
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            # 按 batch 处理
            for i in range(h.size(0)):
                if i in image_indices:
                    # 当前 batch 的 hidden_states
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            # 替换掉 image token 区间，插入 vision feature
                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[:seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h
    


    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # KV Cache 初始化
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # 将输入tokens转换为dense embedding(文本)
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # 当第一次 forward（start_pos == 0）+ 有图像输入时执行视觉增强
        if pixel_values is not None and start_pos == 0:
            # 如果 pixel_values 是 [B,1,N,3,H,W]，去掉无用维度
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            # 多 batch 时 stack 方式不一样
            stack_dim = 1 if bs > 1 else 0
            # 逐图像调用 CLIP 得到 patch embedding(不是batch推理)
            vision_tensors = torch.stack([
                self.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)], dim=stack_dim)
            # 推理图像生成视觉token embeddings, 并将视觉embeddings插入到对应占位image tokens 的 hidden_states 位置
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        '''下面这部分就和LLM完全一样了'''
        # 生成旋转位置编码
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )
        # 逐层 transformer 前向（支持 KV Cache）
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask
            )
            presents.append(present)
        # 最终 LayerNorm
        hidden_states = self.model.norm(hidden_states)
        # MOE 模块的辅助 loss（仅当使用 MOEFeedForward 时）
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        return self.OUT
