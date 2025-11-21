import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
# PreTrainedModel负责权重管理和训练; GenerationMixin负责推理/生成逻辑
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .blocks import *
from llm.models.base_model_configs import MiniMindConfig
# 注册机制
from heltonx.utils.register import MODELS
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix


class MiniMindBlock(nn.Module):
    """Attention + FFN(MoE)
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # FFN(MoE在这里体现)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        # attention
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # 残差连接
        hidden_states += residual
        # FFN
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value





class MiniMindModel(nn.Module):
    """token embedding + Transformer decoder x K + RMSNorm"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # self.embed_tokens的作用相当于word2vec, 将稀疏embedding转换为dense embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 生成旋转位置编码, 并将位置编码注册为模型参数(不更新)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings, 
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # KV Cache 初始化
        past_key_values = past_key_values or [None] * len(self.layers)
        # 生成位置编码
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        # 将输入tokens转换为dense embedding
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 逐层处理特征
        all_layer_kv_cache = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask
            )
            all_layer_kv_cache.append(present)
        # 最后一层输出特征
        hidden_states = self.norm(hidden_states)
        # MoE负载均衡损失
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        return hidden_states, all_layer_kv_cache, aux_loss





@MODELS.register
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """ model(token embedding + Transformer decoder x K + RMSNorm) + linear + softmax
        MiniMind 的因果语言模型(CausalLM)封装类。
        继承 PreTrainedModel + GenerationMixin 的原因：
            - PreTrainedModel 提供 load_state_dict、from_pretrained 等功能
            - GenerationMixin 提供 generate() 相关推理能力(如采样、beam search、kv cache 推理)
    """
    def __init__(self, config: dict, llm_config=None, load_ckpt=None):
        # 将 config 字典转换成 MiniMindConfig 实例（HuggingFace 风格）
        self.llm_config = llm_config if llm_config else MiniMindConfig(**config)
        super(MiniMindForCausalLM, self).__init__(self.llm_config)
        hidden_size = config.get('hidden_size')
        vocab_size = config.get('vocab_size')
        # 不包含分类头的那部分llm
        self.model = MiniMindModel(self.llm_config)
        # 决策头, 将transformer最后一层特征映射为 vocab logits
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # 权重共享：embedding 和 lm_head 共用同一个矩阵 (lm_head其实就是embed_tokens矩阵的转置)
        # 好处: 保证对称性, 减少参数, 提升性能(几乎所有 GPT 系列都这么做)
        self.model.embed_tokens.weight = self.lm_head.weight
        # 输出结构（类似 transformers 的 CausalLMOutputWithPast）
        # 用于存储：
        #   - logits
        #   - past_key_values
        #   - last_hidden_state
        #   - aux_loss
        self.OUT = CausalLMOutputWithPast()
        # 导入权重
        if load_ckpt:
            load_state_dict_with_prefix(self, load_ckpt)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
            input_ids:       输入 token ids, shape = [batch, seq_len]
            attention_mask:  padding mask; padding=0, 非padding=1
            past_key_values: 预存的 KV cache, 用于加速推理, 训练时通常为 None, inference 时为 List[layers x (key, value)]
            use_cache:       是否返回新的 kv cache(用于 generate)
            logits_to_keep:  只计算最后几个 token 的 logits(reduce memory)
                             在 generate() 时非常重要：
                                 - 因为我们只需要最后一个 token 的 logits
                                 - 不需要所有序列的 logits
                             示例：
                                 logits_to_keep=1 → 仅预测最后一个 token
        """
        h, past_kvs, aux_loss = self.model(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
            **args
        )
        # 根据 logits_to_keep 选择要计算 logits 的 token 范围
        # 通常 inference 时只需要最后一个 token 的 logits
        # slice_indices = slice(-1, None, None) == 只取最后一个 token
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 将选取的 hidden states 映射为 vocab logits
        logits = self.lm_head(h[:, slice_indices, :])
        # 组织输出为 HuggingFace 风格的 CausalLMOutputWithPast
        # 这里通过 __setitem__ 方式设置 key 和 value
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
