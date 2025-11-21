
import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast





class RMSNorm(torch.nn.Module):
    """RMSNorm实现(相比LN丢弃了求mean操作)
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 先在 float32 上计算均方根，避免 AMP 的 dtype 问题
        orig_dtype = x.dtype
        # 计算 mean of squares (float32)，保持 keepdim
        variance = x.pow(2).mean(-1, keepdim=True).float()
        # torch.rsqrt是取倒数操作
        inv_rms = torch.rsqrt(variance + self.eps)  # float32
        x_normed = x.float() * inv_rms  # still float32
        # apply weight (broadcast)
        out = self.weight * x_normed  # weight is float32 param
        return out.to(orig_dtype)





def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), 
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0), 
            rope_scaling.get("beta_slow", 1.0)
        )
        if end / orig_max > 1.0:
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            # λ = (β·α - β + 1)/(β·α) YaRN标准公式
            scale = torch.where(torch.arange(dim // 2, device=freqs.device) < corr_dim, (beta * factor - beta + 1) / (beta * factor), 1.0 / factor)
            freqs = freqs * scale

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin





def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed





def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """替代torch.repeat_interleave(x, dim=2, repeats=n_rep), 
       对键值头(Key-Value heads)进行重复扩展以匹配查询头数量(GQA中使用)
        x:     形状为 [bs, slen, num_key_value_heads, head_dim] 的键或值张量
        n_rep: 每个键值头需要重复的次数
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )





class Attention(nn.Module):
    """
    多头注意力机制模块，支持分组查询注意力(GQA)和旋转位置编码(RoPE)
    
    特性：
    - 支持Flash Attention（如果可用）
    - 支持KV缓存用于推理加速
    - 支持旋转位置编码(RoPE)
    - 支持分组查询注意力(GQA)以节省计算和内存
    """
    
    def __init__(self, args: PretrainedConfig):
        super().__init__()
        
        # ==================== 注意力头配置 ====================
        # 设置查询头数和键值头数，支持分组查询注意力(GQA)
        # 注意查询头数必须能被键值头数整除
        self.num_q_heads = args.num_attention_heads
        self.num_kv_heads = (args.num_attention_heads 
                                  if args.num_key_value_heads is None 
                                  else args.num_key_value_heads)
        
        # 计算每个键值头需要重复的次数（用于GQA）
        self.num_repeats = self.num_q_heads // self.num_kv_heads
        self.head_dim = args.hidden_size // self.num_q_heads
        
        # ==================== 投影层 ====================
        # 查询、键、值、输出投影层
        self.q_proj = nn.Linear(args.hidden_size, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, args.hidden_size, bias=False)
        
        # ==================== 正则化层 ====================
        self.attn_dropout = nn.Dropout(args.dropout)
        self.residual_dropout = nn.Dropout(args.dropout)
        self.dropout_rate = args.dropout
        
        # ==================== Flash Attention支持 ====================
        self.enable_flash_attention = (
            hasattr(F, 'scaled_dot_product_attention') and 
            getattr(args, 'flash_attn', False)
        )
        if not self.enable_flash_attention:
            print("警告: 使用标准注意力实现。如需Flash Attention，请升级PyTorch至2.0+版本")

    def forward(
        self,
        x: torch.Tensor,
        pe: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin) 旋转位置编码
        cache_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            position_embeddings: 旋转位置编码的cos和sin值
            cache_kv:  过去的键值缓存，用于推理加速
            use_cache: 是否使用KV缓存
            attn_mask: 注意力掩码
            
        Returns:
            output:   注意力输出 [batch_size, seq_len, hidden_size]
            cache_kv: 更新后的KV缓存(如果use_cache=True)
        """
        batch_size, seq_len, _ = x.shape
        cos, sin = pe
        
        '''线性投影qkv'''
        # 应用查询、键、值投影 num_q_heads > num_kv_heads
        q = self.q_proj(x)  # [bs, seq_len, num_q_heads * dim]
        k = self.k_proj(x)  # [bs, seq_len, num_kv_heads * dim]
        v = self.v_proj(x)  # [bs, seq_len, num_kv_heads * dim]
        # 改变形状为多头格式
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim) 
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        '''添加旋转位置编码'''
        # 对查询和键应用旋转位置编码
        q, k = apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])
        
        '''kv cache'''
        if cache_kv is not None:
            # 将当前的k和v拼在KV cache的最后
            cache_k, cache_v = cache_kv
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)
        # 更新 KV cache
        current_kv_cache = (k, v) if use_cache else None
        
        '''GQA'''
        # 扩展键值头以匹配查询头数量(复制kv)
        # [bs,seq_len, num_kv_heads, head_dim] -> [bs, seq_len, num_q_heads, dim]
        k = repeat_kv(k, self.num_repeats)
        v = repeat_kv(v, self.num_repeats)
        # 转置为注意力计算的格式 [bs, seq_len, num_heads, dim] -> [bs, num_heads, seq_len, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # multi head attention计算
        if self._use_flash_attn(seq_len, attn_mask):
            output = self._flash_attn(q, k, v, attn_mask)
        else:
            output = self._standard_attn(q, k, v, attn_mask, seq_len)
        
        '''输出投影'''
        # 转置回原始格式
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # 应用输出投影和残差dropout
        output = self.o_proj(output)
        output = self.residual_dropout(output)
        
        return output, current_kv_cache



    def _use_flash_attn(self, seq_len: int, attn_mask: Optional[torch.Tensor]) -> bool:
        """判断是否使用Flash Attention
            Flash Attention使用条件:
            1. 启用Flash Attention
            2. 序列长度大于1
            3. 注意力掩码为None或全1 (无padding)
            4. 训练时或推理时都可用
        """
        return (self.enable_flash_attention and seq_len > 1 and 
                (attn_mask is None or torch.all(attn_mask == 1)))


    def _flash_attn(self, q, k, v, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Flash Attention前向传播
           使用PyTorch 2.0+的高效注意力实现，具有更好的内存效率和速度
        """
        # 准备注意力掩码（如果需要）
        attn_mask = None
        if attn_mask is not None:
            attn_mask = attention_mask.view(query.size(0), 1, 1, -1
            ).expand(query.size(0), self.num_q_heads, query.size(2), -1
            ).bool()
        # 应用Flash Attention(调用pytorch内部函数实现)
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=True  # 自回归语言模型的因果掩码
        )


    def _standard_attn(self, q, k, v, attn_mask: Optional[torch.Tensor], seq_len: int) -> torch.Tensor:
        """标准注意力前向传播, 当Flash Attention不可用时使用的回退方案
        """
        # 计算注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 因果掩码(上三角矩阵, 防止看到未来信息)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=attn_scores.device), diagonal=1)
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        # 外部注意力掩码（如padding掩码）
        if attn_mask is not None:
            extended_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -1e9  # 将0变为负无穷
            attn_scores = attn_scores + extended_mask

        # 转换为float32进行softmax以增强数值稳定性
        attn_weights = F.softmax(attn_scores.float(), dim=-1).type_as(q)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 应用注意力权重到值向量
        return attn_weights @ v






class FeedForward(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache
