# 大语言模型模块 (LLM)

大语言模型模块位于 `llm/` 目录，实现了 LLM 和 VLM 的预训练、SFT 和 DPO 等功能。

## 1. 目录结构

```
llm/
├── models/
│   ├── base_models/        # 基础模型
│   │   ├── minimind.py     # MiniMind 模型实现
│   │   └── config.py       # 配置类
│   ├── attention/          # 注意力机制
│   │   ├── flash_attention.py   # Flash Attention
│   │   └── ring_attention.py     # Ring Attention
│   ├── layers/             # Transformer 层
│   │   ├── rms_norm.py     # RMSNorm
│   │   ├── rope.py         # RoPE 旋转位置编码
│   │   └── feedforward.py  # FFN
│   ├── moe/                # MoE 模块
│   │   └── moe_ffn.py      # MoE 前馈网络
│   ├── trainer/            # 训练器
│   │   ├── sft_trainer.py  # SFT 训练器
│   │   └── dpo_trainer.py  # DPO 训练器
│   └── utils/              # 工具
│       ├── tokenization.py # 分词器
│       └── generation.py   # 生成工具
├── datasets/               # 数据集
│   ├── sft_dataset.py      # SFT 数据集
│   └── dpo_dataset.py      # DPO 数据集
├── scripts/                # 训练脚本
│   ├── pretrain.py         # 预训练脚本
│   ├── sft.py              # SFT 脚本
│   └── dpo.py              # DPO 脚本
└── README.md
```

## 2. MiniMind 模型架构

### 2.1 整体结构

```
Input Tokens
    ↓
Embedding Layer (Token Embeddings + RoPE)
    ↓
N × MiniMindBlock
    │   ├── Self-Attention (with GQA)
    │   └── MLP / MoE
    ↓
RMSNorm
    ↓
LM Head (共享权重)
    ↓
Output Distribution
```

### 2.2 配置类

```python
@dataclass
class MiniMindConfig:
    """MiniMind 模型配置"""
    # 模型结构
    vocab_size: int = 64000
    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 3  # GQA
    intermediate_size: int = 3072
    max_position_embeddings: int = 8192
    
    # 范式
    use_moe: bool = False
    num_experts: int = 8
    topk_experts: int = 2
    moe_intermediate_size: int = 3072
    
    # 归一化
    rms_norm_eps: float = 1e-6
    
    # 注意力
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # 初始化
    initializer_range: float = 0.02
```

### 2.3 MiniMindBlock

```python
class MiniMindBlock(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Self-Attention
        self.self_attn = Attention(config)
        
        # MLP 或 MoE
        if config.use_moe:
            self.mlp = MOEFeedForward(config)
        else:
            self.mlp = FeedForward(config)
        
        # 归一化
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(self, hidden_states, position_embeddings, 
                past_key_value=None, use_cache=False, attention_mask=None):
        # Self-Attention with residual
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        
        attn_output, present_key_value = self.self_attn(
            hidden_states, position_embeddings, 
            past_key_value, attention_mask
        )
        hidden_states = residual + attn_output
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value
```

## 3. 注意力机制

### 3.1 标准 Attention

```python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.dropout = config.attention_dropout
    
    def forward(self, hidden_states, position_embeddings, 
                past_key_value=None, attention_mask=None):
        # Q, K, V 投影
        bsz, q_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(bsz, q_len, self.num_attention_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        # 应用 RoPE
        q, k = apply_rotary_pos_emb(q, k, position_embeddings)
        
        # 处理 KV Cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        
        present = (k, v) if use_cache else None
        
        # GQA: 扩展 K, V 到所有注意力头
        k = self._repeat_kv(k, self.num_attention_heads // self.num_key_value_heads)
        v = self._repeat_kv(v, self.num_attention_heads // self.num_key_value_heads)
        
        # 计算注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        return self.o_proj(attn_output), present
```

### 3.2 Flash Attention

```python
class FlashAttention(nn.Module):
    """Flash Attention 实现，使用 PyTorch 的 scaled_dot_product_attention"""
    
    def forward(self, q, k, v, attention_mask=None, dropout_p=0.0):
        """
        Args:
            q: [B, H, L, D]
            k: [B, H, S, D]
            v: [B, H, S, D]
        """
        # PyTorch 2.0+ 的 Flash Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=True  # 自动处理因果掩码
        )
        
        return attn_output
```

## 4. 位置编码

### 4.1 RoPE (Rotary Position Embedding)

RoPE 通过旋转操作将位置信息注入到 Query 和 Key 中。

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算旋转角度"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)  # 复数
    return freqs

def apply_rotary_pos_emb(q, k, freqs_cis):
    """应用 RoPE"""
    q_real, q_imag = q.float().reshape(*q.shape[:-1], -1, 2).unbind(-1)
    k_real, k_imag = k.float().reshape(*k.shape[:-1], -1, 2).unbind(-1)
    
    q_complex = torch.stack([q_real, q_imag], dim=-1)
    k_complex = torch.stack([k_real, k_imag], dim=-1)
    
    q_out = (q_complex * freqs_cis).flatten(3)
    k_out = (k_complex * freqs_cis).flatten(3)
    
    return q_out.type_as(q), k_out.type_as(k)
```

## 5. 归一化

### 5.1 RMSNorm

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
```

## 6. 前馈网络

### 6.1 标准 FFN

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))
```

### 6.2 MoE FFN

```python
class MOEFeedForward(nn.Module):
    """Mixture of Experts"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.topk = config.topk_experts
        
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.num_experts)
        ])
    
    def forward(self, x):
        bsz, seq_len, hidden_dim = x.shape
        
        # Gate 计算
        gate_logits = self.gate(x)
        weights, selected_experts = torch.topk(gate_logits, self.topk, dim=-1)
        
        # Softmax 归一化
        weights = F.softmax(weights, dim=-1, dtype=torch.float)
        
        # 合并 experts 的输出
        x_flat = x.view(-1, hidden_dim)
        weights_flat = weights.view(-1, self.topk)
        selected_flat = selected_experts.view(-1, self.topk)
        
        output = torch.zeros_like(x_flat)
        
        for i in range(self.topk):
            expert_idx = selected_flat[:, i]
            expert_weight = weights_flat[:, i]
            
            for e_idx in range(self.num_experts):
                mask = expert_idx == e_idx
                if mask.any():
                    expert_output = self.experts[e_idx](x_flat[mask])
                    output[mask] += expert_output * expert_weight[mask].unsqueeze(-1)
        
        return output.view(bsz, seq_len, hidden_dim)
```

## 7. LM Head

### 7.1 权重共享

```python
@MODELS.register
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config: dict, llm_config=None, load_ckpt=None):
        super().__init__(self.llm_config)
        
        self.model = MiniMindModel(self.llm_config)
        
        # LM Head 与 Embedding 共享权重
        hidden_size = self.llm_config.hidden_size
        vocab_size = self.llm_config.vocab_size
        
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight  # 权重共享
        
        if load_ckpt:
            self.load_ckpt(load_ckpt)
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None, **args):
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            past_key_values: KV Cache
            use_cache: 是否返回 KV Cache
            logits_to_keep: 保留最后 N 个位置的 logits（用于 generation）
        """
        # 计算有效序列长度
        if logits_to_keep > 0:
            input_ids = input_ids[:, -logits_to_keep:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -logits_to_keep:]
        
        # Embedding
        hidden_states = self.model.embed_tokens(input_ids)
        
        # RoPE 位置编码
        position_ids = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        
        # Transformer 前向
        outputs = self.model(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        hidden_states = outputs[0]
        
        # LM Head
        if logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:]
        
        logits = self.lm_head(hidden_states)
        
        # 计算损失（如果提供 labels）
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.llm_config.vocab_size),
                shift_labels.view(-1)
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if use_cache else None
        )
```

## 8. 训练方法

### 8.1 预训练 (Pretraining)

```python
def pretrain(cfg):
    """预训练 LLM"""
    # 1. 构建模型
    model = MODELS.build_from_cfg(cfg['model'])
    
    # 2. 构建数据集
    train_dataset = DATASETS.build_from_cfg(cfg['train_dataset'])
    
    # 3. 优化器
    optimizer = build_optimizer(model, cfg['optimizer'])
    scheduler = build_scheduler(optimizer, cfg['scheduler'])
    
    # 4. 训练循环
    for epoch in range(cfg['num_epochs']):
        for batch in DataLoader(train_dataset):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

### 8.2 SFT (Supervised Fine-Tuning)

```python
class SFTTrainer:
    def __init__(self, model, train_dataset, val_dataset, cfg):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            for batch in DataLoader(self.train_dataset):
                # SFT 数据格式: prompt + response
                # 使用 instruct/prompt 作为 system prompt
                inputs = self.format_prompt(batch)
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                self.backpropagate(loss)
            
            # 验证
            self.validate()
    
    def format_prompt(self, batch):
        """格式化 SFT 数据"""
        prompts = [f"System: {s}\nUser: {q}\nAssistant: " 
                   for s, q in zip(batch['system'], batch['question'])]
        responses = batch['answer']
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            responses,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        )
        
        # Labels: -100 表示不计算损失的 token
        labels = inputs['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {'input_ids': inputs['input_ids'], 
                'attention_mask': inputs['attention_mask'],
                'labels': labels}
```

### 8.3 DPO (Direct Preference Optimization)

```python
class DPOTrainer:
    def __init__(self, model, ref_model, train_dataset, cfg):
        self.model = model
        self.ref_model = ref_model  # Reference 模型（通常冻结）
        self.beta = cfg.get('beta', 0.1)  # DPO temperature
    
    def compute_logprobs(self, model, input_ids, labels, attention_mask):
        """计算 log probabilities"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for next-token prediction
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[:, :-1].gather(2, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        # Mask invalid tokens
        mask = (labels[:, 1:] != -100).float()
        log_probs = (token_log_probs * mask).sum(-1) / mask.sum(-1)
        
        return log_probs
    
    def train_step(self, batch):
        """DPO 单步训练"""
        # 提取 chosen 和 rejected
        chosen_ids = batch['chosen_input_ids']
        rejected_ids = batch['rejected_input_ids']
        chosen_mask = batch['chosen_attention_mask']
        rejected_mask = batch['rejected_attention_mask']
        
        # Policy 模型 log probs
        policy_chosen = self.compute_logprobs(self.model, chosen_ids, chosen_ids, chosen_mask)
        policy_rejected = self.compute_logprobs(self.model, rejected_ids, rejected_ids, rejected_mask)
        
        # Reference 模型 log probs
        with torch.no_grad():
            ref_chosen = self.compute_logprobs(self.ref_model, chosen_ids, chosen_ids, chosen_mask)
            ref_rejected = self.compute_logprobs(self.ref_model, rejected_ids, rejected_ids, rejected_mask)
        
        # DPO 损失
        chosen_log_ratio = policy_chosen - ref_chosen
        rejected_log_ratio = policy_rejected - ref_rejected
        
        loss = -F.logsigmoid(self.beta * (chosen_log_ratio - rejected_log_ratio)).mean()
        
        return loss
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            for batch in DataLoader(self.train_dataset):
                loss = self.train_step(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
```

## 9. 工具函数

### 9.1 生成函数

```python
def generate(model, tokenizer, prompt, max_length=100, temperature=1.0, top_p=0.9):
    """文本生成"""
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            
            # Top-p 采样
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
```

## 10. 训练配置示例

### 10.1 预训练配置

```yaml
model:
  type: MiniMindForCausalLM
  llm_config:
    vocab_size: 64000
    hidden_size: 768
    num_hidden_layers: 16
    num_attention_heads: 12
    num_key_value_heads: 3
    intermediate_size: 3072
    max_position_embeddings: 8192
    use_moe: false

train_dataset:
  type: PretrainDataset
  data_files: /path/to/pretrain_data
  max_length: 2048

optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 0.1

scheduler:
  type: CosineAnnealingLR
  T_max: 1000
```

### 10.2 SFT 配置

```yaml
model:
  type: MiniMindForCausalLM
  load_ckpt: /path/to/pretrain_checkpoint

sft_dataset:
  type: SFTDataset
  data_files: /path/to/sft_data
  max_length: 2048

trainer:
  type: SFTTrainer
  num_epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
```

### 10.3 DPO 配置

```yaml
model:
  type: MiniMindForCausalLM
  load_ckpt: /path/to/sft_checkpoint

ref_model:
  type: MiniMindForCausalLM
  load_ckpt: /path/to/sft_checkpoint  # 冻结

dpo_dataset:
  type: DPODataset
  data_files: /path/to/dpo_data

trainer:
  type: DPOTrainer
  beta: 0.1
  lr: 1e-6
```

## 11. 扩展方式

### 11.1 新增模型

```python
@MODELS.register
class CustomLLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        # 自定义模型结构
        ...
    
    def forward(self, input_ids, ...):
        # 自定义前向
        ...
```

### 11.2 新增训练方法

```python
class CustomTrainer:
    def __init__(self, model, ...):
        self.model = model
    
    def train_step(self, batch):
        # 自定义训练步骤
        ...
```

## 12. 技术要点

1. **GQA (Grouped Query Attention)**: 减少 KV 头的数量，降低计算和内存成本
2. **RoPE**: 无需显式位置编码，支持更长上下文
3. **权重共享**: Embedding 和 LM Head 共享权重
4. **Flash Attention**: 高效注意力计算
5. **MoE**: 稀疏激活，减少计算量
6. **KV Cache**: 加速自回归生成
