# API 参考

## 核心模块 (heltonx.utils)

### Register 类

注册器基类，用于模块的动态注册和实例化。

```python
from heltonx.utils.register import Register, MODELS, DATASETS
```

#### 方法

##### `register(name)`

装饰器，用于注册模块。

```python
@MODELS.register
class MyModel(nn.Module):
    pass
```

##### `build_from_cfg(cfg, **kwargs)`

从配置字典构建实例。

```python
cfg = {
    "type": "ResNet",
    "depth": 50
}
model = MODELS.build_from_cfg(cfg)
```

##### `get(name)`

获取已注册的模块类。

```python
ResNet = MODELS.get("ResNet")
```

##### `list_modules()`

列出所有已注册的模块名称。

```python
print(MODELS.list_modules())
```

### 预定义注册器

| 注册器 | 用途 |
|--------|------|
| `MODELS` | 模型注册器 |
| `DATASETS` | 数据集注册器 |
| `OPTIMIZERS` | 优化器注册器 |
| `SCHEDULERS` | 学习率调度器注册器 |
| `EVAL_PIPELINES` | 评估管道注册器 |

---

### Hooks (钩子)

#### NecessaryHook

```python
from heltonx.utils.hooks import NecessaryHook
```

训练/评估过程的回调基类。

**构造函数参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `eval_pipeline` | EvaluationPipeline | 评估管道 |
| `save_interval` | int | 保存间隔（epoch） |
| `metric_name` | str | 保存指标的名称 |
| `work_dir` | str | 工作目录 |

**方法：**

| 方法 | 说明 |
|------|------|
| `hook_before_train(runner)` | 训练开始前调用 |
| `hook_after_batch(runner)` | 每个 batch 后调用 |
| `hook_after_epoch(runner)` | 每个 epoch 后调用 |
| `hook_after_eval(runner)` | 评估结束后调用 |

---

### Checkpoint (检查点)

```python
from heltonx.utils.checkpoint import load_checkpoint, save_checkpoint
```

#### `load_checkpoint(model, checkpoint_path, strict=True)`

加载模型检查点。

```python
load_checkpoint(model, "./checkpoints/best.pth")
```

#### `save_checkpoint(model, optimizer, epoch, save_path, **kwargs)`

保存检查点。

```python
save_checkpoint(model, optimizer, epoch, "./checkpoints/epoch_10.pth")
```

---

### Logger (日志)

```python
from heltonx.utils.logger import Logger, TensorboardLogger
```

#### Logger

```python
logger = Logger(
    log_dir="./logs",
    project_name="heltonx",
    use_tensorboard=True,
    use_wandb=False
)
```

**方法：**

| 方法 | 说明 |
|------|------|
| `train_iter_log_printer(step, epoch, optimizer, losses)` | 打印训练迭代日志 |
| `train_epoch_log_printer(epoch, metrics)` | 打印 epoch 日志 |
| `val_log_printer(metrics)` | 打印验证日志 |
| `log_image(tag, image)` | 记录图像 |

---

### Optimizer & Scheduler

```python
from heltonx.utils.optimizer import build_optimizer
from heltonx.utils.scheduler import build_scheduler
```

#### `build_optimizer(model, optimizer_cfg)`

构建优化器。

```python
optimizer = build_optimizer(model, {
    "type": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.05
})
```

#### `build_scheduler(optimizer, scheduler_cfg)`

构建学习率调度器。

```python
scheduler = build_scheduler(optimizer, {
    "type": "CosineAnnealingLR",
    "T_max": 100
})
```

---

## 预训练模块 (heltonx.pretrain)

### 模型

#### Backbone

```python
from heltonx.pretrain.models.backbones import TIMMBackbone, CLIPBackbone
```

**TIMMBackbone：**

```python
backbone = TIMMBackbone(
    model_name="resnet50",
    pretrained=True,
    num_classes=0
)
```

**CLIPBackbone：**

```python
backbone = CLIPBackbone(
    model_name="ViT-B/32",
    pretrained=True
)
```

#### Head

```python
from heltonx.pretrain.models.heads import MLPHead, ProtoHead, ViTHead
```

**MLPHead：**

```python
head = MLPHead(
    in_features=2048,
    hidden_features=1024,
    out_features=1000,
    drop_rate=0.1
)
```

**ProtoHead：**

```python
head = ProtoHead(
    in_features=2048,
    num_classes=1000,
    temperature=0.1
)
```

#### Classifier

```python
from heltonx.pretrain.models.classifiers import MLPNet, MultiTaskClassifier
```

**MLPNet：**

```python
classifier = MLPNet(
    backbone=backbone,
    head=head,
    load_ckpt=None
)

# 训练
losses = classifier(datas, return_loss=True)

# 推理
output = classifier(images, return_loss=False)
```

---

## 目标检测模块 (heltonx.detection)

### 模型

#### Backbone

```python
from heltonx.detection.models.backbones import ResNet, CSPDarknet
```

**ResNet：**

```python
backbone = ResNet(
    depth=50,
    num_stages=4,
    out_indices=(1, 2, 3, 4)
)
```

**CSPDarknet：**

```python
backbone = CSPDarknet(
    depth_multiple=1.0,
    width_multiple=1.0
)
```

#### Neck

```python
from heltonx.detection.models.necks import FPN, PAFPN, C2fPAFPN
```

**FPN：**

```python
neck = FPN(
    in_channels=[512, 1024, 2048],
    out_channels=256,
    num_outs=5
)
```

#### Head

```python
from heltonx.detection.models.heads import FCOSHead, YOLOv5Head
```

**FCOSHead：**

```python
head = FCOSHead(
    in_channels=256,
    num_classes=80,
    num_convs=4
)
```

#### Detector

```python
from heltonx.detection.models.detectors import FCOS, YOLOv5
```

**FCOS：**

```python
detector = FCOS(
    backbone=backbone,
    fpn=fpn,
    head=fcos_head,
    bbox_coder=bbox_coder,
    nms=nms
)

# 训练
losses = detector((images, targets), return_loss=True)

# 推理
boxes, scores, classes = detector.infer(images)
```

---

## 生成模型模块 (heltonx.generation)

### 扩散模型

```python
from heltonx.generation.models.diffusion import DDPM, DDIM
```

#### DDPM

```python
model = DDPM(
    denoise_model=unet,
    img_size=(64, 64),
    num_timesteps=1000,
    beta_schedule="linear"
)

# 训练
noise_pred, noise = model(x_start, t)

# 采样
samples = model.sample(batch_size=16)
```

#### DDIM

```python
model = DDIM(
    denoise_model=unet,
    img_size=(64, 64),
    ddim_num_steps=50,
    ddim_eta=0.0
)

samples = model.ddim_sample(batch_size=16)
```

### 自编码器

```python
from heltonx.generation.models.autoencoder import VAE, VQVAE, CVAE
```

**VAE：**

```python
vae = VAE(
    encoder=encoder,
    decoder=decoder,
    latent_dim=4
)

z, mu, logvar = vae.encode(x)
x_recon = vae.decode(z)
loss, recon_loss, kl_loss = vae.loss(x)
```

**VQVAE：**

```python
vqvae = VQVAE(
    encoder=encoder,
    decoder=decoder,
    latent_dim=256,
    num_embeddings=8192
)

z_q, encoding_idx = vqvae.encode(x)
x_recon = vqvae.decode(z_q)
```

### DiT

```python
from heltonx.generation.models.transformer import DiT, DiTCrossAttention
```

**DiT：**

```python
dit = DiT(
    img_size=(32, 32),
    patch_size=2,
    in_channels=4,
    hidden_size=1152,
    num_heads=16,
    num_layers=28
)

output = dit(x, t, y)
```

---

## LLM 模块 (heltonx.llm)

### 模型

```python
from heltonx.llm.models import MiniMindForCausalLM, MiniMindConfig
```

#### MiniMindConfig

```python
config = MiniMindConfig(
    vocab_size=64000,
    hidden_size=768,
    num_hidden_layers=16,
    num_attention_heads=12,
    num_key_value_heads=3,
    intermediate_size=3072,
    max_position_embeddings=8192,
    use_moe=False
)
```

#### MiniMindForCausalLM

```python
model = MiniMindForCausalLM(
    config=config,
    load_ckpt=None
)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
```

### 训练器

```python
from heltonx.llm.trainer import SFTTrainer, DPOTrainer
```

#### SFTTrainer

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=sft_dataset,
    val_dataset=val_dataset,
    num_epochs=3,
    batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5
)

trainer.train()
```

#### DPOTrainer

```python
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=dpo_dataset,
    beta=0.1,
    learning_rate=1e-6
)

trainer.train()
```

### 工具函数

```python
from heltonx.llm.utils import generate, batch_generate
```

**generate：**

```python
output = generate(
    model,
    tokenizer,
    prompt="Hello, world!",
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    top_k=50
)
```

---

## 通用工具

### 权重初始化

```python
from heltonx.utils.tools import init_weights

init_weights(model, init_type='xavier')
```

### 设备管理

```python
from heltonx.utils.tools import get_device, to_device

device = get_device()
data = to_device(data, device)
```

### 学习率调整

```python
from heltonx.utils.tools import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=100000
)
```
