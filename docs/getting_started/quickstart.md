# 快速开始

本指南将帮助你快速上手 HeltonX 框架，进行模型训练和推理。

## 1. 基础使用

### 1.1 导入模块

```python
import torch
from heltonx.utils.register import MODELS, DATASETS, build_from_cfg
```

### 1.2 从配置构建模型

```python
# 定义模型配置
model_cfg = {
    "type": "MLPNet",
    "backbone": {
        "type": "TIMMBackbone",
        "model_name": "resnet50",
        "pretrained": True
    },
    "head": {
        "type": "MLPHead",
        "in_features": 2048,
        "out_features": 1000
    }
}

# 构建模型
model = MODELS.build_from_cfg(model_cfg)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

## 2. 图像分类任务

### 2.1 训练脚本

```python
from heltonx.pretrain.train import train

# 配置
cfg = {
    "model": {
        "type": "MLPNet",
        "backbone": {
            "type": "TIMMBackbone",
            "model_name": "resnet50",
            "pretrained": True
        },
        "head": {
            "type": "MLPHead",
            "in_features": 2048,
            "out_features": 1000
        }
    },
    "train_dataset": {
        "type": "ImageNetDataset",
        "root": "./data/imagenet",
        "split": "train",
        "transforms": ["random_resized_crop", "horizontal_flip"]
    },
    "val_dataset": {
        "type": "ImageNetDataset",
        "root": "./data/imagenet",
        "split": "val"
    },
    "optimizer": {
        "type": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.05
    },
    "scheduler": {
        "type": "CosineAnnealingLR",
        "T_max": 100
    },
    "epochs": 100,
    "batch_size": 32,
    "eval_interval": 1,
    "save_interval": 10
}

# 开始训练
train(cfg)
```

### 2.2 推理

```python
from heltonx.pretrain.models import build_classifier

# 加载模型
model = build_classifier("resnet50", num_classes=1000)
model.load_checkpoint("./checkpoints/best.pth")

# 推理
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)
```

## 3. 目标检测任务

### 3.1 训练 FCOS 检测器

```python
from heltonx.detection.train import train

cfg = {
    "model": {
        "type": "FCOS",
        "backbone": {
            "type": "ResNet",
            "depth": 50,
            "out_indices": [1, 2, 3, 4]
        },
        "fpn": {
            "type": "FPN",
            "in_channels": [512, 1024, 2048],
            "out_channels": 256
        },
        "head": {
            "type": "FCOSHead",
            "num_classes": 80
        }
    },
    "train_dataset": {
        "type": "COCODataset",
        "img_root": "./data/coco/images",
        "ann_file": "./data/coco/annotations/instances.json"
    },
    "optimizer": {
        "type": "SGD",
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001
    },
    "epochs": 50,
    "batch_size": 16
}

train(cfg)
```

### 3.2 推理

```python
from heltonx.detection.models import build_detector

# 加载模型
detector = build_detector("fcos", backbone="resnet50", num_classes=80)
detector.load_checkpoint("./checkpoints/fcos_best.pth")

# 单图推理
image = torch.randn(1, 3, 640, 640)
boxes, scores, classes = detector.infer(image)

print(f"Detected {len(boxes)} objects")
```

## 4. 生成模型

### 4.1 训练 DDPM

```python
from heltonx.generation.train import train

cfg = {
    "model": {
        "type": "DDPM",
        "denoise_model": {
            "type": "UNet",
            "in_channels": 3,
            "base_channels": 128
        },
        "img_size": [64, 64],
        "num_timesteps": 1000
    },
    "train_dataset": {
        "type": "ImageDataset",
        "root": "./data/celeba",
        "img_size": 64
    },
    "optimizer": {
        "type": "Adam",
        "lr": 0.0002
    },
    "epochs": 100
}

train(cfg)
```

### 4.2 采样生成

```python
from heltonx.generation.models import build_diffusion_model

# 加载模型
model = build_diffusion_model("ddpm", img_size=[64, 64])
model.load_checkpoint("./checkpoints/ddpm.pth")

# 生成图像
model.eval()
with torch.no_grad():
    generated_images = model.sample(batch_size=16)

# DDIM 加速采样
ddim_images = model.ddim_sample(batch_size=16, num_steps=50)
```

## 5. 大语言模型

### 5.1 加载预训练模型

```python
from heltonx.llm.models import MiniMindForCausalLM, MiniMindConfig

# 创建配置
config = MiniMindConfig(
    vocab_size=64000,
    hidden_size=768,
    num_hidden_layers=16,
    num_attention_heads=12,
    use_moe=False
)

# 加载模型
model = MiniMindForCausalLM(config)
model.load_checkpoint("./checkpoints/minimind_pretrain.pth")
```

### 5.2 文本生成

```python
from heltonx.llm.utils import generate

# 生成文本
prompt = "The future of artificial intelligence is"
output = generate(
    model, 
    tokenizer, 
    prompt, 
    max_length=100,
    temperature=0.8,
    top_p=0.9
)

print(output)
```

### 5.3 SFT 训练

```python
from heltonx.llm.trainer import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=sft_dataset,
    val_dataset=val_dataset,
    num_epochs=3,
    batch_size=8,
    gradient_accumulation_steps=4
)

trainer.train()
```

## 6. 自定义扩展

### 6.1 注册自定义模型

```python
import torch.nn as nn
from heltonx.utils.register import MODELS

@MODELS.register
class MyCustomModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x, return_loss=False):
        if return_loss:
            x, y = x
            loss = nn.functional.cross_entropy(self.fc(x), y)
            return {"loss": loss, "logits": self.fc(x)}
        return self.fc(x)

# 使用
model_cfg = {
    "type": "MyCustomModel",
    "in_features": 512,
    "out_features": 10
}
model = MODELS.build_from_cfg(model_cfg)
```

### 6.2 自定义数据集

```python
from torch.utils.data import Dataset
from heltonx.utils.register import DATASETS

@DATASETS.register
class MyCustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = self._load_samples()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 实现数据加载逻辑
        return {"image": image, "label": label}
```

## 7. 分布式训练

### 7.1 DDP 训练

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    pretrain/train_ddp.py \
    --config configs/pretrain/resnet50.yaml
```

### 7.2 Accelerate 训练

```python
from heltonx.utils.hooks_accelerate import NecessaryHookAccelerate

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    hooks=[NecessaryHookAccelerate(...)],
    accelerator=accelerator
)

trainer.train()
```

## 8. 常用命令

```bash
# 训练分类模型
python pretrain/train.py --config configs/pretrain/resnet50.yaml

# 训练检测模型
python detection/train.py --config configs/detection/fcos.yaml

# 评估模型
python pretrain/eval.py --checkpoint checkpoints/best.pth --dataset val

# 导出模型
python export.py --checkpoint checkpoints/best.pth --format onnx
```
