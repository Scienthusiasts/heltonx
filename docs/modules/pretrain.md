# 预训练模块 (Pretrain)

预训练模块位于 `pretrain/` 目录，专注于图像分类任务，采用 `Backbone + Head -> Classifier` 模式。

## 1. 目录结构

```
pretrain/
├── models/
│   ├── backbones/      # 骨干网络
│   │   ├── timm_backbone.py    # TIMM 模型封装
│   │   ├── clip_backbone.py    # CLIP 视觉编码器
│   │   └── dinov3_backbone.py  # DINOv3 视觉编码器
│   ├── heads/          # 分类头
│   │   ├── mlp_head.py         # MLP 分类头
│   │   ├── proto_head.py       # 原型分类头
│   │   └── vit_head.py         # ViT 分类头
│   └── classifiers/     # 分类器（Backbone + Head）
│       ├── mlpnet.py           # MLPNet 分类器
│       └── multi_task.py       # 多任务学习分类器
├── datasets/           # 数据集
│   ├── base_dataset.py
│   └── build_dataset.py
├── losses/            # 损失函数
│   ├── dist_loss.py   # 蒸馏损失
│   └── multi_task_loss.py
├── train.py           # 训练脚本
└── README.md
```

## 2. Backbone (骨干网络)

### 2.1 TIMMBackbone

封装 TIMM 库中的预训练模型。

```python
@MODELS.register
class TIMMBackbone(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=0):
        super().__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)
    
    def get_intermediate_features(self, x):
        """获取中间层特征"""
        return self.model.forward_features(x)
```

**支持模型**: ResNet, EfficientNet, ViT, SwinTransformer 等 TIMM 支持的所有模型

### 2.2 CLIPBackbone

封装 OpenCLIP 的视觉编码器。

```python
@MODELS.register
class CLIPBackbone(nn.Module):
    def __init__(self, model_name='ViT-B/32', pretrained=True):
        super().__init__()
        self.model, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
    
    def forward(self, x):
        return self.model.encode_image(x)
```

### 2.3 DINOv3Backbone

封装 DINOv3 视觉基础模型。

```python
@MODELS.register
class DINOv3Backbone(nn.Module):
    def __init__(self, model_name='dinov3_vitl', pretrained=True):
        super().__init__()
        # 加载 DINOv3 模型
        self.model = torch.hub.load('facebookresearch/dinov3', model_name)
    
    def forward(self, x):
        return self.model(x)
```

## 3. Head (分类头)

### 3.1 MLPHead

标准多层感知机分类头。

```python
@MODELS.register
class MLPHead(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 drop_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, features):
        x = self.fc1(features)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    
    def loss(self, features, targets):
        logits = self.forward(features)
        return {'loss': self.loss_fn(logits, targets), 'logits': logits}
```

### 3.2 ProtoHead

原型学习分类头，基于类原型进行分类。

```python
@MODELS.register
class ProtoHead(nn.Module):
    def __init__(self, in_features, num_classes, temperature=0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        self.temperature = temperature
    
    def forward(self, features):
        # 计算与原型的相似度
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = torch.mm(features, prototypes.t()) / self.temperature
        return logits
    
    def loss(self, features, targets):
        logits = self.forward(features)
        return {'loss': F.cross_entropy(logits, targets), 'logits': logits}
```

### 3.3 ViTHead

ViT 专用分类头，包含 CLS token 和池化操作。

```python
@MODELS.register
class ViTHead(nn.Module):
    def __init__(self, in_features, num_classes, representation_size=None):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, features):
        # features 已经包含 CLS token
        x = self.norm(features)
        return self.fc(x)
```

## 4. Classifier (分类器)

### 4.1 MLPNet

标准分类器，组合 Backbone 和 Head。

```python
@MODELS.register
class MLPNet(nn.Module):
    def __init__(self, backbone, head, load_ckpt=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
        if load_ckpt:
            self.load_ckpt(load_ckpt)
    
    def forward(self, datas, return_loss=False):
        if not return_loss:
            # 推理模式
            feats = self.backbone(datas)
            pred = self.head(feats[-1] if isinstance(feats, tuple) else feats)
            return pred
        else:
            # 训练模式
            x, y = datas[0], datas[1]
            feats = self.backbone(x)
            losses = self.head.loss(
                feats[-1] if isinstance(feats, tuple) else feats, y
            )
            return losses
```

### 4.2 MultiTaskClassifier

多任务学习分类器。

```python
@MODELS.register
class MultiTaskClassifier(nn.Module):
    def __init__(self, backbone, heads: dict):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
    
    def forward(self, datas, return_loss=False):
        if not return_loss:
            feats = self.backbone(datas)
            outputs = {name: head(feats) for name, head in self.heads.items()}
            return outputs
        else:
            x, targets = datas[0], datas[1]
            feats = self.backbone(x)
            losses = {}
            for name, head in self.heads.items():
                losses[name] = head.loss(feats, targets[name])
            total_loss = sum(losses.values())
            return {'loss': total_loss, **losses}
```

## 5. 损失函数

### 5.1 蒸馏损失

用于知识蒸馏的损失函数。

```python
@LOSSES.register
class DistillationLoss(nn.Module):
    def __init__(self, teacher_model, alpha=0.5, temperature=3.0):
        super().__init__()
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits, labels):
        # KL 散度损失
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature),
            F.softmax(teacher_logits / self.temperature),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

## 6. 训练流程

### 6.1 训练脚本结构

```python
def train(cfg):
    # 1. 构建模型
    backbone = MODELS.build_from_cfg(cfg['backbone'])
    head = MODELS.build_from_cfg(cfg['head'])
    model = MODELS.build_from_cfg({**cfg['model'], 'backbone': backbone, 'head': head})
    
    # 2. 构建数据集
    train_dataset = DATASETS.build_from_cfg(cfg['train_dataset'])
    val_dataset = DATASETS.build_from_cfg(cfg['val_dataset'])
    
    # 3. 构建优化器
    optimizer = build_optimizer(model, cfg['optimizer'])
    
    # 4. 构建 Runner
    runner = Runner(model, train_loader, val_loader, optimizer, hooks)
    
    # 5. 开始训练
    runner.train()
```

### 6.2 配置示例

```yaml
model:
  type: MLPNet
  backbone:
    type: TIMMBackbone
    model_name: resnet50
    pretrained: true
  head:
    type: MLPHead
    in_features: 2048
    out_features: 1000

train_dataset:
  type: ImageNetDataset
  root: /path/to/imagenet
  split: train

val_dataset:
  type: ImageNetDataset
  root: /path/to/imagenet
  split: val

optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.05
```

## 7. 扩展方式

### 7.1 新增 Backbone

```python
@MODELS.register
class CustomBackbone(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 自定义网络结构
        ...
    
    def forward(self, x):
        return self.features(x)
```

### 7.2 新增 Head

```python
@MODELS.register
class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes, ...):
        super().__init__()
        ...
    
    def forward(self, features):
        return self.classifier(features)
    
    def loss(self, features, targets):
        logits = self.forward(features)
        return {'loss': F.cross_entropy(logits, targets), 'logits': logits}
```

### 7.3 新增 Classifier

```python
@MODELS.register
class CustomClassifier(nn.Module):
    def __init__(self, backbone, head, ...):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, datas, return_loss=False):
        # 自定义前向逻辑
        ...
```

## 8. 技术要点

1. **特征提取**: Backbone 返回的特征可用于多种任务
2. **损失解耦**: Head 的 `forward()` 和 `loss()` 方法解耦，便于推理和训练
3. **权重加载**: 支持通过 `load_ckpt` 加载预训练权重
4. **多任务支持**: 通过 MultiTaskClassifier 支持多任务学习
