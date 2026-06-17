# HeltonX 整体架构概览

## 1. 项目简介

HeltonX 是一个基于 PyTorch 的通用深度学习框架，目标是提供一个从零实现的、模块化的、可扩展的深度学习工具库。项目强调代码的可读性和可扩展性，适合学习和实际项目使用。

## 2. 目录结构

```
heltonx/
├── heltonx/          # 核心框架模块
│   ├── utils/        # 通用工具（注册器、钩子、检查点、日志等）
│   └── __init__.py
├── pretrain/         # 预训练模块（图像分类任务）
├── detection/        # 目标检测模块（FCOS, YOLOv5）
├── generation/       # 生成模型模块（DDPM, DiT, VAE等）
└── llm/             # 大语言模型模块（LLM, VLM预训练与微调）
```

## 3. 核心设计原则

### 3.1 注册器模式 (Registry Pattern)

整个框架的核心是注册器模式，通过 `@MODELS.register`、`@DATASETS.register` 等装饰器实现模块的动态注册。

```python
# 定义注册器
MODELS = Register()

# 使用装饰器注册
@MODELS.register
class MyModel(nn.Module):
    pass

# 从配置构建
model = MODELS.build_from_cfg(cfg)
```

### 3.2 配置驱动 (Configuration-Driven)

所有模型、数据集、优化器等都通过配置字典进行实例化，实现了代码与配置的分离。

### 3.3 钩子机制 (Hook Mechanism)

通过 `NecessaryHook` 类实现训练/评估过程中的回调点，包括：
- 日志记录
- 评估触发
- 模型保存
- 学习率调度

### 3.4 统一接口设计

- 训练接口: `trainer.train()`
- 评估接口: `trainer.val()` / `detector.infer()`
- 模型接口: `model.forward(datas, return_loss=True/False)`

## 4. 各模块职责

| 模块 | 职责 | 主要任务 |
|------|------|----------|
| `heltonx` | 核心框架 | 注册器、钩子、检查点、日志、优化器 |
| `pretrain` | 图像分类 | Backbone + Head 分类器、蒸馏、多任务学习 |
| `detection` | 目标检测 | FCOS (Anchor-Free)、YOLOv5 (Anchor-Based) |
| `generation` | 生成模型 | DDPM/DDIM、Flow Matching、DiT、VAE/VQVAE |
| `llm` | 大语言模型 | LLM/VLM预训练、SFT、DPO、MoE、Flash Attention |

## 5. 数据流

```
配置文件 (YAML)
    ↓
build_from_cfg() → 实例化各组件
    ↓
Runner/Engine → 管理训练循环
    ↓
NecessaryHook → 控制回调流程
    ↓
Logger/Checker → 记录和保存
```

## 6. 扩展方式

### 6.1 新增模型

```python
@MODELS.register
class NewModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...
    
    def forward(self, datas, return_loss=False):
        if return_loss:
            return self.compute_loss(datas)
        return self.predict(datas)
```

### 6.2 新增数据集

```python
@DATASETS.register
class NewDataset(BaseDataset):
    def __getitem__(self, idx):
        ...
```

### 6.3 新增钩子

```python
class CustomHook(NecessaryHook):
    def hook_after_epoch(self, runner):
        # 自定义逻辑
        pass
```

## 7. 分布式训练支持

框架原生支持两种分布式训练方案：
- **PyTorch DDP**: 原生实现，通过 `train_ddp.py` 使用
- **HuggingFace Accelerate**: 通过 `hooks_accelerate.py` 提供兼容

## 8. 与 PyTorch 的关系

- 基于 PyTorch 深度学习框架
- 封装了训练/评估的通用流程
- 提供了丰富的预训练模型和工具
- 保持了 PyTorch 的灵活性和可扩展性
