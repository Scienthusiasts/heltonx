# HeltonX 核心框架详解

核心框架位于 `heltonx/` 目录，提供了项目级别的通用组件和抽象。

## 1. 目录结构

```
heltonx/
├── utils/
│   ├── register.py      # 注册器实现
│   ├── hooks.py         # 训练钩子（DDP版本）
│   ├── hooks_accelerate.py  # 训练钩子（Accelerate版本）
│   ├── checkpoint.py    # 检查点管理
│   ├── logger.py        # 日志系统
│   ├── optimizer.py     # 优化器工具
│   ├── scheduler.py      # 学习率调度器
│   ├── tools.py         # 通用工具函数
│   └── __init__.py
└── __init__.py
```

## 2. 注册器 (Register)

### 2.1 核心原理

`Register` 类继承自 `dict`，通过装饰器模式实现模块的动态注册。

```python
class Register(dict):
    def add_item(self, key, value):
        self[key] = value
        return value
    
    def register(self, target):
        return lambda x: self.add_item(target, x)
    
    def build_from_cfg(self, cfg: dict, **kwargs):
        # 递归实例化嵌套配置
        module_type = cfg.pop("type")
        for k, v in cfg.items():
            if isinstance(v, dict) and "type" in v:
                cfg[k] = self.build_from_cfg(v)
        return self.build(module_type, **cfg, **kwargs)
```

### 2.2 预定义注册器

```python
MODELS = Register()      # 模型注册器
DATASETS = Register()    # 数据集注册器
OPTIMIZERS = Register()  # 优化器注册器
SCHEDULERS = Register()  # 调度器注册器
EVAL_PIPELINES = Register()  # 评估管道注册器
```

### 2.3 使用示例

```python
# 1. 定义模型
@MODELS.register
class MyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)

# 2. 从配置构建
cfg = {
    "type": "MyModel",
    "in_features": 512,
    "out_features": 10
}
model = MODELS.build_from_cfg(cfg)
```

## 3. 钩子机制 (Hooks)

### 3.1 NecessaryHook 类

`NecessaryHook` 是训练/评估过程中的回调基类，定义了多个可重写的钩子方法。

#### 钩子方法

| 方法 | 调用时机 | 默认行为 |
|------|----------|----------|
| `hook_before_train` | 训练开始前 | 初始化日志 |
| `hook_after_batch` | 每个 batch 后 | 训练日志 |
| `hook_after_epoch` | 每个 epoch 后 | 评估、模型保存 |
| `hook_after_eval` | 评估结束后 | 返回评估指标 |

#### 核心实现

```python
class NecessaryHook():
    def __init__(self, eval_pipeline=None, save_interval=1, ...):
        self.eval_pipeline = eval_pipeline
        self.save_interval = save_interval
    
    def hook_after_batch(self, runner):
        if runner.mode == 'train' and dist.get_rank() == 0:
            runner.runner_logger.train_iter_log_printer(
                runner.cur_step, runner.cur_epoch, 
                runner.optimizer, runner.losses
            )
    
    def hook_after_epoch(self, runner):
        if runner.cur_epoch % runner.eval_interval == 0:
            if self.eval_pipeline:
                flag_metric_name = self.hook_after_eval(runner)
                # 保存最优模型
                self.save_best_model(runner, flag_metric_name)
```

### 3.2 Accelerate 兼容

`hooks_accelerate.py` 提供了与 HuggingFace Accelerate 库的兼容实现。

## 4. 检查点管理 (Checkpoint)

### 4.1 功能

- 加载/保存模型权重
- 支持部分加载（load_ckpt）
- 支持预训练权重加载
- 自动跳过不匹配的参数

### 4.2 核心代码

```python
def load_checkpoint(model, checkpoint_path, strict=True):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 处理不匹配的情况
    model_state = model.state_dict()
    for key in list(state_dict.keys()):
        if key not in model_state:
            del state_dict[key]
    
    model.load_state_dict(state_dict, strict=False)
```

## 5. 日志系统 (Logger)

### 5.1 功能

- 训练过程日志记录
- TensorBoard 支持
- WandB 支持
- 可视化日志打印

### 5.2 日志级别

```python
class LogLevel:
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
```

### 5.3 使用方式

```python
runner_logger = Logger(
    log_dir='./logs',
    project_name='heltonx',
    use_tensorboard=True,
    use_wandb=False
)

# 训练日志
runner_logger.train_iter_log_printer(step, epoch, optimizer, losses)
runner_logger.train_epoch_log_printer(epoch, metrics)
```

## 6. 优化器与调度器

### 6.1 优化器封装

```python
def build_optimizer(model, optimizer_cfg):
    optimizer_type = optimizer_cfg.pop('type')
    if optimizer_type == 'AdamW':
        return torch.optim.AdamW(model.parameters(), **optimizer_cfg)
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(model.parameters(), **optimizer_cfg)
    # ...
```

### 6.2 学习率调度器

支持: CosineAnnealing, StepLR, MultiStepLR, PolynomialLR 等

```python
def build_scheduler(optimizer, scheduler_cfg):
    scheduler_type = scheduler_cfg.pop('type')
    if scheduler_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **scheduler_cfg
        )
    # ...
```

## 7. 通用工具 (Tools)

### 7.1 权重初始化

```python
def init_weights(module, init_type='xavier'):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            init_type == 'kaiming'  # 适用于 ReLU 激活
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
```

### 7.2 设备管理

```python
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    return data.to(device)
```

## 8. 核心 Runner

### 8.1 Runner 工作流程

```
初始化 → 数据加载 → 训练循环 → 评估循环 → 保存结果
   ↓         ↓           ↓           ↓
  模型     DataLoader  for epoch  for batch
```

### 8.2 Runner 核心属性

```python
class Runner:
    def __init__(self, model, train_loader, val_loader, optimizer, hooks, ...):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.hooks = hooks
        self.mode = 'train'  # or 'val', 'train_ddp'
```

## 9. 扩展核心框架

### 9.1 新增注册器

```python
# 在 utils/__init__.py 中添加
LOSSES = Register()

# 在具体模块中使用
@LOSSES.register
class MyLoss(nn.Module):
    pass
```

### 9.2 新增钩子

```python
class VisualizationHook(NecessaryHook):
    def hook_after_epoch(self, runner):
        # 保存可视化结果
        self.visualize_features(runner.model)
```

## 10. 配置示例

```yaml
# heltonx 核心配置
model:
  type: MLPNet
  backbone:
    type: TIMMBackbone
    model_name: resnet50
  head:
    type: MLPHead
    num_classes: 1000

optimizer:
  type: AdamW
  lr: 0.001
  weight_decay: 0.05

scheduler:
  type: CosineAnnealingLR
  T_max: 100

hooks:
  - type: NecessaryHook
    eval_interval: 1
    save_interval: 1
```
