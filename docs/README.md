# HeltonX 项目文档

## 文档概览

HeltonX 是一个基于 PyTorch 的通用深度学习框架，目标是 0~1 实现通用深度学习框架，支持各类下游任务。

### 核心特性

- **Reader-friendly**: 代码结构清晰，注释详尽，易于阅读和理解
- **Beginner-friendly**: 适合深度学习入门学习，从零实现核心组件
- **Coding-friendly**: 配置驱动，模块化设计，便于扩展和定制
- **Build-from-scratch**: 核心组件从零实现，不依赖过多第三方库

### 文档结构

```
docs/
├── README.md                    # 文档首页/概览
├── getting_started/             # 入门指南
│   ├── installation.md          # 安装指南
│   └── quickstart.md            # 快速开始
├── architecture/                # 架构设计
│   ├── overview.md             # 整体架构概览
│   ├── core_framework.md       # heltonx 核心框架
│   └── design_patterns.md     # 设计模式
├── modules/                     # 模块文档
│   ├── pretrain.md             # 预训练模块
│   ├── detection.md            # 目标检测模块
│   ├── generation.md           # 生成模型模块
│   └── llm.md                  # 大语言模型模块
├── api_reference/               # API 参考
│   ├── models.md
│   ├── datasets.md
│   └── hooks.md
├── advanced/                     # 高级主题
│   ├── distributed_training.md
│   └── custom_modules.md
└── contributing/               # 贡献指南
    ├── guidelines.md
    └── testing.md
```

### 快速链接

- [安装指南](getting_started/installation.md)
- [快速开始](getting_started/quickstart.md)
- [整体架构](architecture/overview.md)
- [核心框架详解](architecture/core_framework.md)
- [预训练模块](modules/pretrain.md)
- [目标检测模块](modules/detection.md)
- [生成模型模块](modules/generation.md)
- [大语言模型模块](modules/llm.md)
