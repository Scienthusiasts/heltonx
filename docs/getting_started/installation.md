# 安装指南

## 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0 (如需 GPU 支持)

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/heltonx.git
cd heltonx
```

### 2. 创建虚拟环境（推荐）

```bash
conda create -n heltonx python=3.10
conda activate heltonx
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```python
import torch
import heltonx

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 可选依赖

### 用于特定模块

```bash
# 目标检测
pip install mmdetection  # COCO API

# LLM 模块
pip install transformers accelerate tiktoken

# 生成模型
pip install diffusers peft

# 图像处理
pip install pillow opencv-python
pip install albumentations
```

### 开发依赖

```bash
pip install black isort flake8 pytest
```

## Docker 部署

### 使用 Docker

```bash
# 构建镜像
docker build -t heltonx:latest .

# 运行容器
docker run --gpus all -it heltonx:latest
```

### 使用 NVIDIA Docker

```bash
docker run --gpus all -it -v $(pwd):/workspace heltonx:latest
```

## 常见问题

### 1. CUDA 版本不匹配

如果遇到 CUDA 相关的编译错误，请确保 PyTorch 版本与 CUDA 版本匹配：

```bash
# CUDA 11.3
pip install torch==1.12.0+cu113 torchvision torchaudio+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.8
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Flash Attention 安装问题

某些模块需要 Flash Attention：

```bash
pip install flash-attn --no-build-isolation
```

### 3. 内存不足

如果 GPU 内存不足，可以：
- 减小 batch size
- 使用混合精度训练 (`torch.cuda.amp`)
- 使用梯度累积
