<div align='center'>
    <h1>HeltonX♾️</h1>
    <img src="https://github.com/Scienthusiasts/heltonx/blob/main/demo/logo.png"/>
    <p><em>致力于 0~1 实现通用深度学习框架，基于 Pytorch，支持各类下游任务，不断完善中 ~</em></p>
</div>

## 🚀框架优势

- `reader-friendly`：关键代码包含详细的注释(包括输入输出维度)，易上手阅读
- `beginner-friendly`：关键模块解耦实现，各个模块互不干扰，组织结构清晰
- `coding-friendly`：代码封装程度适中，避免过度封装，易上手修改
- `build-from-scratch`：框架底层直接基于Pytorch，不包含任何第三方上层框架，任何算法设计全流程实现


## ✒️设计逻辑
**框架设计逻辑： 通用(换一个任务, 当前的设计逻辑可以完全套用) / 专用(仅适用于当前任务)**

框架总体逻辑

```
HeltonX:
├─demo       (README展示相关)
├─ckpts      (预训练权重放在这里)
├─heltonx    (通用, 实现基本的train/eval pipeline)
├─pretrain   (专用, 自定义预训练任务)
├─detection  (专用, 自定义检测任务)
├─generation (专用, 自定义生成任务)
└─llm        (专用, 自定义对话大模型)
```

`heltonx` 逻辑

```
heltonx:  
├─configs
│  └─accelerate_yamls
│  ├─accelerate_ddp.yaml         (通用, Accelerator库ddp训练配置文件)
│  └─accelerate_single_gpu.yaml  (通用, Accelerator库单卡训练配置文件)
├─tools
│  ├─train.py            (通用, 训练pipeline)
│  ├─train_accelerate.py (通用, 基于Accelerator库的训练pipeline)
│  └─eval.py             (通用, 评估pipeline)
├─utils            
│  ├─utils.py            (通用, 一些可能用到的方法)
│  ├─wrappers.py         (通用, 一些包装器或抽象类)
│  ├─ckpts_utils.py      (通用, 权重load/save相关逻辑)
│  ├─log_utils.py        (通用, 日志记录逻辑)
│  ├─hooks.py            (通用, 钩子机制, 实现训练,评估时必用的方法)
│  ├─hooks_accelerate.py (通用, 钩子机制(调整适配accelerate), 实现训练,评估时必用的方法)
│  └─register.py         (通用, 注册机制)
├─optimization
│  ├─optimizers.py  (通用, 优化器)
│  └─schedulers.py  (通用, 学习率decay)
└─setup.py (安装脚本)
```

### [`pretrain` 逻辑](https://github.com/Scienthusiasts/heltonx/blob/main/pretrain/README.md)

###  [`detection` 逻辑](https://github.com/Scienthusiasts/heltonx/blob/main/detection/README.md)

###  [`llm` 逻辑](https://github.com/Scienthusiasts/heltonx/blob/main/llm/README.md)

###  `heltonx/utils/register.py`

- `MODELS`：注册nn.Module子类
- `DATASETS`：注册dataset.Dataset子类
- `OPTIMIZERS`：注册torch.optim.Optimizer子类
- `SCHEDULERS`：注册torch.optim.lr_scheduler子类
- `EVALPIPELINES`：注册任务特定的评估pipelines

### `heltonx/utils/hooks.py`

- `NecessaryHook.hook_after_batch`：记录/打印日志
- `NecessaryHook.hook_after_epoch`：**评估**+记录/打印日志+保存权重
  - 注：当使用DDP训练时, 这里只在主节点的模型上评估,其余结点什么也不做
- `NecessaryHook.hook_after_epoch`：**评估**+记录/打印日志

### `xxx/utils/eval_utils.py`

- `XxxEvalPipeline`：用于评估流程(Trainner中一个epoch后的评估 / Evaler中的评估)，传入runner实例用于获取评估相关参数

### `xxx/configs/*.py`  (配置文件)

- model_cfgs：模型相关配置参数
- dataset_cfgs：数据集相关配置参数
- optimizer_cfg：优化器相关配置参数
- scheduler_cfgs：学习率decay相关配置参数
- eval_pipeline_cfgs：任务特定的评估pipeline配置参数

### 其他

- 



## 🔧安装

```bash
conda create -n hx python=3.10
conda activate hx
pip install -r requirements.txt
cd heltonx/heltonx
pip install -e .
```




## 🔥训练

单卡训练 example (xxx指具体下游任务所在的文件夹)

```bash
# 根据具体需求修改config文件里相关配置参数, mode="train"
python ./xxx/tools/train.py --config pretrain/configs/xxxconfigs.py

# 或，使用accelerate库封装过的训练pipeline
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file tools/accelerate_single_gpu.yaml tools/train_accelerate.py --config xxx/configs/xxxconfigs.py
```

DDP 多卡训练 example (xxx指具体下游任务所在的文件夹)

```bash
# 根据具体需求修改config文件里相关配置参数, mode="train_ddp"
sh ./xxx/tools/run.sh
# 或，使用accelerate库封装过的训练pipeline
sh ./xxx/tools/run_accelerate.sh
```



## 🔥评估

```bash
python ./xxx/tools/eval.py --config xxx/configs/xxxconfigs.py
```



## 🔥推理 (待进一步完善)

```bash
python ./xxx/tools/test.py 
```





## 📃更新日志
|     时间     |                             详细                             |
| :----------: | :----------------------------------------------------------: |
| `2025/09/24` | ✅ 代码重构  ✨ **支持注册机制(动态配置自定义参数文件), hook机制**  ✨ **支持Pytorch DDP 分布式训练**  ✅ 自动日志记录, 支持 `tensorboard`  ✅ resume 中断恢复训练  ✅ 支持 scheduler (学习率decay)  ✅ 兼容 `timm` 支持的 Backbone模型，可灵活更换 Backbone  ✅ 兼容 `Albentation` 图像增强方法  ✅ 支持混淆矩阵, accuracy, f1score, mAP 等评估指标  ✅ 深度学习 Hello Word -> 支持图像分类任务  ➡️ 各种表征学习自监督方法 (完善中...) |
| `2025/09/25` | ✅ 添加 openai-clip，qihoo360-fgclip VFM、 clip-zero-shot 分类模型 |
| `2025/09/26` |                    ✅ 添加 timm DINOv3 VFM                    |
| `2025/09/29` |                     ✅ 添加 VFMs 蒸馏损失                     |
| `2025/10/4`  |             ✅ 添加基于 CLIP 蒸馏的多任务分类模型             |
| `2025/10/11` | ➡️ 支持生成任务`./generation`, 支持 DDPM/DDIM (开发中，目前代码逻辑还不够清晰) |
| `2025/10/12` |       ✨ **支持 `Accelerate`(一键 DDP、混合精度训练)**        |
| `2025/10/21` | ✅ 支持检测任务`./detection`，支持 FCOS 检测器，COCO 格式数据读取 |
| `2025/11/1`  |                ➡️ 支持 LLM 训练和推理 (开发中)                |
| `2025/11/21` |                ➡️ 支持 VLM 训练和推理 (开发中)                |
| `2026/03/24` |                ✅ 支持 LDM（Latent Diffusion）                |
| `2026/04/15` |                ✅ 支持 DiT ✅ 支持 Flow 生成范式                |


## ➡️TODO
任务拓展 (将支持更多下游任务, 检测, 分割, 生成...)

