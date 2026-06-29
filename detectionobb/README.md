# DetectionOBB — YOLO26-OBB 有向目标检测

基于 YOLO26 的旋转框 (OBB) 检测复现，与官方 [Ultralytics OBB26](https://github.com/ultralytics/ultralytics) 架构完全一致。复用 `detection/` 目录的 YOLO26 backbone 和 FPN。

## 架构概览

```
detectionobb/
├── utils/
│   ├── obb_ops.py              # 旋转框坐标操作 (xywhr↔4顶点, dist2rbox, rbox2dist, regularize_rboxes)
│   ├── obb_iou.py              # 概率 IoU (batch_probiou, 基于 Bhattacharyya 距离)
│   ├── obb_metrics.py          # OBB mAP 评测 (probiou + COCO 101 点插值)
│   └── eval_utils_obb.py       # 评测流水线 (probiou 版 + DOTA devkit 版, 自动降级)
├── bbox_coders/
│   └── yolo26_obb_bbox_coder.py  # OBB 编解码 (ltrb 直接回归 + dist2rbox, reg_max=1)
├── assigners/
│   └── yolo26_obb_assigner.py    # 旋转框 TAL 分配器 (probiou + 多GT冲突过滤)
├── losses/
│   └── obb_loss.py               # OBB 损失 (box + cls + L1 + angle, o2m+o2o progressive)
├── models/
│   ├── dense_heads/
│   │   └── yolo26_obb_head.py    # OBB 多层检测头 (box/cls/angle 三分支, o2m+o2o 双头)
│   └── detectors/
│       └── yolo26_obb.py         # OBB 检测器 (复用 backbone + FPN)
├── datasets/
│   ├── dota_dataset.py           # DOTA / YOLO-OBB 数据集 (支持 dota_raw 和 yolo_obb 格式)
│   └── preprocess_obb.py         # OBB 增强预处理 (几何变换 + Mosaic + MixUp)
├── tools/
│   └── convert_dota_to_yolo_obb.py  # DOTA 格式转换
├── DOTA_devkit/                  # DOTA 官方评测工具 (可选, 需编译 _polyiou)
└── configs/
    └── yolo26obb_dota_ddp.py     # DOTA 训练配置
```

## 关键设计

### 检测头 (DualOBBConvHead)

与官方 OBB26 完全一致的多层 head 结构，box / cls / angle 三个独立分支:

| 分支 | 结构 | 输出通道 |
|------|------|----------|
| box_head | Conv(ch,c2,3)+BN+SiLU → Conv(c2,c2,3)+BN+SiLU → Conv2d(c2,4,1) | 4 (ltrb) |
| cls_head | DWConv(ch,ch,3)+BN+SiLU → Conv(ch,c3,1)+BN+SiLU → DWConv(c3,c3,3)+BN+SiLU → Conv(c3,c3,1)+BN+SiLU → Conv2d(c3,nc,1) | nc |
| angle_head | Conv(ch,c4,3)+BN+SiLU → Conv(c4,c4,3)+BN+SiLU → Conv2d(c4,1,1) | 1 (弧度) |

- **o2m + o2o 双头**: o2m (one-to-many) 用于训练梯度回传，o2o (one-to-one) 使用 `copy.deepcopy` + detach FPN 特征
- **角度输出**: 原始 logits (弧度)，无 sigmoid 变换 (与官方 OBB26 一致)
- **Bias 初始化**: box=2.0, cls=log(5/nc/(640/stride)²), angle=0.0

### 回归方式

- **ltrb 直接回归**: 无 sigmoid、无 DFL (reg_max=1)
- **框解码**: `dist2rbox(pred_ltrb, pred_angle, anchors)` → xywh (anchor 空间) → ×stride → 像素空间
- **角度**: 原始 logits，由 sin²(2Δθ_wrapped) 损失处理周期性

### 损失函数 (4 项 × 2 分支)

| 损失项 | 计算方式 | gain | 说明 |
|--------|----------|------|------|
| box | probiou 回归 (1-iou) × weight | 7.5 | weight = target_scores.sum(-1)[fg_mask] |
| cls | BCE (软标签, 无 pos_weight) | 0.5 | .sum() / target_scores.sum() |
| L1 | F.l1_loss (stride/imgsz 归一化) | 1.5 | reg_max=1 时 DFL 退化为 L1 |
| angle | sin²(2Δθ_wrapped) × 宽高比权重 | 1.0 | Δθ_wrapped = Δθ - round(Δθ/π)×π |

**双头组合**: `loss = loss_o2m × o2m_weight + loss_o2o × o2o_weight`
- Progressive 衰减: o2m_weight 从 0.8 线性衰减到 0.1，o2o_weight = 1.0 - o2m_weight

### 目标分配 (TAL)

- **select_candidates_in_gts**: 叉积投影法判断锚点是否在旋转矩形内
- **get_box_metrics**: align_metric = cls^α × probiou^β (o2m: α=1.0, β=6.0, topk=13)
- **select_topk_candidates**: 每 GT 选 topk 锚点 + 多 GT 冲突预过滤 (count>1 → 排除)
- **select_highest_overlaps**: 多 GT 竞争锚点分配给 probiou 最大的 GT

### IoU 计算

- 使用 `batch_probiou` (概率 IoU，基于 Bhattacharyya 距离)
- 将 OBB 建模为 2D 高斯分布，纯 PyTorch 张量运算
- 评估时支持 DOTA devkit (精确多边形 IoU)，`_polyiou` 不可用时自动降级到 probiou

## 使用方法

### 1. 数据准备

支持两种标注格式:

**方式 1: DOTA 原始格式** (推荐，无需转换)
```
# 每行: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
195.0 783.0 254.0 760.0 289.0 838.0 230.0 861.0 small-vehicle 0
```

**方式 2: YOLO OBB 格式** (需先用工具转换)
```bash
python -m detectionobb.tools.convert_dota_to_yolo_obb --dota_root /path/to/DOTA
```
```
# 每行: class_idx x1 y1 x2 y2 x3 y3 x4 y4 (归一化到 [0,1])
4 0.19 0.77 0.25 0.74 0.28 0.82 0.22 0.84
```

### 2. 配置

修改 `detectionobb/configs/yolo26obb_dota_ddp.py`:

```python
# 数据路径
train_img_dir = '/path/to/images'
train_label_dir = '/path/to/labels'
label_format = 'dota_raw'  # 或 'yolo_obb'

# 模型规模: n/s/m/l/x
phi = 's'

# 训练参数
img_size = [1024, 1024]
epoch = 48
bs = 4
lr = 1e-3
```

### 3. 训练

```bash
# Accelerate DDP 训练
accelerate launch detectionobb/tools/train_accelerate.py --config detectionobb/configs/yolo26obb_dota_ddp.py
```

### 4. 预训练权重

backbone + FPN 从水平检测权重迁移，head 随机初始化:
```python
backbone=dict(
    load_ckpt=f'ckpts/yolo26{phi}.pt'  # 自动映射 backbone + FPN 权重
)
```

### 5. 评估

配置文件中选择评估流水线:
```python
eval_pipeline_cfgs = dict(
    type="OBBDetectionEvalPipelineDOTADevkit",  # DOTA devkit (需编译 _polyiou)
    # type="OBBDetectionEvalPipeline",          # probiou (无需编译)
)
```

DOTA devkit 需编译 C++ 扩展:
```bash
cd detectionobb/DOTA_devkit && python setup.py build_ext --inplace
```
若 `_polyiou` 不可用，自动降级到 probiou 评估。

## 与官方 Ultralytics OBB26 的对比

| 特性 | 本实现 | 官方 Ultralytics |
|------|--------|-----------------|
| backbone | YOLO26Backbone (复用) | 同 |
| FPN | YOLO26PAFPN (复用) | 同 |
| 检测头 | DualOBBConvHead (多层 Conv+BN+SiLU) | OBB26 (cv2+cv3+cv4 多层 Conv+BN+SiLU) |
| 双头 | o2m + o2o (deep copy + detach) | one2many + one2one (deep copy + detach) |
| 回归方式 | ltrb 直接回归 (reg_max=1) | ltrb 直接回归 (reg_max=1) |
| 框解码 | dist2rbox (anchor 空间) | dist2rbox (同) |
| 角度输出 | 原始 logits (OBB26 一致) | 原始 (OBB26) / sigmoid (OBB) |
| L1 损失 | F.l1_loss (stride/imgsz 归一化) | F.l1_loss (同, 官方变量名 loss_dfl) |
| 角度损失 | sin²(2Δθ_wrapped) + 宽高比权重 | 同 |
| TAL | probiou + 多GT冲突过滤 | 同 |
| IoU | batch_probiou | batch_probiou (同) |
| mAP 插值 | COCO 101 点 | COCO 101 点 (同) |
| Progressive | o2m: 0.8→0.1 epoch衰减 | o2m: 0.8→0.1 step衰减 |
| 推理 | NMS-free (o2o + topk) | NMS-free (同) |
