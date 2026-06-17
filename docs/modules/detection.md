# 目标检测模块 (Detection)

目标检测模块位于 `detection/` 目录，实现了两种主流检测器：FCOS (Anchor-Free) 和 YOLOv5 (Anchor-Based)。

## 1. 目录结构

```
detection/
├── models/
│   ├── backbones/      # 骨干网络
│   │   ├── resnet.py         # ResNet 骨干网络
│   │   └── cspdarknet.py     # CSPDarknet 骨干网络
│   ├── necks/          # 特征金字塔网络
│   │   ├── fpn.py            # Feature Pyramid Network
│   │   ├── pafpn.py          # Path Aggregation FPN
│   │   └── c2f_pafpn.py      # C2f PA-FPN
│   ├── heads/          # 检测头
│   │   ├── fcos_head.py      # FCOS 检测头
│   │   └── yolov5_head.py    # YOLOv5 检测头
│   ├── assigners/      # 正负样本分配
│   │   ├── fcos_assigner.py  # FCOS Assigner
│   │   └── yolov5_assigner.py # YOLOv5 Assigner
│   ├── bbox_codecs/    # 边界框编解码
│   │   ├── delta_xywh.py     # Delta 编解码
│   │   └── fd_mode.py       # FCOS 专用编解码
│   ├── detectors/      # 检测器
│   │   ├── fcos.py          # FCOS 检测器
│   │   └── yolov5.py        # YOLOv5 检测器
│   └── losses/         # 损失函数
│       ├── fcos_loss.py     # FCOS 损失
│       └── yolov5_loss.py   # YOLOv5 损失
├── datasets/           # 数据集
│   ├── coco.py              # COCO 数据集
│   └── voc.py               # VOC 数据集
├── augmentations/     # 数据增强
│   ├── mosaic.py       # Mosaic 增强
│   └── mixup.py        # MixUp 增强
├── inference/          # 推理相关
│   └── nms.py          # NMS 实现
├── train.py            # 训练脚本
└── README.md
```

## 2. 检测器架构

### 2.1 FCOS (Anchor-Free)

FCOS 是一种无锚点检测器，基于全卷积网络进行目标检测。

```
输入图像
    ↓
Backbone (ResNet)
    ↓
FPN (多尺度特征金字塔)
    ↓
FCOS Head (分类 + 中心度 + 回归)
    ↓
输出: cls_logits, cnt_logits, reg_preds
```

#### FCOS 检测器实现

```python
@MODELS.register
class FCOS(nn.Module):
    def __init__(self, backbone, fpn, head, bbox_coder, nms, 
                 assigner=None, loss=None):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.bbox_coder = bbox_coder
        self.nms = nms
        self.assigner = assigner
        self.loss_fn = loss
    
    def forward(self, datas, return_loss=True):
        if return_loss:
            # 训练模式
            images, targets = datas
            features = self.backbone(images)
            pyramid_features = self.fpn(features)
            cls_logits, cnt_logits, reg_preds = self.head(pyramid_features)
            
            # 损失计算
            losses = self.loss_fn(
                cls_logits, cnt_logits, reg_preds, targets, pyramid_features
            )
            return losses
        else:
            # 推理模式
            images = datas
            features = self.backbone(images)
            pyramid_features = self.fpn(features)
            cls_logits, cnt_logits, reg_preds = self.head(pyramid_features)
            return cls_logits, cnt_logits, reg_preds
    
    @torch.no_grad()
    def infer(self, image, vis_heatmap=False, save_vis_path=None):
        # 单图推理
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # 前向传播
        cls_logits, cnt_logits, reg_preds = self.forward(image, return_loss=False)
        
        # 解码边界框
        boxes = self.bbox_coder.decode(reg_preds, cls_logits)
        
        # NMS 后处理
        boxes, scores, classes = self.nms(boxes, cls_logits)
        
        return boxes, scores, classes
```

### 2.2 YOLOv5 (Anchor-Based)

YOLOv5 是一种基于锚点的单阶段检测器，采用 CSPDarknet 作为骨干网络。

```
输入图像
    ↓
Backbone (CSPDarknet)
    ↓
Neck (PAFPN / C2fPAFPN)
    ↓
YOLOv5 Head (多尺度检测头)
    ↓
输出: 3个尺度的预测
```

#### YOLOv5 检测器实现

```python
@MODELS.register
class YOLOv5(nn.Module):
    def __init__(self, backbone, neck, head, bbox_coder, nms):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.bbox_coder = bbox_coder
        self.nms = nms
    
    def forward(self, datas, return_loss=True):
        if return_loss:
            images, targets = datas
            features = self.backbone(images)
            pyramid_features = self.neck(features)
            predictions = self.head(pyramid_features)
            
            losses = self.compute_loss(predictions, targets)
            return losses
        else:
            images = datas
            features = self.backbone(images)
            pyramid_features = self.neck(features)
            return self.head(pyramid_features)
    
    @torch.no_grad()
    def infer(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        predictions = self.forward(image, return_loss=False)
        boxes = self.bbox_coder.decode(predictions)
        return self.nms(boxes)
```

## 3. Backbone (骨干网络)

### 3.1 ResNet

```python
@MODELS.register
class ResNet(nn.Module):
    def __init__(self, depth=50, num_stages=4, out_indices=(0,1,2,3)):
        super().__init__()
        # ResNet 实现，支持 ResNet-18/34/50/101
        self.stem = nn.Sequential(...)
        self.res_layers = nn.ModuleList([...])
        self.out_indices = out_indices
    
    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.res_layers):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
        return outputs
```

### 3.2 CSPDarknet

```python
@MODELS.register  
class CSPDarknet(nn.Module):
    def __init__(self, depth_multiple=1.0, width_multiple=1.0):
        super().__init__()
        self.stem = Conv(3, 32, 3, 1)
        self.stages = nn.ModuleList([...])  # CSPStage
        self.out_indices = (2, 3, 4)  # P3, P4, P5
    
    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outputs.append(x)
        return outputs
```

## 4. Neck (特征金字塔网络)

### 4.1 FPN (Feature Pyramid Network)

```python
@MODELS.register
class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, 1, 1) 
            for _ in in_channels
        ])
        self.extra_convs = nn.ModuleList([...])  # P6, P7
    
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        
        # 自顶向下融合
        laterals = [l_conv(inputs[i]) for i, l_conv in enumerate(self.lateral_convs)]
        
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2)
        
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        
        # 添加额外层级
        if self.extra_convs:
            outs.append(self.extra_convs[0](outs[-1]))
        
        return outs
```

### 4.2 PAFPN (Path Aggregation FPN)

在 FPN 基础上增加了自底向上的路径增强。

```python
@MODELS.register
class PAFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs):
        super().__init__()
        self.fpn = FPN(in_channels, out_channels, num_outs)
        self.downsample_convs = nn.ModuleList([...])
    
    def forward(self, inputs):
        # FPN 自顶向下
        outs = self.fpn(inputs)
        
        # 自底向上路径增强
        for i in range(len(outs) - 1, 0, -1):
            downsampled = self.downsample_convs[i-1](outs[i-1])
            outs[i-1] = torch.max(outs[i-1], downsampled)
        
        return outs
```

### 4.3 C2fPAFPN

带 C2f 模块的 PAFPN，用于 YOLOv5。

```python
@MODELS.register
class C2fPAFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs, num_blocks=3):
        super().__init__()
        self.reduce_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.c2f_blocks = nn.ModuleList([C2f(out_channels, out_channels, num_blocks) for _ in range(num_outs)])
    
    def forward(self, inputs):
        laterals = [reduce(inputs[i]) for i in range(len(inputs))]
        
        # 自顶向下
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(laterals[i], scale_factor=2)
        
        # 自底向上
        outs = [laterals[0]]
        for i in range(1, len(laterals)):
            downsampled = F.max_pool2d(laterals[i-1], kernel_size=3, stride=2, padding=1)
            outs.append(self.c2f_blocks[i](torch.cat([laterals[i], downsampled], dim=1)))
        
        return outs
```

## 5. Head (检测头)

### 5.1 FCOSHead

```python
@MODELS.register
class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_convs=4):
        super().__init__()
        self.cls_convs = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, 3, 1, 1) for _ in range(num_convs)
        ])
        self.cnt_convs = nn.Sequential(*[...]
        self.reg_convs = nn.Sequential(*[...]
        
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, 1, 1)
        self.cnt_logits = nn.Conv2d(in_channels, 1, 3, 1, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4, 3, 1, 1)
    
    def forward(self, features):
        cls_logits = []
        cnt_logits = []
        reg_preds = []
        
        for feat in features:
            cls_logits.append(self.cls_logits(self.cls_convs(feat)))
            cnt_logits.append(self.cnt_logits(self.cnt_convs(feat)))
            reg_preds.append(self.reg_pred(self.reg_convs(feat)))
        
        return cls_logits, cnt_logits, reg_preds
```

### 5.2 YOLOv5Head

YOLOv5 检测头，使用解耦头设计。

```python
@MODELS.register
class YOLOv5Head(nn.Module):
    def __init__(self, num_classes, anchors, in_channels):
        super().__init__()
        # 3 个尺度: P3, P4, P5
        self.cls_convs = nn.ModuleList([
            nn.Sequential(*[Conv(in_channels[i], in_channels[i], 3, 1) for _ in range(3)]) 
            for i in range(3)
        ])
        self.reg_convs = nn.ModuleList([...])
        
        self.cls_preds = nn.ModuleList([
            nn.Conv2d(in_channels[i], num_classes * 3, 1) for i in range(3)
        ])
        self.reg_preds = nn.ModuleList([
            nn.Conv2d(in_channels[i], 3 * 4, 1) for i in range(3)
        ])
        self.obj_preds = nn.ModuleList([
            nn.Conv2d(in_channels[i], 3 * 1, 1) for i in range(3)
        ])
    
    def forward(self, features):
        outputs = []
        for i, feat in enumerate(features):
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)
            
            cls_out = self.cls_preds[i](cls_feat)
            reg_out = self.reg_preds[i](reg_feat)
            obj_out = self.obj_preds[i](reg_feat)
            
            outputs.append([cls_out, reg_out, obj_out])
        
        return outputs
```

## 6. Assigner (正负样本分配)

### 6.1 FCOSAssigner

基于 FCOS 的样本分配策略。

```python
@MODELS.register
class FCOSAssigner:
    def assign(self, predictions, targets, images):
        """
        Args:
            predictions: [cls_logits, cnt_logits, reg_preds]
            targets: GT boxes and labels
        Returns:
            assign_result: 每个预测位置的分配结果
        """
        cls_logits, cnt_logits, reg_preds = predictions
        
        # 计算每个位置对应的 GT
        # 基于中心点和特征图尺度的匹配
        # 只分配在 GT 边界框内的正样本
        
        return assign_result
```

### 6.2 YOLOv5Assigner

基于 IoU 的样本分配策略。

```python
@MODELS.register
class YOLOv5Assigner:
    def assign(self, predictions, targets, images):
        """
        使用 Anchor 匹配和 IoU 计算进行样本分配
        """
        # 计算预测框与 GT 的 IoU
        # 使用 max IoU 确定正负样本
        # 返回分配结果
```

## 7. BBox Coder (边界框编解码)

### 7.1 DeltaXYWH

标准增量编解码。

```python
@MODELS.register
class DeltaXYWHBBoxCoder:
    def encode(self, bboxes, anchors):
        """编码: GT -> 预测增量"""
        dx = (bboxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
        dy = (bboxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
        dw = torch.log(bboxes[:, 2] / anchors[:, 2])
        dh = torch.log(bboxes[:, 3] / anchors[:, 3])
        return torch.stack([dx, dy, dw, dh], dim=-1)
    
    def decode(self, preds, scores):
        """解码: 预测增量 -> 真实坐标"""
        # 将增量转换为最终边界框
        ...
        return boxes
```

### 7.2 FCOSBBoxCoder

FCOS 专用的边界框编解码。

```python
@MODELS.register
class FCOSBBoxCoder:
    def encode(self, bboxes, points, strides):
        """基于点的编码"""
        # 计算相对于特征图上点的偏移
        ...
    
    def decode(self, reg_preds, cls_logits):
        """从中心点+偏移解码边界框"""
        # 获取预测的类别和回归值
        # 转换为边界框坐标
        ...
```

## 8. 损失函数

### 8.1 FCOSLoss

```python
class FCOSLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, loss_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight
    
    def forward(self, cls_logits, cnt_logits, reg_preds, targets, features):
        """
        计算 FCOS 总损失
        """
        # 1. 分类损失 (Focal Loss)
        cls_loss = self.focal_loss(cls_logits, targets)
        
        # 2. 中心度损失 (BCE)
        cnt_loss = self.bce_loss(cnt_logits, targets)
        
        # 3. 回归损失 (GIoU Loss)
        reg_loss = self.giou_loss(reg_preds, targets)
        
        return {
            'cls_loss': cls_loss * self.loss_weight,
            'cnt_loss': cnt_loss * self.loss_weight,
            'reg_loss': reg_loss * self.loss_weight,
            'total_loss': cls_loss + cnt_loss + reg_loss
        }
```

### 8.2 YOLOv5Loss

```python
class YOLOv5Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.box_loss = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, predictions, targets):
        """
        计算 YOLOv5 总损失
        """
        # 1. 置信度损失
        obj_loss = self.compute_obj_loss(predictions, targets)
        
        # 2. 分类损失
        cls_loss = self.compute_cls_loss(predictions, targets)
        
        # 3. 边界框损失
        box_loss = self.compute_box_loss(predictions, targets)
        
        return {
            'obj_loss': obj_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'total_loss': obj_loss + cls_loss + box_loss
        }
```

## 9. 数据增强

### 9.1 Mosaic

将 4 张图片拼接成一张。

```python
class Mosaic:
    def __init__(self, img_size=640):
        self.img_size = img_size
    
    def __call__(self, images, targets):
        # 随机选择 4 张图片
        # 拼接成 2x2 网格
        # 调整 GT boxes 坐标
        return mosaic_img, mosaic_targets
```

### 9.2 MixUp

混合两张图片和标签。

```python
class MixUp:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    
    def __call__(self, img1, target1, img2, target2):
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        # 混合标签
        return mixed_img, mixed_targets
```

## 10. NMS (非极大值抑制)

```python
class NMS:
    def __init__(self, score_threshold=0.05, nms_threshold=0.5, max_detections=100):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
    
    @torch.no_grad()
    def __call__(self, boxes, scores, classes):
        """
        Args:
            boxes: [N, 4] (x1, y1, x2, y2)
            scores: [N] 或 [N, num_classes]
            classes: [N] (可选)
        Returns:
            最终检测结果
        """
        # 1. 过滤低分框
        # 2. 按类别分组
        # 3. 每类独立 NMS
        # 4. 合并结果
        return final_boxes, final_scores, final_classes
```

## 11. 训练配置示例

```yaml
model:
  type: FCOS
  backbone:
    type: ResNet
    depth: 50
    out_indices: [1, 2, 3, 4]
  fpn:
    type: FPN
    in_channels: [512, 1024, 2048]
    out_channels: 256
  head:
    type: FCOSHead
    num_classes: 80
    num_convs: 4
  bbox_coder:
    type: FCOSBBoxCoder
  nms:
    type: NMS
    score_threshold: 0.05
    nms_threshold: 0.5

train_dataset:
  type: COCODataset
  img_root: /path/to/coco/images
  ann_file: /path/to/coco/annotations/instances.json

optimizer:
  type: SGD
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0005
```

## 12. 扩展方式

### 新增检测器

```python
@MODELS.register
class NewDetector(nn.Module):
    def __init__(self, backbone, neck, head, ...):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        ...
    
    def forward(self, datas, return_loss=True):
        # 自定义检测逻辑
        ...
    
    @torch.no_grad()
    def infer(self, image):
        # 自定义推理逻辑
        ...
```

### 新增 Head

```python
@MODELS.register
class NewHead(nn.Module):
    def forward(self, features):
        # 返回预测结果
        return predictions
```