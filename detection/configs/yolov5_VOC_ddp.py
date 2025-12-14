train_ann_json_path = '/mnt/yht/data/VOC0712/VOC2007/Annotations/coco/train.json'
valid_ann_json_path = '/mnt/yht/data/VOC0712/VOC2007/Annotations/coco/test.json'
train_img_dir = '/mnt/yht/data/VOC0712/VOC2007/JPEGImages'
valid_img_dir = '/mnt/yht/data/VOC0712/VOC2007/JPEGImages'
# 类别名
cat_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
             "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
cat_maps = None

phi = 's'
anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]] 
anchors_mask=[[0,1,2], [3,4,5], [6,7,8]]
nc = len(cat_names)
mode = 'train_ddp'
seed = 42
log_dir = r'./log/yolov5_voc_train_ddp'
img_size = [640, 640]
epoch = 12 * 4
bs = 8
lr = 1e-3
warmup_lr = 1e-5
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 1
resume = None


'''模型配置参数'''
model_cfgs = dict(
    type="YOLOv5",
    nc=nc, 
    img_size=img_size, 
    anchors=anchors,
    anchors_mask=anchors_mask,
    load_ckpt=load_ckpt,
    nms_score_thr=0.01,
    nms_iou_thr=0.3, 
    nms_agnostic=False,
    backbone=dict(
        type="YOLOv5CSPDarknet",
        phi=phi,
        out_layers=[2,3,4],
        froze_backbone=False,
        load_ckpt=f'ckpts/yolo/cspdarknet_{phi}_v6.1_backbone.pth'
    ), 
    fpn=dict(
        type="YOLOv5PAFPN",
        phi=phi,
    ), 
    heads=dict(
        type="YOLOv5Head",
        phi=phi,
        nc=nc, 
        img_size=img_size, 
        anchors=anchors,
        anchors_mask=anchors_mask,
        label_smoothing=0,
        layers_num=3,
        cls_loss=dict(
            # type="FocalLoss",
            # reduction="mean",
            # gamma=2.0, 
            # alpha=0.25
            type="BCELoss",
            reduction="mean"
        ),
        box_loss=dict(
            type="IoULoss",
            iou_type='giou',
            xywh=True,
            reduction="none",
        ),
        obj_loss=dict(
            type="BCELoss",
            reduction="mean"
        ), 
        assigner=dict(
            type="YOLOv5Assigner",
            img_size=img_size, 
            anchors=anchors,
            anchors_mask=anchors_mask,
            threshold=4,
        )
    )
)
'''数据集配置参数'''
dataset_cfgs=dict(
    train_dataset_cfg=dict(
        type="COCODataset",
        nc=nc, 
        cat_names=cat_names,
        ann_json_path=train_ann_json_path, 
        img_dir=train_img_dir,
        img_size=img_size, 
        mode='train', 
        mosaic_p=0.5, 
        mixup_p=0.0,
        map=cat_maps
    ),
    valid_dataset_cfg=dict(
        type="COCODataset",
        nc=nc, 
        cat_names=cat_names,
        ann_json_path=valid_ann_json_path, 
        img_dir=valid_img_dir,
        img_size=img_size, 
        mode='valid', 
        map=cat_maps
    ),
    train_bs=bs,
    valid_bs=1,
    num_workers=8,
    train_shuffle=True,
    valid_shuffle=False
)
'''优化器配置参数'''
optimizer_cfgs=dict(
    type="AdamW",
    lr=lr,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
'''学习率衰减策略配置参数'''
scheduler_cfgs=dict(
    base_schedulers_cfgs=dict(
        type="StepLR",
        # 每间隔step_size个epoch更新学习率
        step_size=1,
        # 每次学习率变为原来的gamma倍
        gamma=lr_decay**(1/epoch),
    ),
    warmup_schedulers_cfgs=dict(
            type="WarmupScheduler",
            min_lr=warmup_lr,
            warmup_epochs=1
    )
)

'''任务特定的评估pipeline'''
eval_pipeline_cfgs = dict(
    type="DetectionEvalPipeline"
)