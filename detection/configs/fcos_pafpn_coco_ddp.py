train_ann_json_path = '/mnt/yht/data/COCO/annotations/instances_train2017.json'
valid_ann_json_path = '/mnt/yht/data/COCO/annotations/instances_val2017.json'
train_img_dir = '/mnt/yht/data/COCO/train2017'
valid_img_dir = '/mnt/yht/data/COCO/val2017'
# 类别名
cat_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# COCO数据集需要类别id映射
cat_maps = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 
    24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29, 35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38, 44:39, 46:40, 
    47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49, 56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59, 
    67:60, 70:61, 72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73, 85:74, 86:75, 87:76, 88:77, 89:78, 90:79}

nc = 80
mode = 'train_ddp'
seed = 42
log_dir = r'./log/fcos_pafpn_coco_train_ddp'
img_size = [640, 640]
epoch = 12 * 3
bs = 8
lr = 2e-4
warmup_lr = 1e-5
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 1
resume = None


'''模型配置参数'''
model_cfgs = dict(
    type="FCOS",
    img_size=img_size,
    nc=nc, 
    load_ckpt=load_ckpt,
    nms_score_thr=0.05,
    nms_iou_thr=0.3, 
    nms_agnostic=False,
    bbox_coder=dict(
        type="FCOSBBoxCoder",
        strides=[8, 16, 32, 64, 128]
    ),
    backbone=dict(
        type="TIMMBackbone",
        model_name="resnet50.a1_in1k",
        pretrained=False,
        out_layers=[2,3,4],
        froze_backbone=False,
        load_ckpt='ckpts/backbone_resnet50.a1_in1k.pt'
    ), 
    fpn = dict(
        type="PAFPN",
        in_channels=[512, 1024, 2048],
        out_channel=256,
        num_extra_levels=2
    ),
    head=dict(
        type="FCOSHead",
        nc=nc, 
        in_channel=256, 
        cnt_loss=dict(
            type="BCELoss",
            reduction="mean"
        ), 
        cls_loss=dict(
            type="FocalLoss",
            reduction="none",
            gamma=2.0, 
            alpha=0.25
        ),
        reg_loss=dict(
            type="IoULoss",
            iou_type='giou',
            xywh=False,
            reduction="mean",
        ),
        assigner=dict(
            type="FCOSAssigner",
            img_size=img_size, 
            strides=[8, 16, 32, 64, 128], 
            limit_ranges=[[-1,64],[64,128],[128,256],[256,512],[512,999999]], 
            sample_radiu_ratio=1.5
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