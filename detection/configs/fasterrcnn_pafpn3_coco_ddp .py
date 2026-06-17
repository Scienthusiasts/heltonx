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

nc = len(cat_names)
mode = 'train_ddp'
seed = 42
log_dir = r'./log/faster_rcnn_pafpn3_coco_train_ddp'
img_size = [640, 640]
epoch = 12*3
bs = 8
lr = 2e-4
warmup_decay = 1e-2
warmup_epochs = 1
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 1
resume = None


'''模型配置参数'''
model_cfgs = dict(
    type="FasterRCNN",
    nc=nc,
    img_size=img_size,
    load_ckpt=load_ckpt,
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
        num_extra_levels=0
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        featmap_strides=[8, 16, 32],
        anchor_generator=dict(
            type="AnchorGenerator",
            strides=[8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            scales=[8],
        ),
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=(0., 0., 0., 0.),
            target_stds=(1., 1., 1., 1.),
        ),
        cls_loss=dict(
            type="BCELoss",
            reduction="mean",
        ),
        reg_loss=dict(
            type="SmoothL1Loss",
            beta=1.0,
            reduction="mean",
        ),
        num_samples=256,
        pos_fraction=0.5,
        nms_pre=2000,
        max_per_img=1000,
        nms_thr=0.7,
    ),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="RoIAlign",
            output_size=(7, 7),
            use_multilevel=True,
            featmap_strides=[4, 8, 16, 32, 64],
            aligned=True,
        ),
        bbox_head=dict(
            type="BBoxFCHead",
            in_channels=256,
            roi_feat_size=7,
            num_classes=nc,
            fc_out_channels=1024,
            num_fcs=2,
            with_cls=True,
            with_reg=True,
            reg_class_agnostic=False,
        ),
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=True,
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=(0., 0., 0., 0.),
            target_stds=(0.1, 0.1, 0.2, 0.2),
        ),
        cls_loss=dict(
            type="CrossEntropyLoss",
            reduction="mean",
        ),
        reg_loss=dict(
            type="SmoothL1Loss",
            beta=1.0,
            reduction="mean",
        ),
        num_samples=512,
        pos_fraction=0.25,
        score_thr=0.05,
        max_per_img=100,
        nms_thr=0.5,
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
    weight_decay=1e-4,
)
'''学习率衰减策略配置参数'''
scheduler_cfgs=dict(
    base_schedulers_cfgs=dict(
        type="CosineAnnealingLR",
        T_max=epoch - warmup_epochs,
        eta_min=lr * lr_decay,
    ),
    warmup_schedulers_cfgs=dict(
            type="WarmupScheduler",
            min_lr=lr * warmup_decay,
            warmup_epochs=warmup_epochs
    )
)

'''任务特定的评估pipeline'''
eval_pipeline_cfgs = dict(
    type="DetectionEvalPipeline"
)
