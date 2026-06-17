train_ann_json_path = '/mnt/yht/data/COCO/annotations/instances_train2017.json'
valid_ann_json_path = '/mnt/yht/data/COCO/annotations/instances_val2017.json'
train_img_dir = '/mnt/yht/data/COCO/train2017'
valid_img_dir = '/mnt/yht/data/COCO/val2017'
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
cat_maps = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 13:11, 14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 
    24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29, 35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38, 44:39, 46:40, 
    47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47, 54:48, 55:49, 56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56, 63:57, 64:58, 65:59, 
    67:60, 70:61, 72:62, 73:63, 74:64, 75:65, 76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73, 85:74, 86:75, 87:76, 88:77, 89:78, 90:79}

nc = len(cat_names)
mode = 'train_ddp'
seed = 42
log_dir = r'./log/yolo26x_coco_train_ddp'
img_size = [640, 640]
epoch = 12 * 4
bs = 8
lr = 1e-3
warmup_decay = 1e-2
warmup_epochs = 1
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 1
resume = None

# 模型规模: n/s/m/l/x
phi = 'x'

'''模型配置参数'''
model_cfgs = dict(
    type="YOLO26",
    img_size=img_size,
    nc=nc,
    score_thr=0.001,
    max_det=300,
    load_ckpt=load_ckpt,
    backbone=dict(
        type="YOLO26Backbone",
        phi=phi,
        out_layers=[1,2,3],
        froze_backbone=False,
        load_ckpt=f'ckpts/yolo26{phi}.pt'
    ),
    fpn=dict(
        type="YOLO26PAFPN",
        phi=phi,
    ),
    head=dict(
        type="YOLO26Head",
        phi=phi,
        nc=nc,
        img_size=img_size,
        layers_num=3,
        loss_fn=dict(
            type="YOLO26Loss",
            nc=nc,
            img_size=img_size,
            box_gain=7.5,
            cls_gain=0.5,
            o2m_init=0.8,
            final_o2m=0.1,
            total_epochs=epoch,
            cls_loss=dict(type="YOLO26ClsLoss"),
            box_loss=dict(type="YOLO26BoxLoss"),
            assigner_o2m=dict(
                type="YOLO26Assigner",
                topk=13,
                alpha=1.0,
                beta=6.0,
                stride=[8, 16, 32],
            ),
            assigner_o2o=dict(
                type="YOLO26Assigner",
                topk=1,
                alpha=1.0,
                beta=6.0,
                stride=[8, 16, 32],
            ),
            bbox_coder=dict(
                type="YOLO26BBoxCoder",
                nc=nc,
            ),
        ),
    ),
)

'''数据集配置参数'''
dataset_cfgs = dict(
    train_dataset_cfg=dict(
        type="COCODataset",
        nc=nc,
        cat_names=cat_names,
        ann_json_path=train_ann_json_path,
        img_dir=train_img_dir,
        img_size=img_size,
        mode='train',
        mosaic_p=0.5,
        mixup_p=0.1,
        map=cat_maps,
    ),
    valid_dataset_cfg=dict(
        type="COCODataset",
        nc=nc,
        cat_names=cat_names,
        ann_json_path=valid_ann_json_path,
        img_dir=valid_img_dir,
        img_size=img_size,
        mode='valid',
        map=cat_maps,
    ),
    train_bs=bs,
    valid_bs=1,
    num_workers=4,
    train_shuffle=True,
    valid_shuffle=False,
)

'''优化器配置参数'''
optimizer_cfgs = dict(
    # type="SGD",
    # lr=lr,
    # momentum=0.937,
    # weight_decay=5e-4,
    # nesterov=True,
    type="AdamW",
    lr=lr,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# 梯度裁剪
grad_clip = 10.0

'''学习率衰减策略配置参数'''
scheduler_cfgs = dict(
    base_schedulers_cfgs=dict(
        type="CosineAnnealingLR",
        T_max=epoch - warmup_epochs,
        eta_min=lr * lr_decay,
    ),
    warmup_schedulers_cfgs=dict(
        type="WarmupScheduler",
        min_lr=lr * 0.01,
        warmup_epochs=warmup_epochs,
    ),
)

'''评估pipeline'''
eval_pipeline_cfgs = dict(
    type="DetectionEvalPipeline",
)
