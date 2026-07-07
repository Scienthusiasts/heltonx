# DOTA 1.0 类别 (15 类) — 顺序即为类别索引
cat_names = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

# 数据路径 (需修改为实际路径)
# 方式1: YOLO OBB 格式 (归一化坐标, 需先用 convert_dota_to_yolo_obb.py 转换)
# train_img_dir = '/mnt/yht/data/DOTA/images/train'
# train_label_dir = '/mnt/yht/data/DOTA/labels/train'
# valid_img_dir = '/mnt/yht/data/DOTA/images/val'
# valid_label_dir = '/mnt/yht/data/DOTA/labels/val'
# label_format = 'yolo_obb'

# 方式2: DOTA 原始格式 (绝对像素坐标, 无需转换)
train_img_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/images'
train_label_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/1.0/trainval/annfiles'
valid_img_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/images'
valid_label_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/1.0/val/annfiles'
label_format = 'dota_raw'

# 模型规模: n/s/m/l/x
phi = 's'

nc = len(cat_names)
mode = 'train_ddp'
seed = 42
log_dir = rf'./log/yolo26{phi}_obb_dota_obb_train_ddp'
img_size = [1024,1024]
epoch = 100
bs = 4
lr = 2e-4
warmup_decay = 1e-2
warmup_epochs = 1
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 1
resume = None # 'log/yolo26s_obb_dota_obb_train_ddp/2026-06-30-08-03-27_train_ddp/train_epoch84.pt'



'''模型配置参数'''
model_cfgs = dict(
    type="YOLO26OBB",
    img_size=img_size,
    nc=nc,
    score_thr=0.05,
    max_det=300,
    load_ckpt=load_ckpt,
    backbone=dict(
        type="YOLO26Backbone",
        phi=phi,
        out_layers=[1, 2, 3],
        froze_backbone=False,
        load_ckpt=f'ckpts/yolo26{phi}.pt'
    ),
    fpn=dict(
        type="YOLO26PAFPN",
        phi=phi,
    ),
    head=dict(
        type="YOLO26OBBHead",
        phi=phi,
        nc=nc,
        img_size=img_size,
        layers_num=3,
        loss_fn=dict(
            type="YOLO26OBBLoss",
            nc=nc,
            img_size=img_size,
            box_gain=7.5,
            cls_gain=0.5,
            l1_gain=1.5,      # ★ L1 回归损失权重 (与官方 YOLO26 hyp.dfl 一致, reg_max=1)
            angle_gain=1.0,
            o2m_init=0.8,
            final_o2m=0.1,
            total_epochs=epoch,
            cls_loss=dict(type="YOLO26OBBClsLoss"),
            box_loss=dict(type="YOLO26OBBBoxLoss"),
            l1_loss=dict(type="YOLO26OBBL1Loss"),      # ★ L1 回归损失 (reg_max=1, 无 DFL)
            angle_loss=dict(type="YOLO26AngleLoss", lambda_val=3),
            assigner_o2m=dict(
                type="YOLO26OBBAssigner",
                topk=13,     # ★ 与官方 RotatedTaskAlignedAssigner 一致: topk=13
                alpha=1.0,   # ★ 与官方 RotatedTaskAlignedAssigner 一致: alpha=1.0
                beta=6.0,
                stride=[8, 16, 32],
            ),
            assigner_o2o=dict(
                type="YOLO26OBBAssigner",
                topk=1,
                alpha=1.0,   # ★ 与官方 RotatedTaskAlignedAssigner 一致: alpha=1.0
                beta=6.0,
                stride=[8, 16, 32],
            ),
            bbox_coder=dict(
                type="YOLO26OBBBBoxCoder",
                nc=nc,
                reg_max=1,   # ★ 与官方 YOLO26 一致: reg_max=1 → 无 DFL, 直接回归 ltrb
            ),
        ),
    ),
)

'''数据集配置参数'''
dataset_cfgs = dict(
    train_dataset_cfg=dict(
        type="DOTADataset",
        nc=nc,
        cat_names=cat_names,
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        img_size=img_size,
        mode='train',
        label_format=label_format,
        mosaic_p=0.5,
        mixup_p=0.1,
        filter_no_obb=True,
    ),
    valid_dataset_cfg=dict(
        type="DOTADataset",
        nc=nc,
        cat_names=cat_names,
        img_dir=valid_img_dir,
        label_dir=valid_label_dir,
        img_size=img_size,
        mode='valid',
        label_format=label_format,
        filter_no_obb=False,
    ),
    train_bs=bs,
    valid_bs=1,
    num_workers=4,
    train_shuffle=True,
    valid_shuffle=False,
)

'''优化器配置参数'''
optimizer_cfgs = dict(
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
    # type="OBBDetectionEvalPipeline",
    type="OBBDetectionEvalPipelineDOTADevkit",
)
