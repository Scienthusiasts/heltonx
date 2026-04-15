trainset_path = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/1.5'
mode = 'train_ddp'
seed = 42
log_dir = r'./log/ddpm_unet_DOTAMask_train_ddp'
img_size = [32, 32]
dim = 128
epoch = 1500
bs = 64
lr = 2e-4
warmup_lr = lr*1e-2
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 20
resume = None

DOTA_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 
    'basketball-court', 'storage-tank',  'soccer-ball-field', 
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
)

'''模型配置参数'''
model_cfgs = dict(
    type="DDPM",
    img_size=img_size,
    batch_size=bs,
    load_ckpt=load_ckpt,
    schedule_name="linear_beta_schedule",
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    loss_type='huber',
    denoise_model=dict(
        type="UNet",
        input_dim=16,
        output_dim=16,
        # 配置 encoder / decoder 每一层的通道数
        layer_dims=[dim*1, dim*1, dim*2, dim*4],
    )
)
'''数据集配置参数'''
dataset_cfgs=dict(
    train_dataset_cfg=dict(
        type="DOTAOBBMaskDataset",
        class_names=DOTA_CLASSES,
        label_dir=trainset_path,
        img_size=img_size,        
        ori_img_size=[1024, 1024] 
    ),
    valid_dataset_cfg=None,
    train_bs=bs,
    num_workers=8,
    train_shuffle=True
)
'''优化器配置参数'''
optimizer_cfgs=dict(
    type="Adam",
    lr=lr
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
    type="MaskGenerationEvalPipeline"
)