img_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/images'
label_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/1.5'
mode = 'train_ddp'
seed = 42
log_dir = r'./log/controlldm_dit_DOTA_train_ddp'
img_size = [256, 256]
mask_size = [32, 32]
latent_dim = 16
epoch = 1500
bs = 32
lr = 2e-4
warmup_lr = lr*1e-2
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 15
resume = None

DOTA_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 
    'basketball-court', 'storage-tank',  'soccer-ball-field', 
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
)

'''模型配置参数'''
model_cfgs = dict(
    type="MaskLDM",
    vae=dict(
        type='HFVAE',
        weight_dir='ckpts/hugging_face/vae-kl-f8-d16',
        latent_dim=latent_dim,
        down_scale=8,
    ),
    img_size=img_size,
    batch_size=bs,
    load_ckpt=load_ckpt,
    schedule_name="linear_beta_schedule",
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    loss_type='huber',
    # CFG configs
    cfg_drop_prob=0.15,  # 训练时丢弃 Mask 条件的概率 (CFG Dropout)
    cfg_scale=2.0,       # 推理时条件引导的强度 (CFG Scale)
    denoise_model=dict(
        type="DiT",
        in_channels=latent_dim + len(DOTA_CLASSES) + 1, 
        out_channels=latent_dim,
        depth=12, 
        hidden_size=768, 
        patch_size=2, 
        num_heads=12, 
        learn_sigma=False, 
        use_condition=False,
    )
)
'''数据集配置参数'''
dataset_cfgs=dict(
    train_dataset_cfg=dict(
        type="ImageMaskDataset",
        class_names=DOTA_CLASSES,
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,  
        mask_size=mask_size,      
        ori_img_size=[1024, 1024],
        img_mean=[0.5, 0.5, 0.5], 
        img_std=[0.5, 0.5, 0.5]
    ),
    valid_dataset_cfg=None,
    train_bs=bs,
    num_workers=8,
    train_shuffle=True
)


'''优化器配置参数'''
optimizer_cfgs=dict(
    type="AdamW",
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
    type="ControlGenerationEvalPipeline"
)