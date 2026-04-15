trainset_path = r'//mnt/yht/data/school_persons'
mode = 'train_ddp'
seed = 42
log_dir = r'./log/ldm_dit_schoolpersons_train_ddp'
img_size = [256, 256]
dim = 128
latent_dim = 16
epoch = 1000
bs = 32
lr = 2e-4
warmup_lr = lr*1e-2
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 10
resume = None


'''模型配置参数'''
model_cfgs = dict(
    type="LDM",
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
    denoise_model=dict(
        type="DiT",
        in_channels=latent_dim, 
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
        type="GenDataset",
        img_dir=trainset_path,
        img_size=img_size,
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
    type="GenerationEvalPipeline"
)