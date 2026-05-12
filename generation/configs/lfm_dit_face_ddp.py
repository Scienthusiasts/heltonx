trainset_path = r'/mnt/yht/data/face_256'
mode = 'train_ddp'
seed = 42
log_dir = r'./log/lfm_dit_face_train_ddp'
img_size = [256, 256]
dim = 128
latent_dim = 16
epoch = 400
bs = 32
lr = 1e-4
warmup_decay = 1e-2
warmup_epochs = 2
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 4
resume = None


'''模型配置参数'''
model_cfgs = dict(
    type="LFM",
    vae=dict(
        type='HFVAE',
        weight_dir='ckpts/hugging_face/vae-kl-f8-d16',
        latent_dim=latent_dim,
        down_scale=8,
    ),
    img_size=img_size,
    batch_size=bs,
    sampling_steps=50,
    load_ckpt=load_ckpt,
    loss_type='huber',
    denoise_model=dict(
        type="DiT",
        in_channels=latent_dim, 
        depth=12, 
        hidden_size=384, 
        patch_size=2, 
        num_heads=6, 
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
    type="Adam",
    lr=lr
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
    type="GenerationEvalPipeline"
)