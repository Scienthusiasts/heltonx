trainset_path = r'/mnt/yht/data/celeba_256'
mode = 'train'
seed = 42
log_dir = r'./log/vqvae_gatedpixelcnn_Celeba_train'
img_size = [256, 256]
dim = 64
epoch = 1000
bs = 32
lr = 5e-4
warmup_lr = lr*1e-2
lr_decay = 1e-1
load_ckpt = None
log_interval = 50
eval_interval = 5
resume = None
# 梯度裁剪策略
grad_clip=1.0

'''模型配置参数'''
model_cfgs = dict(
    type='VQVAE_PixelCNN',  
    pixelcnn_model = dict(
        type='GatedPixelCNN',
        n_layers=3,
        num_embeddings=512, 
        dim=dim*4, 
    ),
    vqvae_model = dict(
        type='VQVAE',   
        input_dim=3,
        layer_dims=[dim, dim*2, dim*4, dim*8, dim],  
        num_embeddings=512,
        embedding_dim=dim*4,
        img_size=img_size,
        encoder_loss_w=0.2,
        codebook_loss_w=1,
        load_ckpt=r'log/vqvae_Celeba_train/2025-12-25-20-25-06_train/110.pt',
    ),
)
'''数据集配置参数'''
dataset_cfgs=dict(
    train_dataset_cfg=dict(
        type="GenDataset",
        img_dir=trainset_path,
        img_size=img_size,
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
    type="GenerationEvalPipeline"
)