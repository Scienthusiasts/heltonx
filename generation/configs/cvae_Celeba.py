trainset_path = r'/mnt/yht/data/face_256/celeba_256/train'
json_data_path = r'/mnt/yht/data/face_256/celeba256_captions_qwen3vlflash_structure.jsonl'
tokenizer_cfg_dir = r'/mnt/yht/code/HeltonPretrain/llm/tokenizer_configs/minimind2'

mode = 'train'
seed = 42
log_dir = r'./log/cvae_Celeba_train'
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
resume = 'log/cvae_Celeba_train_ddp/2026-01-21-01-45-07_train_ddp/train_epoch995.pt'
# 梯度裁剪策略
grad_clip=1.0

'''模型配置参数'''
model_cfgs = dict(
    type='CVAE',   
    img_size=img_size,
    input_dim=3,
    layer_dims=[dim, dim*2, dim*4, dim*6, dim*8, dim],  
    kld_weight=0.0002,
    latent_dim=dim*4,
    # 条件相关参数:
    condition_emb_dim=dim*4,
    vocab_size=6400,
    z_drop_prob=0.5, 
    z_drop_ratio=0.75,
    c_proj_model = dict(
       type='LightBERT',   
       emb_dim=dim*4, 
       n_layers=4, 
       heads=8, 
       max_len=192, 
       dropout=0.0
    ),
    load_ckpt=load_ckpt
)
'''数据集配置参数'''
dataset_cfgs=dict(
    train_dataset_cfg=dict(
        type="GenCaptionDataset",
        img_dir=trainset_path,
        json_data_path=json_data_path,
        tokenizer_cfg_dir=tokenizer_cfg_dir,
        img_size=img_size,
        max_length=192
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
    type="ConditionGenerationEvalPipeline"
)