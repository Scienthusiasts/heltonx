'''VLM Pretrain主要是做Image Caption任务, 还不涉及具体细节的问答'''
json_data_path = '/mnt/yht/data/vlm/pretrain_data_qwen3vlflash_.jsonl'
imgs_dir = '/mnt/yht/data/vlm/pretrain_images'
img_size = [224, 224]
tokenizer_cfg_dir = '/data/yht/code/HeltonPretrain/llm/tokenizer_configs/minimind2'
vocab_size = 6400

mode = 'train_ddp'
seed = 42
log_dir = r'./log/vlm/minimindv_dinov3_pretrain512'
epoch = 12
bs = 32
lr = 4e-4
warmup_lr = 1e-5
lr_decay = 1e-1
load_ckpt = 'log/llm/minimind_sft2048/2025-11-03-22-01-31_train_ddp/last.pt'
vision_model_path = 'ckpts/hugging_face/DINOv3s'
vision_emb_dim = 384
log_interval = 50
eval_interval = 1
resume = None
# 梯度累加策略, bs等效于 bs*grad_accumulate
grad_accumulate=None
# 梯度裁剪策略
grad_clip=1.0



'''模型配置参数'''
model_cfgs = dict(
    type="PretrainVLM",
    vlm=dict(
        type="MiniMindForCausalVLM",
        load_ckpt=load_ckpt, 
        vision_encoder = dict(
            type='DINOv3',
            weight_dir=vision_model_path
        ),
        config=dict(
            v_hidden_size=vision_emb_dim,  # 视觉tokens初始维度
            hidden_size=768,               # 模型tokens维度
            num_hidden_layers=16,          # transformer 堆叠层数
            vocab_size=vocab_size,         # 使用的词表的大小(单词数)
            use_moe=False, 
            inference_rope_scaling=False,
        ),
    ),
    # 损失就是常规的多分类交叉熵损失(类别数为词表大小vocab_size)
    loss=dict(
        type="CELoss",
        reduction='none'
    )
)
'''数据集配置参数'''
dataset_cfgs=dict(
    train_dataset_cfg=dict(
        type="VLMPretrainDataset",
        imgs_dir=imgs_dir,
        img_size=img_size,
        json_data_path=json_data_path, 
        tokenizer_cfg_dir=tokenizer_cfg_dir, 
        max_length=512,
    ),
    valid_dataset_cfg=None,
    train_bs=bs,
    num_workers=8,
    train_shuffle=True,
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
# eval_pipeline_cfgs = None