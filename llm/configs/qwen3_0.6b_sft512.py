json_data_path = '/data/yht/data/llm/sft_512.jsonl'
tokenizer_cfg_dir = '/data/yht/code/HeltonPretrain/llm/tokenizer_configs/qwen3'

mode = 'train_ddp'
seed = 42
log_dir = r'./log/llm/qwen3_0.6b_sft512'
epoch = 4
bs = 4
lr = 5e-7
warmup_lr = 4e-7
lr_decay = 5e-1
load_ckpt = '/data/yht/code/HeltonPretrain/ckpts/hugging_face/Qwen-0.6B'
log_interval = 50
eval_interval = 1
resume = None
# 梯度累加策略, bs等效于 bs*grad_accumulate
grad_accumulate=4
# 梯度裁剪策略
grad_clip=1.0



'''模型配置参数'''
model_cfgs = dict(
    type="SFTLLM",
    # 直接用hugging_face的权重
    llm=dict(
        type='AutoModelForCausalLM',
        weight_dir=load_ckpt
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
        type="SFTDataset",
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