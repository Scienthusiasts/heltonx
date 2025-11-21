#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,3
export CUDA_VISIBLE_DEVICES=1,2

# training 
cd /mnt/yht/code/HeltonPretrain



'''LLM'''
# pretrain_minimind
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     llm/tools/train_accelerate.py \
#     --config /data/yht/code/HeltonPretrain/llm/configs/minimind_pretrain.py

# sft_512_minimind
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     llm/tools/train_accelerate.py \
#     --config /data/yht/code/HeltonPretrain/llm/configs/minimind_sft512.py

# sft_2048_minimind
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     llm/tools/train_accelerate.py \
#     --config /data/yht/code/HeltonPretrain/llm/configs/minimind_sft2048.py

# sft_512_cot_minimind
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     llm/tools/train_accelerate.py \
#     --config /data/yht/code/HeltonPretrain/llm/configs/minimind_sft_cot.py

# dpo_1024_minimind
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     llm/tools/train_accelerate.py \
#     --config /data/yht/code/HeltonPretrain/llm/configs/minimind_dpo1024.py

# pretrain_512_Qwen3-0.6B
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     llm/tools/train_accelerate.py \
#     --config /data/yht/code/HeltonPretrain/llm/configs/qwen3_0.6b_pretrain.py

# sft_512_Qwen3-0.6B
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     llm/tools/train_accelerate.py \
#     --config /data/yht/code/HeltonPretrain/llm/configs/qwen3_0.6b_sft512.py


'''VLM'''
# sft_512_minimindv_clip
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     llm/tools/train_accelerate.py \
#     --config /data/yht/code/HeltonPretrain/llm/configs/minimindv_clip_pretrain512.py

# sft_512_minimindv_dinov3
PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
    llm/tools/train_accelerate.py \
    --config /data/yht/code/HeltonPretrain/llm/configs/minimindv_dinov3_pretrain512.py