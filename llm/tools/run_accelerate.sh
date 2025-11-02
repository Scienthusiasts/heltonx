#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,3
# export CUDA_VISIBLE_DEVICES=1,2

# training 
cd /mnt/yht/code/HeltonPretrain



# pretrain_minimind
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     llm/tools/train_accelerate.py \
#     --config /data/yht/code/HeltonPretrain/llm/configs/minimind_pretrain.py

# sft_512_minimind
PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
    llm/tools/train_accelerate.py \
    --config /data/yht/code/HeltonPretrain/llm/configs/minimind_sft.py

