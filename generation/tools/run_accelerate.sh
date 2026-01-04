#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=0,1,3
# export CUDA_VISIBLE_DEVICES=0,3
# export CUDA_VISIBLE_DEVICES=1,2

# training 
cd /mnt/yht/code/HeltonPretrain


# ddpm_unet_FlickrBreeds
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config /mnt/yht/code/HeltonPretrain/generation/configs/ddpm_unet_ddp.py


# ddpm_unet_DIOR
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/ddpm_unet_DIOR_ddp.py


# ddpm_unet_Celeba
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/ddpm_unet_Celeba_ddp.py

# cvae_Celeba
PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
    generation/tools/train_accelerate.py \
    --config generation/configs/cvae_Celeba_ddp.py


# 单卡(训练)
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_single_gpu.yaml ./generation/tools/train_accelerate.py --config generation/configs/cvae_Celeba.py
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_single_gpu.yaml ./generation/tools/train_accelerate.py --config generation/configs/vqvae_pixelcnn_Celeba.py
# 验证:
# python -m detection.tools.eval --config detection/configs/yolov5_coco_eval.py
# 推理:
# python -m detection.tools.test
