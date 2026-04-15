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


# ddpm_unet_GCC
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/ddpm_unet_GCC_ddp.py


# ddpm_unet_DOTAMask
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/ddpm_unet_DOTAMask_ddp.py


# ddpm_dit_DOTAMask
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/ddpm_dit_DOTAMask_ddp.py


# cvae_Celeba
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/cvae_Celeba_ddp.py


# ldm_unet_face
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/ldm_unet_face_ddp.py


# ldm_dit_face
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/ldm_dit_face_ddp.py

# lfm_dit_face
PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
    generation/tools/train_accelerate.py \
    --config generation/configs/lfm_dit_face_ddp.py


# ldm_dit_schoolpersons
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/ldm_dit_schoolpersons_ddp.py


# class_ldm_dit_IN1K
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/classldm_dit_IN1K_ddp.py


# control_ldm_unet_DOTA
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/controlldm_unet_DOTA_ddp.py

# control_ldm_dit_DOTA
# PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
#     generation/tools/train_accelerate.py \
#     --config generation/configs/controlldm_dit_DOTA_ddp.py


# 单卡(训练)
# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_single_gpu.yaml ./generation/tools/train_accelerate.py --config generation/configs/vae_Celeba.py
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_single_gpu.yaml ./generation/tools/train_accelerate.py --config generation/configs/vqvae_pixelcnn_Celeba.py
# 验证:
# python -m generation.tools.eval --config generation/configs/ddpm_unet_Celeba_eval.py
# 推理:
# python -m detection.tools.test
