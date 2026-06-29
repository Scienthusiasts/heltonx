#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,3
# export CUDA_VISIBLE_DEVICES=1,2

# training 
cd /mnt/yht/code/HeltonPretrain





# yolov26_dota
PYTHONPATH=. /mnt/yht/env/yht_pretrain/bin/accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_ddp.yaml \
    detectionobb/tools/train_accelerate.py \
    --config /mnt/yht/code/HeltonPretrain/detectionobb/configs/yolo26obb_dota_ddp.py



# 单卡(训练):
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_single_gpu.yaml ./detection/tools/train_accelerate.py --config detection/configs/yolov5_VOC.py
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. accelerate launch --config_file heltonx/configs/accelerate_yamls/accelerate_single_gpu.yaml ./detection/tools/train_accelerate.py --config detection/configs/yolov5_coco.py

# 验证:
# CUDA_VISIBLE_DEVICES=2 python -m detection.tools.eval --config detection/configs/yolov5_coco_eval.py

# 推理:
# python -m detection.tools.test
