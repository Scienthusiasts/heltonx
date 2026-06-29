# coding=utf-8
"""OBB 评估脚本

使用训练好的模型在验证集上评估 mAP。

用法:
    python -m detectionobb.tools.eval --config detectionobb/configs/yolo26obb_dota_ddp.py
    python -m detectionobb.tools.eval --config detectionobb/configs/yolo26obb_dota_ddp.py \\
        --ckpt log/yolo26s_obb_dota_obb_train_ddp/xxx/last.pt --thr 0.01
"""
# 需要import才能注册
from detectionobb import *

import argparse
from heltonx.utils.utils import dynamic_import_class
from heltonx.tools.eval import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OBB Evaluation')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--ckpt', type=str, default=None, help='权重文件路径 (覆盖config中的load_ckpt)')
    parser.add_argument('--thr', type=float, default=None, help='推理得分阈值 (覆盖模型默认值)')
    args = parser.parse_args()

    config_path = args.config
    # 使用动态导入模块导入参数文件
    cargs = dynamic_import_class(config_path, get_class=False)

    model_cfgs = cargs.model_cfgs.copy()
    # 覆盖 checkpoint 路径
    if args.ckpt is not None:
        model_cfgs['load_ckpt'] = args.ckpt
    if args.thr is not None:
        model_cfgs['score_thr'] = args.thr

    print(f"Model: {model_cfgs['type']}, Ckpt: {model_cfgs.get('load_ckpt', 'None')}")

    # 初始化runner
    runner = Evaler(cargs.seed, cargs.log_dir, model_cfgs, cargs.dataset_cfgs)
    # 注册 Hook
    # 任务特定的评估pipeline
    eval_pipeline = EVALPIPELINES.build_from_cfg(cargs.eval_pipeline_cfgs)
    hook = NecessaryHook(eval_pipeline)
    runner.register_hook("after_eval", hook.hook_after_eval)
    runner.eval()
