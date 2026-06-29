"""OBB 评测流水线

与 detection/utils/eval_utils.py 的区别:
- 推理输出 xywhr (5维) 而非 xyxy (4维)
- mAP 计算使用 probiou (通过 obb_metrics)
- 不依赖 COCO API, 使用自定义评测逻辑
"""

import os
import shutil
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from detectionobb.utils.obb_metrics import compute_obb_map
from detectionobb.utils.obb_ops import regularize_rboxes, xywhr2xyxyxyxy
from heltonx.utils.register import EVALPIPELINES
from heltonx.utils.utils import to_device


def map_rboxes_to_origin_size(rboxes, orig_size, target_size):
    """将 [S, S] 图像上的 xywhr 预测映射回原始图像 [H, W] 尺寸

    Args:
        rboxes (np.ndarray): [N, 5] xywhr (基于 resize+padding 后的图像)
        orig_size (tuple): 原图尺寸 [H, W]
        target_size (int): 网络输入尺寸 S

    Returns:
        (np.ndarray): [N, 5] 映射回原图坐标的 xywhr
    """
    if len(rboxes) == 0:
        return rboxes

    H, W = orig_size
    S = target_size
    r = S / max(H, W)

    if H > W:
        new_w = int(W * r)
        pad_w = (S - new_w) / 2
        pad_h = 0
    else:
        new_h = int(H * r)
        pad_w = 0
        pad_h = (S - new_h) / 2

    rboxes = rboxes.copy().astype(float)
    # 去除 padding 偏移
    rboxes[:, 0] -= pad_w
    rboxes[:, 1] -= pad_h
    # 映射回原始比例 (cx, cy, w, h 需要缩放, angle 不变)
    rboxes[:, 0:4] /= r
    # 限制中心在图像内
    rboxes[:, 0] = np.clip(rboxes[:, 0], 0, W)
    rboxes[:, 1] = np.clip(rboxes[:, 1], 0, H)

    return rboxes


@EVALPIPELINES.register
class OBBDetectionEvalPipeline:
    """OBB 检测评估流水线 (基于 probiou 的 mAP)"""
    def __init__(self):
        pass

    def __call__(self, runner, model=None):
        device = runner.device
        if model is None:
            model = runner.model
        model.eval()

        valid_dataloader = runner.valid_dataloader
        nc = model.nc
        img_size = model.img_size[0] if isinstance(model.img_size, list) else model.img_size

        all_preds = []
        all_gts = []

        with torch.no_grad():
            for batch_datas in tqdm(valid_dataloader, desc='OBB Evaluating'):
                batch_datas = to_device(batch_datas, device, non_blocking=True)
                imgs, rboxes_list, labels_list, img_ids, raw_sizes = batch_datas

                bs = imgs.shape[0]
                if bs == 1:
                    results = [model.infer(imgs)]
                else:
                    results = model.infer(imgs)

                for b in range(bs):
                    boxes_np, scores_np, classes_np = results[b]

                    if len(boxes_np) > 0:
                        # ★★★ 关键修复: 存到 CPU/numpy 而非 GPU!
                        #   原代码将所有 pred/gt 都存为 GPU tensor → 整个验证集在显存中累积 → OOM
                        pred_boxes = regularize_rboxes(torch.from_numpy(boxes_np).float()).numpy()
                    else:
                        pred_boxes = np.zeros((0, 5), dtype=np.float32)

                    pred_scores = scores_np if len(scores_np) > 0 else np.zeros(0, dtype=np.float32)
                    pred_classes = classes_np.astype(np.int64) if len(classes_np) > 0 else np.zeros(0, dtype=np.int64)

                    all_preds.append({
                        'boxes': pred_boxes, 'scores': pred_scores, 'classes': pred_classes,
                    })

                    gt_rb = rboxes_list[b].cpu()
                    gt_lb = labels_list[b].cpu()
                    if len(gt_rb) > 0:
                        gt_boxes = regularize_rboxes(gt_rb.float()).numpy()
                    else:
                        gt_boxes = np.zeros((0, 5), dtype=np.float32)
                    gt_classes = gt_lb.long().numpy() if len(gt_lb) > 0 else np.zeros(0, dtype=np.int64)

                    all_gts.append({
                        'boxes': gt_boxes, 'classes': gt_classes,
                    })

        results = compute_obb_map(all_preds, all_gts, nc, iou_thrs=[0.5])

        # ★ 先提取结果, 再清理 GPU 缓存
        evaluations = dict(val_map=results['mAP'], val_ap50=results['mAP_50'])
        per_class_ap = results['per_class_ap']

        # ★★★ 评估完成后清理 GPU 缓存, 防止 CUDA caching allocator 保留评估期间的显存
        del all_preds, all_gts, results
        torch.cuda.empty_cache()

        if per_class_ap:
            print('\n--- Per-class AP (OBB, IoU=0.5) ---')
            for cls_id, ap in per_class_ap.items():
                name = runner.valid_dataset.cat_names[cls_id] if cls_id < len(runner.valid_dataset.cat_names) else str(cls_id)
                print(f'  {name}: {ap:.4f}')

        return evaluations, "val_map"


@EVALPIPELINES.register
class OBBDetectionEvalPipelineDOTADevkit:
    """OBB 检测评估流水线 (基于 DOTA devkit, 使用 poly iou)

    使用 DOTA devkit 的 voc_eval 进行评估, 评估结果与官方 DOTA 一致。
    流程: 推理 → 坐标映射回原图 → Task1 格式保存 → voc_eval 逐类评估

    ★ 优雅降级: 若 _polyiou C++ 模块未编译, 自动回退到 probiou 评估流水线
    """

    def __init__(self, temp_dir='eval_tmp'):
        self.temp_dir = temp_dir

    def _try_import_voc_eval(self):
        """尝试导入 DOTA devkit 的 voc_eval, 失败时返回 None"""
        import sys
        devkit_path = os.path.join(os.path.dirname(__file__), '../DOTA_devkit')
        if devkit_path not in sys.path:
            sys.path.insert(0, devkit_path)
        try:
            from dota_evaluation_task1 import voc_eval
            return voc_eval
        except (ImportError, ModuleNotFoundError) as e:
            print(f'\n[WARNING] DOTA devkit _polyiou 模块未编译, '
                  f'回退到 probiou 评估: {e}')
            print('[TIP] 如需使用 DOTA devkit 评估, 请编译 _polyiou:')
            print('      cd detectionobb/DOTA_devkit && python setup.py build_ext --inplace')
            return None

    def __call__(self, runner, model=None):
        # ★ 检查 _polyiou 是否可用, 不可用时自动回退
        voc_eval = self._try_import_voc_eval()
        if voc_eval is None:
            fallback = OBBDetectionEvalPipeline()
            return fallback(runner, model)
        device = runner.device
        if model is None:
            model = runner.model
        model.eval()

        valid_dataloader = runner.valid_dataloader
        nc = model.nc
        img_size = model.img_size[0] if isinstance(model.img_size, list) else model.img_size
        cat_names = runner.valid_dataset.cat_names
        label_dir = runner.valid_dataset.label_dir

        # 创建临时目录保存 detection 结果文件
        det_save_dir = os.path.join(runner.log_dir, self.temp_dir)
        os.makedirs(det_save_dir, exist_ok=True)

        # 每个类别的检测结果: dict[classname] = list of (img_name, score, poly_8)
        class_dets = {name: [] for name in cat_names}

        with torch.no_grad():
            for batch_datas in tqdm(valid_dataloader, desc='OBB DOTADevkit'):
                batch_datas = to_device(batch_datas, device, non_blocking=True)
                imgs, rboxes_list, labels_list, img_ids, raw_sizes = batch_datas

                bs = imgs.shape[0]
                if bs == 1:
                    results = [model.infer(imgs)]
                else:
                    results = model.infer(imgs)

                for b in range(bs):
                    boxes_np, scores_np, classes_np = results[b]
                    if len(boxes_np) == 0:
                        continue

                    # 映射回原始 patch 坐标
                    boxes_orig = map_rboxes_to_origin_size(
                        boxes_np, raw_sizes[b], img_size)
                    # regularize 到 [0, π/2)
                    boxes_t = torch.from_numpy(boxes_orig).float()
                    boxes_t = regularize_rboxes(boxes_t).numpy()

                    # xywhr → 8 顶点 polygon
                    corners = xywhr2xyxyxyxy(torch.from_numpy(boxes_t)).numpy()  # [N, 4, 2]

                    img_name = img_ids[b]  # 如 "P0000__1024__2472___1648"
                    for i in range(len(boxes_t)):
                        cls_idx = int(classes_np[i])
                        cls_name = cat_names[cls_idx]
                        score = float(scores_np[i])
                        poly = corners[i].reshape(-1)  # [8]: x1,y1,x2,y2,x3,y3,x4,y4
                        poly_str = ' '.join(f'{c:.2f}' for c in poly)
                        class_dets[cls_name].append(f'{img_name} {score:.4f} {poly_str}')

        # 写入 detection 文件 (Task1_{classname}.txt 格式)
        for cls_name, dets in class_dets.items():
            if not dets:
                continue
            save_path = os.path.join(det_save_dir, f'Task1_{cls_name}.txt')
            with open(save_path, 'w') as f:
                f.write('\n'.join(dets))

        # 生成 imagesetfile (所有 patch 名)
        imagesetfile = os.path.join(det_save_dir, 'imgnamefile.txt')
        # 从 class_dets 或 valid_dataloader 收集所有图像名
        all_img_names = set()
        for dets in class_dets.values():
            for line in dets:
                all_img_names.add(line.split(' ')[0])
        # 补充: 可能有些图没有检测结果但有 GT
        # 直接从 valid_dataset 获取所有 img_ids 更可靠
        all_full_names = []
        for idx in range(len(runner.valid_dataset)):
            img_path = runner.valid_dataset.img_files[idx]
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            all_full_names.append(img_id)

        with open(imagesetfile, 'w') as f:
            for name in all_full_names:
                f.write(name + '\n')

        # 逐类评估
        annopath = os.path.join(label_dir, '{:s}.txt')
        detpath = os.path.join(det_save_dir, 'Task1_{:s}.txt')

        class_aps = {}
        total_ap = 0.0
        class_count = 0
        for cls_name in cat_names:
            detfile = detpath.format(cls_name)
            if not os.path.exists(detfile):
                class_aps[cls_name] = 0.0
                continue

            rec, prec, ap, gts, dets = voc_eval(
                detpath, annopath, imagesetfile, cls_name,
                ovthresh=0.5, use_07_metric=True)
            ap_val = float(ap.mean())
            class_aps[cls_name] = ap_val
            total_ap += ap_val
            class_count += 1

        mAP = total_ap / class_count if class_count > 0 else 0.0

        # 打印
        print('\n--- Per-class AP (DOTADevkit, IoU=0.5) ---')
        for cls_name, ap_val in class_aps.items():
            print(f'  {cls_name}: {ap_val:.4f}')
        print(f'\n  mAP: {mAP:.4f}')

        # 清理临时文件
        shutil.rmtree(det_save_dir, ignore_errors=True)

        evaluations = dict(val_map=mAP, val_ap50=mAP)

        # ★ 评估完成后清理 GPU 缓存
        torch.cuda.empty_cache()

        return evaluations, "val_map"
