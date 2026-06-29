"""OBB 评测指标

基于 probiou 的 mAP 计算, 与 ultralytics OBBMetrics 一致:
- IoU 计算: batch_probiou (概率 IoU)
- AP 插值: COCO 101 点
- 支持 mAP@0.5 和 mAP@0.5:0.95

★ 优化: 每张图预计算 probiou 矩阵一次, 供所有类复用
"""

import numpy as np
import torch
from detectionobb.utils.obb_iou import batch_probiou


def ap_per_class_obb(tp, conf, pred_cls, target_cls, iou_thr=0.5):
    """计算每个类别的 AP (基于 probiou)

    Args:
        tp (Tensor): [N] 是否为 TP (1/0), 按 conf 降序排列前
        conf (Tensor): [N] 置信度
        pred_cls (Tensor): [N] 预测类别
        target_cls (Tensor): [M] GT 类别
        iou_thr (float): IoU 阈值

    Returns:
        dict: {class_id: ap}
    """
    unique_classes = torch.unique(torch.cat([pred_cls, target_cls])) if len(target_cls) > 0 else torch.unique(pred_cls)
    ap_dict = {}

    for c in unique_classes:
        pred_mask = pred_cls == c
        target_mask = target_cls == c

        n_tp = tp[pred_mask].sum().item()
        n_gt = target_mask.sum().item()

        if n_gt == 0 or n_tp == 0:
            ap_dict[int(c)] = 0.0
            continue

        # 按 conf 降序
        conf_c = conf[pred_mask]
        tp_c = tp[pred_mask]
        order = conf_c.argsort(descending=True)
        tp_c = tp_c[order]

        # 累加 TP/FP
        tp_cum = tp_c.cumsum(0)
        fp_cum = (1 - tp_c).cumsum(0)

        recall = tp_cum / (n_gt + 1e-16)
        precision = tp_cum / (tp_cum + fp_cum + 1e-16)

        # COCO 101 点插值
        device = recall.device
        mrec = torch.cat([torch.tensor([0.0], device=device), recall, torch.tensor([1.0], device=device)])
        mpre = torch.cat([torch.tensor([0.0], device=device), precision, torch.tensor([0.0], device=device)])
        mpre = mpre.flip(0).cummax(0)[0].flip(0)

        x = torch.linspace(0, 1, 101, device=device)
        ap = torch_interp(x, mrec, mpre).mean()
        ap_dict[int(c)] = ap.item()

    return ap_dict


def torch_interp(x, xp, fp):
    """PyTorch 版 np.interp: 线性插值"""
    sort_idx = xp.argsort()
    xp = xp[sort_idx]
    fp = fp[sort_idx]
    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1] + 1e-16)
    idx = torch.searchsorted(xp, x, right=True) - 1
    idx = idx.clamp(0, len(slopes) - 1)
    return fp[idx] + slopes[idx] * (x - xp[idx])


def compute_obb_map(all_preds, all_gts, num_classes, iou_thrs=None):
    """计算 OBB mAP (优化版: 预计算 probiou 矩阵, 避免重复计算)

    Args:
        all_preds: list of dict, 每个 dict 包含:
            - 'boxes': [N, 5] xywhr
            - 'scores': [N]
            - 'classes': [N]
        all_gts: list of dict, 每个 dict 包含:
            - 'boxes': [M, 5] xywhr
            - 'classes': [M]
        num_classes (int): 类别数
        iou_thrs (list): IoU 阈值列表, 默认 [0.5]

    Returns:
        dict: mAP, mAP_50, per_class_ap
    """
    if iou_thrs is None:
        iou_thrs = [0.5]

    # === Step 1: 预计算每张图的 probiou 矩阵 (一次性) ===
    # ★★★ 优化: 只在计算 probiou 时临时用 GPU, 计算完立即转 CPU, 避免显存累积
    # 存储每张图的匹配信息: (predictions sorted by score, precomputed iou matrix)
    num_images = len(all_preds)
    image_data = []  # list of (pred_scores_sorted, pred_classes_sorted, pred_boxes_sorted,
                     #        gt_boxes, gt_classes, iou_matrix)

    # 选择计算设备: 有 GPU 时用 GPU 计算 probiou (更快), 结果存 CPU
    compute_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for preds, gts in zip(all_preds, all_gts):
        # ★ 支持 numpy 和 tensor 两种输入格式 (eval pipeline 改为存 numpy)
        def _to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(dtype)
            return x.to(dtype) if torch.is_tensor(x) else torch.tensor(x, dtype=dtype)

        pred_boxes = _to_tensor(preds['boxes'], torch.float32)
        pred_scores = _to_tensor(preds['scores'], torch.float32)
        pred_classes = _to_tensor(preds['classes'], torch.long)
        gt_boxes = _to_tensor(gts['boxes'], torch.float32)
        gt_classes = _to_tensor(gts['classes'], torch.long)

        N, M = len(pred_boxes), len(gt_boxes)

        if N == 0 or M == 0:
            image_data.append(None)
            continue

        # 按置信度降序排列 (一次排序, 所有类共用)
        order = pred_scores.argsort(descending=True)
        pred_scores_sorted = pred_scores[order]
        pred_classes_sorted = pred_classes[order]
        pred_boxes_sorted = pred_boxes[order]

        # ★ 一次性计算全部 N×M probiou 矩阵 (GPU加速, 结果转CPU节省显存)
        iou_matrix = batch_probiou(
            pred_boxes_sorted.to(compute_device),
            gt_boxes.to(compute_device)
        ).cpu()  # [N, M] 计算完立即转CPU, 释放GPU显存

        image_data.append({
            'scores': pred_scores_sorted,
            'classes': pred_classes_sorted,
            'iou_matrix': iou_matrix,
            'gt_classes': gt_classes,
        })

    # === Step 2: 对每个阈值和每个类进行匹配 ===
    all_ap = {}
    for thr in iou_thrs:
        class_aps = {}
        for c in range(num_classes):
            tp_list, conf_list = [], []
            n_gt_total = 0

            for img_data in image_data:
                if img_data is None:
                    continue

                pred_scores = img_data['scores']
                pred_classes = img_data['classes']
                gt_classes = img_data['gt_classes']
                iou_matrix = img_data['iou_matrix']

                # 当前类的预测和 GT
                pred_mask = pred_classes == c
                gt_mask = gt_classes == c

                N_c = pred_mask.sum().item()
                M_c = gt_mask.sum().item()
                n_gt_total += M_c

                if N_c == 0 or M_c == 0:
                    continue

                # 预计算 iou_matrix: [N, M], 取当前类的子集
                # 注意: pred_classes 已排序, iou_matrix 对应排序后的 preds
                gt_indices = gt_mask.nonzero(as_tuple=True)[0]  # [M_c]

                tp = torch.zeros(N_c, dtype=torch.float32, device=pred_scores.device)
                gt_matched = torch.zeros(M_c, dtype=torch.bool, device=pred_scores.device)

                # 遍历预测 (已按置信度降序)
                local_i = 0
                for global_i in range(len(pred_scores)):
                    if not pred_mask[global_i]:
                        continue
                    cls_val = pred_classes[global_i].item()
                    # 同类且未匹配的 GT (在 gt_indices 中查找)
                    # gt_classes[gt_indices] 全是 c
                    unmatched = ~gt_matched
                    if unmatched.sum() == 0:
                        break  # 所有 GT 已匹配

                    # 从预计算矩阵直接取 probiou 值
                    ious = iou_matrix[global_i, gt_indices][unmatched]
                    best_iou, best_local = ious.max(0)
                    if best_iou >= thr:
                        # 找到对应的 gt_indices 中的实际位置
                        local_unmatched = unmatched.nonzero(as_tuple=True)[0]
                        gt_matched[local_unmatched[best_local]] = True
                        tp[local_i] = 1.0
                    local_i += 1

                tp_list.append(tp)
                conf_list.append(pred_scores[pred_mask])

            if n_gt_total == 0 or len(tp_list) == 0:
                class_aps[c] = 0.0
                continue

            tp_cat = torch.cat(tp_list)
            conf_cat = torch.cat(conf_list)
            pred_cls_cat = torch.full_like(conf_cat, c, dtype=torch.long)
            target_cls_cat = torch.full((n_gt_total,), c, dtype=torch.long, device=conf_cat.device)

            ap_dict = ap_per_class_obb(tp_cat, conf_cat, pred_cls_cat, target_cls_cat, thr)
            class_aps[c] = ap_dict.get(c, 0.0)

        all_ap[thr] = class_aps

    # 计算 mAP
    mAP_50 = np.mean(list(all_ap[0.5].values())) if 0.5 in all_ap else 0.0
    if len(iou_thrs) > 1:
        all_mAPs = [np.mean(list(all_ap[thr].values())) for thr in iou_thrs]
        mAP = np.mean(all_mAPs)
    else:
        mAP = mAP_50

    return {
        'mAP': mAP,
        'mAP_50': mAP_50,
        'per_class_ap': all_ap.get(0.5, {}),
    }
