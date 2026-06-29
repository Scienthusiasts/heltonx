import torch
import numpy as np
import cv2
import math


def xywhr2xyxyxyxy(x):
    """将 xywhr 格式转换为 4 个顶点坐标

    Args:
        x (Tensor | ndarray): [N, 5] 或 [B, N, 5], 格式 (cx, cy, w, h, angle_radians)

    Returns:
        (Tensor | ndarray): [N, 4, 2] 或 [B, N, 4, 2], 4 个顶点坐标
    """
    is_torch = isinstance(x, torch.Tensor)
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if is_torch
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i:i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def xyxyxyxy2xywhr(x):
    """将 4 个顶点坐标转换为 xywhr 格式 (使用 cv2.minAreaRect)

    Args:
        x (Tensor | ndarray): [N, 8] 或 [N, 4, 2], 4 个顶点坐标

    Returns:
        (Tensor | ndarray): [N, 5], 格式 (cx, cy, w, h, angle_radians)
        角度范围 [-pi/4, 3pi/4)
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(points), -1, 2)
    rboxes = []
    for pts in points:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        theta = angle / 180 * np.pi
        if w < h:
            w, h = h, w
            theta += np.pi / 2
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi
        rboxes.append([cx, cy, w, h, theta])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def regularize_rboxes(rboxes):
    """规范化旋转框角度到 [0, pi/2)

    ★ 注意: 官方 ultralytics regularize_rboxes 使用 w>h 条件归一化到 [0, π),
    但我们使用 t%π>=π/2 条件归一化到 [0, π/2), 两种表示数学等价。
    保持 [0, π/2) 范围以确保与下游 eval/DOTA_devkit 代码兼容。

    Args:
        rboxes (Tensor): [N, 5] xywhr 格式

    Returns:
        (Tensor): [N, 5] 规范化后的旋转框 (angle in [0, π/2))
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    swap = t % math.pi >= math.pi / 2
    w_ = torch.where(swap, h, w)
    h_ = torch.where(swap, w, h)
    t = t % (math.pi / 2)
    return torch.stack([x, y, w_, h_, t], dim=-1)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """从距离分布和角度解码旋转框坐标

    与官方 ultralytics dist2rbox 一致:
    - lt, rb = pred_dist.split(2): 左上/右下距离
    - 中心偏移 = (rb - lt) / 2, 用 2D 旋转矩阵旋转
    - w = l + r, h = t + b

    Args:
        pred_dist (Tensor): [bs, N, 4] 预测的 ltrb 距离
        pred_angle (Tensor): [bs, N, 1] 预测角度 (弧度)
        anchor_points (Tensor): [N, 2] 锚点中心
        dim (int): 拆分维度

    Returns:
        (Tensor): [bs, N, 4] 解码后的 (x, y, w, h)
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def rbox2dist(anchor_points, rboxes, angle, dim=-1):
    """将旋转框编码为 ltrb 距离 (dist2rbox 的逆操作)

    与官方 ultralytics rbox2dist 一致:
    l = w/2 - xf, t = h/2 - yf, r = w/2 + xf, b = h/2 + yf

    Args:
        anchor_points (Tensor): [N, 2] 锚点中心
        rboxes (Tensor): [bs, N, 4] 旋转框 (x, y, w, h)
        angle (Tensor): [bs, N, 1] 角度

    Returns:
        (Tensor): [bs, N, 4] ltrb 距离
    """
    cos, sin = torch.cos(angle), torch.sin(angle)
    xy = rboxes[..., :2] - anchor_points
    xf, yf = xy.split(1, dim=dim)
    # 逆旋转: [xf, yf] = R^T(theta) @ [x, y]
    xf_orig = xf * cos + yf * sin
    yf_orig = -xf * sin + yf * cos
    w, h = rboxes[..., 2:3], rboxes[..., 3:4]
    target_l = w / 2 - xf_orig  # l = w/2 - xf (左距离)
    target_t = h / 2 - yf_orig  # t = h/2 - yf (上距离)
    target_r = w / 2 + xf_orig  # r = w/2 + xf (右距离)
    target_b = h / 2 + yf_orig  # b = h/2 + yf (下距离)
    return torch.cat([target_l, target_t, target_r, target_b], dim=dim)


def xywhr2xyxy(x):
    """将 xywhr 转换为轴对齐外接矩形 xyxy

    Args:
        x (Tensor): [N, 5] xywhr

    Returns:
        (Tensor): [N, 4] xyxy (轴对齐外接矩形)
    """
    corners = xywhr2xyxyxyxy(x)  # [N, 4, 2]
    x1 = corners[..., 0].min(dim=-1)[0]
    y1 = corners[..., 1].min(dim=-1)[0]
    x2 = corners[..., 0].max(dim=-1)[0]
    y2 = corners[..., 1].max(dim=-1)[0]
    return torch.stack([x1, y1, x2, y2], dim=-1)
