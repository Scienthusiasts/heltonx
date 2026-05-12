# coding=utf-8
"""
HeltonX 包装器模块
提供模型和数据加载的封装功能，如不保存包装器、DDP 安全数据集等
"""
import torch.nn as nn
import random
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
import torch.distributed as dist







class NoSaveWrapper(nn.Module):
    """不保存权重的模块包装器

    使用此包装器包装的 nn.Module 模块，在保存权重时不会保存该模块的权重
    常用于蒸馏模型中不需要保存的 teacher 模型

    注意：在 DDP 环境下，state_dict 会被 DDP 内部使用，因此需要特殊处理
    """

    def __init__(self, module: nn.Module):
        """初始化 NoSaveWrapper

        Args:
            module (nn.Module): 要包装的模块
        """
        super().__init__()
        self.module = module
        self._no_save = True  # 标记为不需要保存的模块

    def forward(self, *args, **kwargs):
        """前向传播

        Args:
            *args: 位置参数，传递给被包装模块
            **kwargs: 关键字参数，传递给被包装模块

        Returns:
            被包装模块的输出
        """
        return self.module(*args, **kwargs)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """返回空字典，不保存内部模块的权重

        Args:
            *args: 位置参数
            destination: 目标字典
            prefix: 键名前缀
            keep_vars: 是否保留变量

        Returns:
            空字典（保留 DDP 所需的内部状态）
        """
        # 调用父类的 state_dict 获取基础结构
        result = super().state_dict(destination, prefix, keep_vars)
        # 清空内部模块的权重
        result.clear()
        return result

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """重写此方法以阻止保存内部模块的参数

        Args:
            destination: 目标字典
            prefix: 键名前缀
            keep_vars: 是否保留变量
        """
        pass  # 不保存任何内容

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """重写此方法以阻止加载内部模块的参数

        Args:
            state_dict: 状态字典
            prefix: 键名前缀
            local_metadata: 本地元数据
            strict: 是否严格检查
            missing_keys: 缺失的键列表
            unexpected_keys: 多余的键列表
            error_msgs: 错误消息列表
        """
        # 清理相关的 state_dict 项，但不加载到内部模块
        keys_to_remove = [k for k in state_dict.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del state_dict[key]  







class DDPSafeDataset:
    """DDP 安全数据集加载器

    在分布式训练环境中安全地加载数据集，确保所有进程使用相同的数据
    """

    def ddp_safe_load(self, load_fn, verbose=True):
        """DDP 安全的数据广播

        在分布式训练时，只在主进程计算广播数据，然后传递给其他进程。
        在非分布式训练时，直接调用 load_fn 获取数据。

        注意：load_fn 返回的应为一个 dict，其中 value 必须是可序列化为 torch.Tensor 的数据。
        COCO API 对象等复杂对象不应通过此方法传递，应在各 rank 独立初始化。

        Args:
            load_fn (callable): 数据加载函数，返回 {'key': value} 字典
            verbose (bool, optional): 是否打印加载信息，默认 True

        Returns:
            加载的数据对象（所有 rank 一致）
        """
        if not dist.is_initialized():
            if verbose:
                print("[Single] Loading dataset directly...")
            return load_fn()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            if verbose:
                print(f"[Rank 0] Computing dataset data ...")
            data = load_fn()
        else:
            data = None

        # 同步所有 rank，确保 rank 0 已完成计算
        dist.barrier()

        # ---- 各字段广播策略 ----
        # 'filter_img_inds': list[int]  — 唯一需要广播的大数据（过滤后的 id 列表）
        # 'img_inds':         list[int]  — 也是 list，但通常很小
        # 'dataset_num':      int         — 直接广播标量
        # 'coco':             忽略 — 各 rank 独立初始化

        if rank == 0:
            result = data
        else:
            result = {}

        # 确定 broadcast 使用的设备 (NCCL 后端需要 GPU tensor)
        broadcast_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 广播 dataset_num（标量 tensor）
        num = torch.tensor([0], dtype=torch.long, device=broadcast_device)
        if rank == 0:
            num[0] = data.get('dataset_num', 0)
        dist.broadcast(num, src=0)
        if rank != 0:
            result['dataset_num'] = num.item()

        # 广播 img_inds（list[int]）
        if rank == 0:
            img_inds_list = data.get('img_inds', [])
            img_inds_len = len(img_inds_list)
        else:
            img_inds_len = 0
            img_inds_list = []

        len_tensor = torch.tensor([img_inds_len], dtype=torch.long, device=broadcast_device)
        dist.broadcast(len_tensor, src=0)
        img_inds_len = len_tensor.item()

        if rank == 0:
            img_inds_tensor = torch.tensor(img_inds_list, dtype=torch.long, device=broadcast_device)
        else:
            img_inds_tensor = torch.empty(img_inds_len, dtype=torch.long, device=broadcast_device)
        dist.broadcast(img_inds_tensor, src=0)
        if rank != 0:
            result['img_inds'] = img_inds_tensor.tolist()

        # 广播 filter_img_inds（list[int]）
        if rank == 0:
            f_inds_list = data.get('filter_img_inds', [])
            f_inds_len = len(f_inds_list)
        else:
            f_inds_len = 0
            f_inds_list = []

        len_tensor2 = torch.tensor([f_inds_len], dtype=torch.long, device=broadcast_device)
        dist.broadcast(len_tensor2, src=0)
        f_inds_len = len_tensor2.item()

        if rank == 0:
            f_inds_tensor = torch.tensor(f_inds_list, dtype=torch.long, device=broadcast_device)
        else:
            f_inds_tensor = torch.empty(f_inds_len, dtype=torch.long, device=broadcast_device)
        dist.broadcast(f_inds_tensor, src=0)
        if rank != 0:
            result['filter_img_inds'] = f_inds_tensor.tolist()

        # 同步结束
        dist.barrier()
        return result