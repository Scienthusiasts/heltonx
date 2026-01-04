from typing import Any
import torch.nn as nn
import random
import torch
import numpy as np
import argparse
from functools import partial
import importlib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
import torch.distributed as dist






def multi_apply(func, *args, **kwargs) -> Any:
    """将函数应用到一组参数上(常用于多尺度预测)
      - 如果 func 每次返回多个值(tuple/list),则返回 tuple(list, list, ...)
      - 如果 func 每次返回单个值(如 Tensor),则直接返回 list

    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = list(map(pfunc, *args))
    # 没有输入时返回空列表
    if len(map_results) == 0:
        return []  

    first = map_results[0]
    # 如果每次返回的是 tuple/list -> 保持原 mmcv 行为：按位置聚合并返回 tuple(list,...)
    if isinstance(first, (tuple, list)):
        # 确保每个返回的元素长度一致会由 zip 处理
        return tuple(map(list, zip(*map_results)))
    else:
        # 单输出时，直接返回 list，便于直接赋值使用
        return list(map_results)
    



def dynamic_import_class(module_path, class_name='module_name', get_class=True):
    '''动态导入类
    '''
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if get_class:
        return getattr(module, class_name)
    else:
        return module
    



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    # 多卡
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    args = parser.parse_args()
    return args




def natural_key(s: str):
    # 如果类名都是数字，这会把 '10' 放到 '2' 之后；否则保持字典序
    try:
        return int(s)
    except Exception:
        return s.lower()




def to_device(batch, device, non_blocking=True):
    """递归地将所有tensor移动到指定device
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    elif isinstance(batch, dict):
        return {k: to_device(v, device, non_blocking) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device, non_blocking) for v in batch)
    else:
        # 非tensor类型直接返回
        return batch
    




def init_weights(model, init_type='he', mean=0, std=0.01):
    '''
    根据模型层的类型自动选择初始化方法。
    支持遍历 Conv, Linear, BatchNorm, Embedding 等不同结构。
    '''
    
    def _init_func(m):
        classname = m.__class__.__name__

        # 1. 处理卷积层 (Conv) 和 全连接层 (Linear)
        # 这些层通常需要使用 Kaiming, Xavier 或 Normal 初始化
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ConvTranspose2d)):
            if init_type == 'he':
                # Kaiming 初始化通常用于 ReLU 网络
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                # Xavier (Glorot) 初始化通常用于 Sigmoid/Tanh 网络
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, mean=mean, std=std)
            elif init_type == 'uniform':
                nn.init.uniform_(m.weight, a=-std, b=std)
            elif init_type == 'orthogonal':
                 nn.init.orthogonal_(m.weight)
            
            # 自动处理这些层的 bias (如果存在)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # 2. 处理归一化层 (BatchNorm, LayerNorm, GroupNorm, InstanceNorm)
        # 归一化层的权重(gamma)通常初始化为1，偏置(beta)初始化为0
        # 注意：这里不使用 init_type，因为 BN 层如果不初始化为 1/0 可能会导致模型无法训练
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                            nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        # 3. 处理嵌入层 (Embedding)
        # Embedding 层通常使用较小的正态分布或均匀分布
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=std)

        # 4. 处理 LSTM / GRU (RNN 类)
        # RNN 的初始化比较特殊，通常对循环权重使用正交初始化
        elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:  # Input-Hidden weights
                    if init_type == 'xavier':
                        nn.init.xavier_uniform_(param.data)
                    else: # Default orthogonal often works best for RNNs
                        nn.init.orthogonal_(param.data)
                elif 'weight_hh' in name: # Hidden-Hidden weights
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
                    # LSTM 的 forget gate bias 有时建议初始化为 1 (可选优化)
    
    # 使用 PyTorch 的 apply 方法递归遍历所有子模块
    print(f"Applying {init_type} initialization to {model.__class__.__name__}...")
    model.apply(_init_func)
    




def seed_everything(seed):
    '''设置全局种子
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def worker_init_fn(worker_id, seed, rank=0):
    """设置Dataloader的种子
       为每个 worker 设置了一个基于初始种子和 worker ID 的独特的随机种子, 
       这样每个 worker 将产生不同的随机数序列，从而有助于数据加载过程的随机性和多样性
    """
    # rank*1000 + worker_id 避免每一个子进程数据采样重复
    worker_seed = seed + rank*1000 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def accelerate_worker_init_fn(worker_id):
    """使用accelerate封装后使用的worker_init_fn, 乌苏再手动传seed和rank
    """
    worker_info = torch.utils.data.get_worker_info()
    # 每个 worker 的基础种子来源于主进程
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)



def set_dataloader_epoch(dataloader, epoch, base_seed):
    """保证 DataLoader resume 后的随机性与原训练一致
    Args:
        dataloader: DataLoader 对象
        epoch:      当前 epoch
        base_seed:  训练时固定的基础随机种子
    """
    # 处理 DistributedSampler
    # DDP时, 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果
    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    # 处理普通 DataLoader shuffle
    elif hasattr(dataloader, "sampler"):
        # 如果使用了 RandomSampler，说明启用了 shuffle
        from torch.utils.data import RandomSampler
        if isinstance(dataloader.sampler, RandomSampler):
            seed = base_seed + epoch
            seed_everything(seed)