"""
HeltonX 优化器模块
自动注册 PyTorch 原生优化器
"""
import torch.optim as optim
from torch import nn
import inspect

from heltonx.utils.register import OPTIMIZERS

# 批量注册 torch.optim 里的常见优化器
# 支持的优化器: ASGD, Adadelta, Adagrad, Adam, AdamW, Adamax,
# LBFGS, NAdam, Optimizer, RAdam, RMSprop, Rprop, SGD, SparseAdam
for name, obj in inspect.getmembers(optim, inspect.isclass):
    # 只注册 Optimizer 子类
    if issubclass(obj, optim.Optimizer):
        OPTIMIZERS.add_item(name, obj)
