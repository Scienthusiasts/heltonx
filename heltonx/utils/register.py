# coding=utf-8
"""
HeltonX 注册机制模块
提供自动注册类装饰器，用于动态注册和管理模型、数据集、优化器等模块
"""
import torch.nn as nn
import torch.utils.data.dataset as data


class Register(dict):
    """自动注册基类（继承自 Python 字典）

    可作为装饰器自动注册 nn.Module 或其他类，支持通过配置字典动态实例化
    """

    def __init__(self, *args, **kwargs):
        """初始化注册器

        注册器内部字典用于存储已注册的类对象：
        - key: 类名（字符串）
        - value: 类对象（未实例化）
        """
        super(Register, self).__init__(*args, **kwargs)

    def add_item(self, key, value):
        """注册核心方法

        Args:
            key (str): 注册名称（通常为类名）
            value: 要注册的类或函数对象

        Returns:
            注册的对象（支持装饰器链式调用）

        Raises:
            Exception: 如果 value 不可调用
        """
        if not callable(value):
            raise Exception(f"Error: {value} must be callable!")
        if key in self:
            print(f"\033[31mWarning:\033[0m {value.__name__} already exists and will be overwritten!")
        self[key] = value
        return value

    def register(self, target):
        """注册装饰器

        Args:
            target: 要注册的类/函数，或注册名称（字符串）

        Returns:
            如果 target 是可调用对象，返回注册后的对象；
            如果 target 是字符串，返回一个装饰器函数
        """
        # 传入的target是函数或类
        if callable(target):    
            return self.add_item(target.__name__, target)
        # 如果传入的是字符串，返回一个装饰器函数，用这个字符串作为注册名
        else:                   
            return lambda x : self.add_item(target, x)

    def build(self, key, *args, **kwargs):
        """根据 key 实例化对应的类

        Args:
            key (str): 注册名称（类名）
            *args: 传递给类构造器的位置参数
            **kwargs: 传递给类构造器的关键字参数

        Returns:
            实例化后的类对象

        Raises:
            KeyError: 如果 key 未注册
            TypeError: 如果实例化失败
        """
        if key not in self:
            available_modules = list(self.keys())
            raise KeyError(
                f"'{key}' is not registered!\n"
                f"Available modules: {available_modules}"
            )
        cls = self[key]
        try:
            return cls(*args, **kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate '{key}' with args={args}, kwargs={kwargs}\n"
                f"Error: {e}"
            ) from e

    def build_from_cfg(self, cfg: dict, **kwargs):
        """根据配置字典实例化模块

        支持递归处理嵌套配置，自动处理模块列表和元组

        Args:
            cfg (dict): 配置字典，格式: {"type": "MyLinear", "in_features": 10, "out_features": 5}
            **kwargs: 额外传入的参数（如 optimizer 的 params）

        Returns:
            实例化后的类对象

        Raises:
            ValueError: 如果 cfg 为 None
            TypeError: 如果 cfg 不是字典
            KeyError: 如果 cfg 缺少 'type' 键
        """
        if cfg is None:
            raise ValueError("cfg cannot be None")
        
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, got {type(cfg)}")
            
        if "type" not in cfg:
            raise KeyError("cfg must contain the key 'type'")

        # cfg.pop("type") 会修改原字典, 如果外部还有引用, 会产生错误, 因此用copy避免外部再次调用时已被更改
        cfg = cfg.copy()
        module_type = cfg.pop("type")

        # 递归实例化模块(因为有时候存在嵌套的情况，一个模块的初始化参数包含另一个模块)
        for k, v in cfg.items():
            # 子模块
            if isinstance(v, dict) and "type" in v:   
                cfg[k] = self.build_from_cfg(v)
            # 模块列表
            elif isinstance(v, list):                 
                cfg[k] = [self.build_from_cfg(i) if isinstance(i, dict) and "type" in i else i for i in v]
            # 元组（也支持递归处理）
            elif isinstance(v, tuple):
                cfg[k] = tuple([
                    self.build_from_cfg(i) if isinstance(i, dict) and "type" in i else i 
                    for i in v
                ])

        # 合并 cfg 和 额外参数 kwargs，kwargs 优先级更高
        return self.build(module_type, **cfg, **kwargs)



# 关键点：模块级单例注册器, 这样其他文件中直接import就行, 无需初始化
MODELS = Register()
DATASETS = Register()
OPTIMIZERS = Register()
SCHEDULERS = Register()
EVALPIPELINES = Register()