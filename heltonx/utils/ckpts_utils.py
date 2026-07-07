# 负责权重load 和save相关逻辑
import torch.nn as nn
import torch
import os
import types
import sys
import random
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class _DummyClass:
    """虚拟类，用于替代反序列化时未安装的第三方库类（如 ultralytics.nn.tasks.DetectionModel）

    当 pickle 反序列化 nn.Module 子类时，会调用 __setstate__ 传入模型状态字典。
    _DummyClass 需要能接收这些状态，并支持通过 state_dict() 提取张量参数。
    """
    def __init__(self, *args, **kwargs):
        self._state = {}

    def __setstate__(self, state):
        self._state = state

    def __getstate__(self):
        return self._state

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self._state.get(name, _DummyClass())

    def state_dict(self):
        """从 pickle 恢复的 nn.Module 状态中提取所有张量参数

        nn.Module.__getstate__ 返回的字典中：
        - '_parameters': OrderedDict of Parameter
        - '_buffers': OrderedDict of Buffer
        - '_modules': OrderedDict of sub-modules (递归)
        """
        return self._extract_tensors(self._state, prefix='')

    def _extract_tensors(self, state, prefix):
        """递归提取所有参数和缓冲区张量"""
        result = {}
        params = state.get('_parameters', {})
        if isinstance(params, dict):
            for k, v in params.items():
                if isinstance(v, torch.Tensor):
                    result[prefix + k] = v

        buffers = state.get('_buffers', {})
        if isinstance(buffers, dict):
            for k, v in buffers.items():
                if isinstance(v, torch.Tensor):
                    result[prefix + k] = v

        modules = state.get('_modules', {})
        if isinstance(modules, dict):
            for k, v in modules.items():
                if isinstance(v, _DummyClass):
                    result.update(v._extract_tensors(v._state, prefix=prefix + k + '.'))
                elif isinstance(v, nn.Module):
                    # 真实 nn.Module — 直接遍历 __dict__ 递归提取
                    # (不能用 .state_dict()，因为其子模块可能是 _DummyClass)
                    result.update(self._extract_tensors(v.__dict__, prefix=prefix + k + '.'))
                elif isinstance(v, dict):
                    result.update(self._extract_tensors(v, prefix=prefix + k + '.'))

        return result


class _FakeModule(types.ModuleType):
    """虚拟模块，对任意属性访问返回 _DummyClass 或嵌套的 _FakeModule

    用于注入 sys.modules，使 pickle 反序列化时无需真正安装第三方库。
    例如: sys.modules['ultralytics'] = _FakeModule('ultralytics')
    则 ultralytics.nn.tasks.DetectionModel 等任意路径都能解析成功。
    """
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []  # 使 Python import 机制将其视为 package，允许导入子模块

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return _DummyClass

    def __call__(self, *args, **kwargs):
        return _DummyClass(*args, **kwargs)


class _FakeModuleFinder:
    """Meta path finder，拦截指定前缀的所有 import 请求并返回 _FakeModule

    与直接注入 sys.modules 不同，finder 能处理任意深度的子模块导入，
    例如 ultralytics.nn.tasks.DetectionModel 需要 import ultralytics.nn.tasks，
    finder 会在 import 时自动创建并缓存虚拟子模块。
    """
    def __init__(self, prefix):
        self.prefix = prefix  # 如 'ultralytics'

    def find_module(self, fullname, path=None):
        if fullname == self.prefix or fullname.startswith(self.prefix + '.'):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = _FakeModule(fullname)
        sys.modules[fullname] = mod
        return mod


def _load_ckpt_safe(path, map_location='cpu'):
    """安全加载 checkpoint，兼容包含第三方库引用的权重文件

    优先使用 torch.load，若因缺少依赖模块失败，
    则通过 sys.meta_path 注入虚拟模块 finder 后重试 torch.load。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except (ModuleNotFoundError, AttributeError) as e:
        err_str = str(e)
        # 检测是否为缺少第三方库导致的错误
        fake_prefixes = ('ultralytics',)
        need_fake = [p for p in fake_prefixes if p in err_str]
        if not need_fake:
            raise
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"⚠️  torch.load 失败({e.__class__.__name__}: {e})，"
                  f"注入虚拟模块后重试...")
        # 安装 meta path finder 拦截所有 ultralytics.* 的 import
        finders = [_FakeModuleFinder(p) for p in need_fake]
        for f in finders:
            sys.meta_path.insert(0, f)
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        finally:
            # 移除 finder
            for f in finders:
                if f in sys.meta_path:
                    sys.meta_path.remove(f)
            # 清理注入的虚拟模块
            for p in need_fake:
                keys_to_remove = [k for k in sys.modules if k == p or k.startswith(p + '.')]
                for k in keys_to_remove:
                    sys.modules.pop(k, None)












def load_state_dict_with_prefix(model, load_ckpt, prefixes_to_try=['model.', 'module.', 'encoder.', 'backbone.', 'teacher.', 'student.'], state_dict=None):
    """自动处理权重键名前缀不匹配问题（双向适配）
    Args:
        model: 要加载权重的模型
        load_ckpt: 权重文件路径
        prefixes_to_try: 要尝试的前缀列表，默认包含常见的训练保存前缀
    Returns:
        加载了权重的模型
    """
    use_ddp = dist.is_initialized()
    if not state_dict:
        if not use_ddp or use_ddp and dist.get_rank() == 0:
            print(f"➡️  loading ckpt: {load_ckpt}")

        state_dict = _load_ckpt_safe(load_ckpt, map_location='cpu')
    
    # 处理 ultralytics 等第三方库的 checkpoint：加载结果可能是 _DummyClass 对象
    if isinstance(state_dict, _DummyClass):
        state_dict = state_dict.state_dict()
    
    # 首先提取模型权重（处理checkpoint中可能包含的其他信息）
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'model' in state_dict:
        m = state_dict['model']
        # ultralytics checkpoint 的 'model' 可能是 _DummyClass 对象
        state_dict = m.state_dict() if isinstance(m, _DummyClass) else m
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model_state_dict = model.state_dict()

    if not use_ddp or use_ddp and dist.get_rank() == 0:
        print(f"模型键数量: {len(model_state_dict)}, 权重键数量: {len(state_dict)}")
    
    # 尝试不同的前缀匹配策略
    best_match_ratio = -1  # 使用 -1 确保至少选一个策略
    best_state_dict = state_dict  # 默认使用原始权重
    best_strategy = "原始匹配"

    # 策略1: 原始匹配（不处理前缀）
    matching_keys = set(model_state_dict.keys()) & set(state_dict.keys())
    match_ratio = len(matching_keys) / len(model_state_dict) if model_state_dict else 0
    if match_ratio > best_match_ratio:
        best_match_ratio = match_ratio
        best_state_dict = state_dict
        best_strategy = "原始匹配"

    # 策略2: 去除权重中的前缀（权重比模型多前缀）
    for prefix in prefixes_to_try:
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        matching_keys = set(model_state_dict.keys()) & set(new_state_dict.keys())
        match_ratio = len(matching_keys) / len(model_state_dict) if model_state_dict else 0

        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            best_state_dict = new_state_dict
            best_strategy = f"去除权重前缀 '{prefix}'"

    # 策略3: 为权重添加前缀（模型比权重多前缀）
    for prefix in prefixes_to_try:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = prefix + key
            new_state_dict[new_key] = value

        matching_keys = set(model_state_dict.keys()) & set(new_state_dict.keys())
        match_ratio = len(matching_keys) / len(model_state_dict) if model_state_dict else 0

        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            best_state_dict = new_state_dict
            best_strategy = f"添加权重前缀 '{prefix}'"

    # 策略4: ultralytics 数字索引键映射
    # ultralytics 用扁平 Sequential: model.0.conv.weight, model.1.conv.weight, ...
    # 我们的模型用命名模块: stem.0.conv.weight, dark3.0.conv.weight, ...
    # 需要提供 key_map: {数字索引前缀: 命名前缀}
    if best_match_ratio < 0.5 and hasattr(model, '_ultralytics_key_map'):
        key_map = model._ultralytics_key_map()
        if key_map:
            new_state_dict = {}
            for key, value in state_dict.items():
                # 去除可能的 'model.' 前缀
                clean_key = key
                for prefix in prefixes_to_try:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        break
                # 尝试数字索引映射
                mapped = False
                for num_idx, named_idx in key_map.items():
                    if clean_key.startswith(num_idx + '.'):
                        new_key = named_idx + clean_key[len(num_idx):]
                        new_state_dict[new_key] = value
                        mapped = True
                        break
                if not mapped:
                    new_state_dict[clean_key] = value

            matching_keys = set(model_state_dict.keys()) & set(new_state_dict.keys())
            match_ratio = len(matching_keys) / len(model_state_dict) if model_state_dict else 0

            if match_ratio > best_match_ratio:
                best_match_ratio = match_ratio
                best_state_dict = new_state_dict
                best_strategy = f"ultralytics 数字索引映射 ({len(key_map)} 组)"
    
    if not use_ddp or use_ddp and dist.get_rank() == 0:
        print(f"最佳匹配策略: {best_strategy}, 匹配度: {best_match_ratio:.1%}")
    
    # 过滤掉形状不匹配的权重，避免load_state_dict报RuntimeError
    mismatch_keys = []
    filtered_state_dict = {}
    for key, value in best_state_dict.items():
        if key in model_state_dict:
            if value.shape != model_state_dict[key].shape:
                mismatch_keys.append((key, value.shape, model_state_dict[key].shape))
                continue
        filtered_state_dict[key] = value

    # 使用过滤后的权重加载
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    # 详细输出匹配情况
    if not use_ddp or use_ddp and dist.get_rank() == 0:
        if mismatch_keys:
            print(f"⚠️  形状不匹配跳过的键 ({len(mismatch_keys)}个):")
            for key, ckpt_shape, model_shape in mismatch_keys[:5]:
                print(f"   - {key}: checkpoint {list(ckpt_shape)} -> model {list(model_shape)}")
            if len(mismatch_keys) > 5:
                print(f"   ... 还有 {len(mismatch_keys) - 5} 个")

        if missing_keys:
            print(f"⚠️  缺失的键 ({len(missing_keys)}个):")
            for key in missing_keys[:5]:
                print(f"   - {key}")
            if len(missing_keys) > 5:
                print(f"   ... 还有 {len(missing_keys) - 5} 个")
        
        if unexpected_keys:
            print(f"⚠️  多余的键 ({len(unexpected_keys)}个):")
            for key in unexpected_keys[:5]:
                print(f"   - {key}")
            if len(unexpected_keys) > 5:
                print(f"   ... 还有 {len(unexpected_keys) - 5} 个")
        
        actual_loaded = len(model_state_dict) - len(missing_keys)
        actual_ratio = actual_loaded / len(model_state_dict) if model_state_dict else 0
        print(f"✅ 权重加载完成 - 实际加载度: {actual_ratio:.1%} ({actual_loaded}/{len(model_state_dict)})")
    
    return model










def _save_rng_state():
    """保存全局 RNG 状态 (python/numpy/torch/cuda), 用于断点续训时恢复随机性"""
    rng_state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state['cuda'] = torch.cuda.get_rng_state_all()
    return rng_state


def _load_rng_state(rng_state):
    """恢复全局 RNG 状态, 保证断点续训后的随机性与不中断完全一致"""
    if 'python' in rng_state:
        random.setstate(rng_state['python'])
    if 'numpy' in rng_state:
        np.random.set_state(rng_state['numpy'])
    if 'torch' in rng_state:
        torch.random.set_rng_state(rng_state['torch'])
    if 'cuda' in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state['cuda'])


def save_ckpt(epoch, eval_interval, model, scheduler, log_dir, args_history, flag_metric_name=None):
    """保存模型权重和训练断点

    Args:
        epoch (int): 当前epoch数
        eval_interval (int): 评估间隔，每隔多少个epoch评估一次
        model (nn.Module): 网络模型实例，支持DDP封装
        scheduler: 学习率调度器实例（包含优化器）
        log_dir (str): 日志文件保存目录
        args_history: 训练参数历史记录实例
        flag_metric_name (str, optional): 用于判断最优模型的指标名称，默认None
    """
    # ckpt一定不包含ddp那层封装的module
    ckpt = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    # checkpoint_dict能够恢复断点训练
    # 注意：checkpoint_dict 中保存的是未解包的 state_dict，用于 resume 时能正确加载
    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  # 未解包的 state_dict，用于 resume
        'optim_state_dict': scheduler.optimizer.state_dict(),
        'sched_state_dict': scheduler.base_scheduler.state_dict(),
        # ★★★ 断点续训关键: 保存 WarmupScheduler 完整状态 (不仅是 base_scheduler)
        'warmup_sched_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        # ★★★ 断点续训关键: 保存 RNG 状态, 保证 resume 后随机性与不中断一致
        'rng_state': _save_rng_state(),
    }
    torch.save(checkpoint_dict, os.path.join(log_dir, f"train_epoch{epoch}.pt"))
    # 保存 ckpt（已解包的）用于推理或微调
    torch.save(ckpt, os.path.join(log_dir, "last.pt"))
    # 如果本次Epoch的参考指标最大，则保存网络参数
    if flag_metric_name:
        flag_metric_list = args_history.args_history_dict[flag_metric_name]
        best_flag_metric_val = max(flag_metric_list)
        # ★★★ 修复: 用实际 epoch 号判断, 不依赖列表索引 (与 log_utils 同步修复)
        n = len(flag_metric_list)
        eval_start = epoch - (n - 1) * eval_interval
        eval_epochs = [eval_start + i * eval_interval for i in range(n)]
        # 找到所有最大值对应的实际 epoch (可能有多个相同最大值)
        best_epochs = [eval_epochs[i] for i, v in enumerate(flag_metric_list) if v == best_flag_metric_val]
        if epoch in best_epochs:
            torch.save(ckpt, os.path.join(log_dir, f'best_{flag_metric_name}.pt'))




def train_resume(resume, model, optimizer, scheduler, runner_logger, batch_nums):
    """恢复断点训练

    ★★★ 断点续训保证:
    1. 模型权重完全恢复
    2. 优化器状态 (动量/方差等) 完全恢复
    3. 学习率调度器状态完全恢复 (WarmupScheduler + base_scheduler)
    4. RNG 状态 (python/numpy/torch/cuda) 完全恢复 → 后续 epoch 的随机性与不中断一致
    5. DataLoader shuffle 通过 fit_epoch 中的 seed_everything(seed+epoch) 保证确定性

    Args:
        resume (str): 断点checkpoint文件路径
        model (nn.Module): 网络模型实例
        optimizer: 优化器实例
        scheduler: 学习率调度器实例 (WarmupScheduler)
        runner_logger: 运行日志记录器实例
        batch_nums (int): 每个epoch包含的batch数量

    Returns:
        int: 恢复后的起始epoch数
    """  
    ckpt = _load_ckpt_safe(resume, map_location="cpu")
    # resume后开始的epoch
    resume_epoch = ckpt['epoch'] + 1

    # ★★★ 1. 恢复模型权重
    model = load_state_dict_with_prefix(model, load_ckpt=None, state_dict=ckpt['model_state_dict'])

    # ★★★ 2. 恢复优化器状态 (动量/方差/step计数)
    optimizer.load_state_dict(ckpt['optim_state_dict'])

    # ★★★ 3. 恢复调度器状态 (优先用完整的 WarmupScheduler 状态)
    if 'warmup_sched_state_dict' in ckpt and ckpt['warmup_sched_state_dict'] is not None:
        # 新格式: 保存了完整 WarmupScheduler 状态 (包含 base_scheduler + warmup 参数)
        scheduler.load_state_dict(ckpt['warmup_sched_state_dict'])
    else:
        # 兼容旧格式: 只保存了 base_scheduler 状态
        scheduler.base_scheduler.load_state_dict(ckpt['sched_state_dict'])
        # 旧格式无法恢复 WarmupScheduler.last_epoch, 手动计算
        scheduler.last_epoch = ckpt['epoch']  # 0-indexed: epoch_N → last_epoch = N-1+1 = N

    # ★★★ 4. 恢复 RNG 状态 → 保证后续 epoch 的 dropout/augmentation/随机操作一致
    if 'rng_state' in ckpt:
        _load_rng_state(ckpt['rng_state'])
    else:
        # 兼容旧 checkpoint: 没有 RNG 状态, 警告用户
        use_ddp = dist.is_initialized()
        if not use_ddp or use_ddp and dist.get_rank() == 0:
            print("⚠️  checkpoint 不含 RNG 状态, resume 后随机操作 (dropout/aug) 可能与不中断训练不一致")

    # 主节点才进行日志记录
    use_ddp = dist.is_initialized()
    if not use_ddp or use_ddp and dist.get_rank() == 0:
        runner_logger.logger.info(f'resume:{resume}')
        runner_logger.logger.info(f'resume_epoch:{resume_epoch}')
        # 导入上一次中断训练时的args
        json_dir, _ = os.path.split(resume)
        runner_logger.argsHistory.loadRecord(json_dir)

    return resume_epoch

