"""
HeltonX 学习率调度器模块
支持 PyTorch 原生调度器及自定义 Warmup 调度器
"""
import inspect
import torch.optim.lr_scheduler as lr_scheduler

from heltonx.utils.register import SCHEDULERS

# 批量注册 torch.optim.lr_scheduler 里的常见scheduler
# 支持: ChainedScheduler, ConstantLR, CosineAnnealingLR, CosineAnnealingWarmRestarts,
# CyclicLR, ExponentialLR, LRScheduler, LambdaLR, LinearLR, MultiStepLR,
# MultiplicativeLR, OneCycleLR, PolynomialLR, ReduceLROnPlateau, SequentialLR, StepLR
for name, obj in inspect.getmembers(lr_scheduler, inspect.isclass):
    # 排除私有类（以 _ 开头的）
    if name.startswith("_"):
        continue
    # 过滤出属于 lr_scheduler 模块的类（排除继承链里导入的外部类）
    if obj.__module__ == lr_scheduler.__name__:
        SCHEDULERS.add_item(name, obj)


@SCHEDULERS.register
class WarmupScheduler:
    """带 warmup 的通用学习率调度器

    在训练初期逐步增加学习率，避免早期训练不稳定；
    warmup 结束后切换到基础调度器继续学习率调整。
    """

    def __init__(
        self,
        base_scheduler,
        optimizer,
        batch_num,
        warmup_epochs=5,
        min_lr=0.0,
        last_epoch=0
    ):
        """初始化 WarmupScheduler

        Args:
            base_scheduler: 已经实例化的基础学习率调度器，warmup结束后使用此调度器
            optimizer: 已经实例化的优化器
            batch_num (int): 一个epoch包含的batch数量，用于计算warmup步数
            warmup_epochs (int, optional): warmup阶段的epoch数，默认5
            min_lr (float, optional): warmup起始学习率（全局），默认0.0。
                当optimizer有多个param_groups时（如backbone_lr_mult），
                各组的起始lr按与最大lr的比例缩放，保持倍率关系一致。
            last_epoch (int, optional): 上次训练结束时的epoch数，用于断点恢复，默认0
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.batch_num = batch_num
        
        # base_scheduler 的初始 lr 作为 warmup 的目标 lr
        self.target_lrs = [group['lr'] for group in optimizer.param_groups]

        # 各 param_group 的 warmup 起始 lr，按与最大 target_lr 的比例缩放 min_lr
        # 确保 backbone_lr_mult 等倍率关系在 warmup 期间保持一致
        max_target_lr = max(self.target_lrs)
        if max_target_lr > 0:
            self.min_lrs = [min_lr * (tl / max_target_lr) for tl in self.target_lrs]
        else:
            self.min_lrs = [min_lr] * len(self.target_lrs)

        # warmup 起始 lr 设置为各组对应的 min_lr
        for idx, group in enumerate(optimizer.param_groups):
            group['lr'] = self.min_lrs[idx]
        self.last_epoch = last_epoch

    def get_warmup_lr(self, batch):
        """获取当前batch的学习率（warmup阶段）

        Args:
            batch (int): 当前batch在epoch中的索引

        Returns:
            list: 各参数组的学习率列表
        """
        # warmup阶段：从min_lr线性增长到target_lr
        # last_epoch是0-indexed: epoch1→0, epoch2→1, ...
        # 当last_epoch < warmup_epochs时处于warmup阶段
        # 使用1-based步数，使lr从第一个batch就开始增长
        current_step = self.last_epoch * self.batch_num + batch + 1
        total_steps = self.warmup_epochs * self.batch_num
        warmup_factor = min(current_step / total_steps, 1.0)
        return [min_lr + (target_lr - min_lr) * warmup_factor
                for min_lr, target_lr in zip(self.min_lrs, self.target_lrs)]

    def step(self, batch, epoch):
        """执行一步学习率调度

        Args:
            batch (int): 当前batch在epoch中的索引
            epoch (int): 当前epoch数
        """
        # 新epoch开始时更新last_epoch（0-indexed: epoch1→0, epoch2→1, ...）
        if epoch - 1 > self.last_epoch:
            self.last_epoch = epoch - 1

        if self.last_epoch < self.warmup_epochs:
            # warmup 阶段: 手动逐batch更新optimizer.lr
            lr_list = self.get_warmup_lr(batch)
            for idx, group in enumerate(self.optimizer.param_groups):
                group['lr'] = lr_list[idx]
        else:
            # warmup结束: 使用base_scheduler按epoch更新lr
            # 仅在每个epoch的第一个batch调用step，避免重复步进
            if batch == 0:
                self.base_scheduler.step()


    def get_last_lr(self):
        """获取最近一次调度的学习率

        Returns:
            list: 各参数组最近的学习率列表
        """
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """返回调度器的状态字典，用于保存检查点"""
        state = {
            'base_scheduler': self.base_scheduler.state_dict() if self.base_scheduler else None,
            'warmup_epochs': self.warmup_epochs,
            'min_lr': self.min_lr,
            'min_lrs': self.min_lrs,
            'target_lrs': self.target_lrs,
            'last_epoch': self.last_epoch,
        }
        return state

    def load_state_dict(self, state_dict):
        """从状态字典恢复调度器状态"""
        if state_dict.get('base_scheduler') and self.base_scheduler:
            self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self.warmup_epochs = state_dict.get('warmup_epochs', self.warmup_epochs)
        self.min_lr = state_dict.get('min_lr', self.min_lr)
        self.min_lrs = state_dict.get('min_lrs', self.min_lrs)
        self.target_lrs = state_dict.get('target_lrs', self.target_lrs)
        self.last_epoch = state_dict.get('last_epoch', self.last_epoch)












if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim

    epoch = 50
    lr = 2e-4
    warmup_decay = 1e-2
    lr_decay = 1e-1
    warmup_epochs = 1

    base_schedulers_cfgs=dict(
        type="CosineAnnealingLR",
        T_max=epoch - warmup_epochs,
        eta_min=lr * lr_decay,
    )
    warmup_schedulers_cfgs=dict(
        type="WarmupScheduler",
        min_lr=lr * warmup_decay,
        warmup_epochs=warmup_epochs
    )
    # 假设一个简单的模型
    model = nn.Linear(256, 2)

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr)
    base_scheduler = SCHEDULERS.build_from_cfg(base_schedulers_cfgs, optimizer=optimizer)
    scheduler = SCHEDULERS.build_from_cfg(warmup_schedulers_cfgs, base_scheduler=base_scheduler, optimizer=optimizer, batch_num=100)

    # 数据 (只是示例, 随便造点数据)
    x = torch.randn(1000, 256)
    y = torch.randint(0, 2, (1000,))

    criterion = nn.CrossEntropyLoss()


    for epoch in range(1, epoch+1):  # 共训练 20 个 epoch
        for iter in range(100):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # 打印当前学习率
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}, Batch {iter}, Loss={loss.item():.4f}, LR={lr:.6f}")

            scheduler.step(epoch=epoch, batch=iter)
