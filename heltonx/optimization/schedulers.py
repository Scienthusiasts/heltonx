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
            min_lr (float, optional): warmup起始学习率，默认0.0
            last_epoch (int, optional): 上次训练结束时的epoch数，用于断点恢复，默认0
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.batch_num = batch_num
        
        # base_scheduler 的初始 lr 作为 warmup 的目标 lr
        self.target_lrs = [group['lr'] for group in optimizer.param_groups]

        # warmup 起始 lr 设置为 min_lr
        for group in optimizer.param_groups:
            group['lr'] = self.min_lr
        self.last_epoch = last_epoch

    def get_warmup_lr(self, batch):
        """获取当前batch的学习率（warmup阶段）

        Args:
            batch (int): 当前batch在epoch中的索引

        Returns:
            list: 各参数组的学习率列表
        """
        if self.last_epoch < self.warmup_epochs:
            # 0-based warmup，Epoch 1 lr = min_lr
            warmup_factor = (self.last_epoch * self.batch_num + batch) / (self.warmup_epochs * self.batch_num)
            return [self.min_lr + (target_lr - self.min_lr) * warmup_factor
                    for target_lr in self.target_lrs]
        else:
            return self.base_scheduler.get_last_lr()

    def step(self, batch, epoch):
        """执行一步学习率调度

        Args:
            batch (int): 当前batch在epoch中的索引
            epoch (int): 当前epoch数
        """
        if epoch - 1 > self.last_epoch:
            self.last_epoch = epoch-1

        if self.last_epoch <= self.warmup_epochs:
            # warmup 阶段手动更新 optimizer.lr(更新周期以batch为单位)
            lr_list = self.get_warmup_lr(batch)
            for idx, group in enumerate(self.optimizer.param_groups):
                group['lr'] = lr_list[idx]
        else:
            # 其余阶段更新 optimizer.lr(更新周期以epoch为单位)
            self.base_scheduler.step(epoch - self.warmup_epochs)


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
        self.target_lrs = state_dict.get('target_lrs', self.target_lrs)
        self.last_epoch = state_dict.get('last_epoch', self.last_epoch)












if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim

    base_schedulers_cfgs = {
        # "type": "StepLR",
        # # 每间隔step_size个epoch更新学习率
        # "step_size": 1,
        # # 每次学习率变为原来的gamma倍
        # "gamma": 0.5,
        "type": "CosineAnnealingLR",
        "T_max": 400 - 2,
        "eta_min": 2e-5,
    }
    warmup_schedulers_cfgs = {
        "type": "WarmupScheduler",
        "min_lr": 2e-6,
        "warmup_epochs": 2
    }
    # 假设一个简单的模型
    model = nn.Linear(256, 2)

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=2e-4)
    base_scheduler = SCHEDULERS.build_from_cfg(base_schedulers_cfgs, optimizer=optimizer)
    scheduler = SCHEDULERS.build_from_cfg(warmup_schedulers_cfgs, base_scheduler=base_scheduler, optimizer=optimizer, batch_num=100)

    # 数据 (只是示例, 随便造点数据)
    x = torch.randn(1000, 256)
    y = torch.randint(0, 2, (1000,))

    criterion = nn.CrossEntropyLoss()


    for epoch in range(1, 400+1):  # 共训练 20 个 epoch
        for iter in range(1563):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # 打印当前学习率
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}, Batch {iter}, Loss={loss.item():.4f}, LR={lr:.6f}")

            scheduler.step(epoch=epoch, batch=iter)
