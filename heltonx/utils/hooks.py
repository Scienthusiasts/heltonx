# coding=utf-8
"""
HeltonX Hook 机制模块（PyTorch DDP 实现）
提供训练过程中的回调钩子，支持日志记录、模型保存、评估等功能
"""
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from heltonx.utils.ckpts_utils import save_ckpt




class NecessaryHook:
    """必要的训练/评估回调钩子

    提供训练和评估过程中常用的回调功能，包括日志记录、模型保存和评估执行
    """

    def __init__(self, eval_pipeline=None):
        """初始化 NecessaryHook

        Args:
            eval_pipeline: 任务特定的评估管道实例，若为None则跳过评估
        """
        self.eval_pipeline = eval_pipeline

    def hook_after_batch(self, runner):
        """batch级别日志回调

        Args:
            runner: 训练器实例，包含当前训练状态信息
        """
        # 只有主进程或非DDP模式才打印日志
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if runner.mode == 'train' or (runner.mode == 'train_ddp' and is_main_process):
            # 记录/打印日志
            runner.runner_logger.train_iter_log_printer(runner.cur_step, runner.cur_epoch, runner.optimizer, runner.losses)


    def hook_after_epoch(self, runner):
        """epoch级别日志回调 + 保存权重

        每个epoch结束后执行，包含评估（可选）和模型保存

        Args:
            runner: 训练器实例，包含当前训练状态信息
        """
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if runner.mode == 'train' or (runner.mode == 'train_ddp' and is_main_process):
            if runner.cur_epoch % runner.eval_interval == 0 or runner.cur_epoch == runner.epoch:
                # 评估+记录/打印日志
                if self.eval_pipeline:
                    flag_metric_name = self.hook_after_eval(runner)
                else:
                    flag_metric_name = None
                # 保存权重
                save_ckpt(runner.cur_epoch, runner.eval_interval, runner.model, runner.scheduler,
                        runner.log_dir, runner.runner_logger.argsHistory, flag_metric_name)


    def hook_after_eval(self, runner):
        """评估回调

        执行模型评估并记录评估指标

        Args:
            runner: 训练器实例，包含当前训练状态信息

        Returns:
            tuple: (evaluations dict, flag_metric_name str)
        """
        # 评估
        evaluations, flag_metric_name = self.eval_pipeline(runner)
        # 记录/打印日志
        runner.runner_logger.train_epoch_log_printer(runner.cur_epoch, evaluations, flag_metric_name)
        return flag_metric_name