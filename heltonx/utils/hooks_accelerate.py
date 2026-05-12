# coding=utf-8
"""
HeltonX Hook 机制模块（Accelerate 实现）
提供基于 Accelerate 的分布式训练回调钩子
"""
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from heltonx.utils.ckpts_utils import save_ckpt




class NecessaryHook:
    """必要的训练/评估回调钩子（Accelerate 实现）

    提供训练和评估过程中常用的回调功能，自动处理分布式训练中的进程同步
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
            runner: 训练器实例，包含当前训练状态信息（Accelerate封装）
        """
        if runner.accelerator.is_main_process:
            # 记录/打印日志
            runner.runner_logger.train_iter_log_printer(runner.cur_step, runner.cur_epoch, runner.optimizer, runner.losses)


    def hook_after_epoch(self, runner):
        """epoch级别日志回调 + 保存权重

        每个epoch结束后执行，包含评估（可选）和模型保存

        Args:
            runner: 训练器实例，包含当前训练状态信息（Accelerate封装）
        """
        if (runner.cur_epoch % runner.eval_interval == 0 or runner.cur_epoch == runner.epoch) and runner.accelerator.is_main_process:
            # 评估+记录/打印日志
            if self.eval_pipeline:
                flag_metric_name = self.hook_after_eval(runner)
            else:
                flag_metric_name = None
            # 保存权重
            if runner.accelerator.is_main_process:
                model_unwrapped = runner.accelerator.unwrap_model(runner.model)
                save_ckpt(runner.cur_epoch, runner.eval_interval, model_unwrapped, runner.scheduler,
                        runner.log_dir, runner.runner_logger.argsHistory, flag_metric_name)

    def hook_after_eval(self, runner):
        """评估回调

        执行模型评估并记录评估指标，自动解包Accelerate封装的模型

        Args:
            runner: 训练器实例，包含当前训练状态信息（Accelerate封装）

        Returns:
            tuple: (evaluations dict, flag_metric_name str)
        """
        # 需要解包构建一个非DDP包装的模型副本，否则如果只用gpu0上的模型推理时, 
        # 一些操作会使用跨进程通信(allreduce / broadcast), 此时会产生阻塞
        model_unwrapped = runner.accelerator.unwrap_model(runner.model)
        # 评估
        evaluations, flag_metric_name = self.eval_pipeline(runner, model_unwrapped)
        # 记录/打印日志
        runner.runner_logger.train_epoch_log_printer(runner.cur_epoch, evaluations, flag_metric_name)
        return flag_metric_name