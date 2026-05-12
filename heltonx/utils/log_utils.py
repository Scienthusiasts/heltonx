# coding=utf-8
"""
HeltonX 日志工具模块
提供训练日志记录、TensorBoard 支持和模型信息打印功能
"""
import os
import json
import torch
import logging
import time
import datetime
import argparse
from torch import nn
from logging import Logger
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Optional

# 多卡并行训练
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP





class ArgsHistory:
    """训练参数历史记录器

    记录训练或验证过程中的各类变量（如 loss、lr 等）以及 TensorBoard 日志
    以字典形式存储，例: {'loss1': [...], 'loss2': [...], ...}
    """

    def __init__(self, json_save_dir, mode):
        """初始化 ArgsHistory

        Args:
            json_save_dir (str): JSON 记录文件的保存目录
            mode (str): 运行模式，可选值：'train', 'eval', 'train_ddp'
        """
        # NOTE: 多卡训练时只在主进程创建 TensorBoard writer
        if mode in ['train', 'eval'] or (mode == 'train_ddp' and dist.get_rank() == 0):
            self.tb_writer = SummaryWriter(log_dir=json_save_dir)
        self.json_save_dir = json_save_dir
        self.args_history_dict = {}  # 所有变量记录在 self.args_history_dict 中


    def record(self, key, value):
        """更新变量的历史记录

        Args:
            key (str): 要记录的变量名称
            value: 要记录的变量数值（会自动转换为 float 类型）

        Returns:
            None
        """
        # 可能存在json格式不支持的类型, 因此统一转成float类型
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        # 如果日志中还没有这个变量，则新建
        if key not in self.args_history_dict.keys():
            self.args_history_dict[key] = []
        # 更新history dict
        self.args_history_dict[key].append(value)
        # 顺便更新tensorboard（仅在tb_writer存在时）
        if hasattr(self, 'tb_writer') and self.tb_writer is not None:
            self.tb_writer.add_scalar(key, value, len(self.args_history_dict[key]))


    def saveRecord(self):
        """以 JSON 格式保存历史记录

        将 args_history_dict 中的所有变量记录保存到 JSON 文件
        """
        if not os.path.isdir(self.json_save_dir):os.makedirs(self.json_save_dir) 
        json_save_path = os.path.join(self.json_save_dir, 'args_history.json')
        # 保存
        with open(json_save_path, 'w') as json_file:
            json.dump(self.args_history_dict, json_file)


    def loadRecord(self, json_load_dir):
        """加载上一次训练的历史记录

        Args:
            json_load_dir (str): JSON 文件所在目录（用于断点恢复）
        """
        json_path = os.path.join(json_load_dir, 'args_history.json')
        with open(json_path, "r", encoding="utf-8") as json_file:
            self.args_history_dict = json.load(json_file)








class ModelInfoLogger:
    """模型信息记录器

    用于生成类似 MMDetection 的模型参数表格，包含参数名称、形状、值范围等信息
    """
    
    @staticmethod
    def get_parameter_info(model: nn.Module, optimizer:optim.Optimizer) -> List[Dict]:
        """获取模型参数信息
        
        Args:
            model: 模型实例
            optimizer: 优化器实例（用于获取学习率和权重衰减）
            
        Returns:
            参数信息列表
        """
        param_info = []
        
        # 创建参数到优化器参数组的映射
        param_to_group = {}
        for group_idx, param_group in enumerate(optimizer.param_groups):
            for param in param_group['params']:
                param_to_group[param] = group_idx
        
        for name, param in model.named_parameters():
            # 获取参数形状
            shape = "X".join(str(dim) for dim in param.shape)
            
            # 获取参数值范围
            if param.numel() > 0:
                min_val = param.data.min().item()
                max_val = param.data.max().item()
                value_scale = f"Min:{min_val:.3f} Max:{max_val:.3f}"
            else:
                value_scale = "Min:0.000 Max:0.000"
            
            # 判断是否被优化（通常是可训练参数）
            optimized = "Y" if param.requires_grad else "N"
            
            # 默认值
            lr = 0.001
            wd = 0.0
            
            # 从优化器中获取实际的学习率和权重衰减
            if param in param_to_group:
                group_idx = param_to_group[param]
                param_group = optimizer.param_groups[group_idx]
                lr = param_group.get('lr', lr)
                wd = param_group.get('weight_decay', wd)
            
            param_info.append({
                'name': name,
                'optimized': optimized,
                'shape': shape,
                'value_scale': value_scale,
                'lr': lr,
                'wd': wd
            })
        
        return param_info
    

    @staticmethod
    def create_model_table(param_info: List[Dict], max_name_width: int = 80, 
                        max_shape_width: int = 15) -> str:
        """创建模型参数表格
        Args:
            param_info: 参数信息列表
            max_name_width: 名称列最大宽度
            max_shape_width: 形状列最大宽度
            
        Returns:
            格式化后的表格字符串
        """
        if not param_info:
            return "No parameters found."
        
        # 表格列定义
        columns = [
            {'name': 'Parameter Name', 'key': 'name', 'width': max_name_width},
            {'name': 'Optimized', 'key': 'optimized', 'width': 10},
            {'name': 'Shape', 'key': 'shape', 'width': max_shape_width},
            {'name': 'Value Scale [Min, Max]', 'key': 'value_scale', 'width': 25},
            {'name': 'Init Lr', 'key': 'lr', 'width': 8},
            {'name': 'Wd', 'key': 'wd', 'width': 8}
        ]
        
        # 计算表格总宽度
        table_width = sum(col['width'] for col in columns) + len(columns) * 3 + 1
        
        # 创建表格标题
        title = "Model Information"
        title_line = f"+{'-' * (table_width - 2)}+"
        title_row = f"|{title:^{table_width - 2}}|"
        # 创建表头分隔线
        header_separator = "+".join(["-" * (col['width'] + 2) for col in columns])
        header_separator = f"+{header_separator}+"
        # 创建表头
        header = "|"
        for col in columns:
            header += f" {col['name']:^{col['width']}} |"
        # 构建表格
        table_lines = []
        table_lines.append(title_line)
        table_lines.append(title_row)
        table_lines.append(header_separator)
        table_lines.append(header)
        table_lines.append(header_separator)
        
        # 添加数据行（居中对齐）
        for info in param_info:
            row = "|"
            for col in columns:
                value = info[col['key']]
                if col['key'] in ['lr', 'wd']:
                    formatted_value = f"{value:.5f}" if isinstance(value, float) else str(value)
                else:
                    formatted_value = str(value)
                
                if len(formatted_value) > col['width']:
                    formatted_value = formatted_value[:col['width']-3] + "..."
                
                row += f" {formatted_value:^{col['width']}} |"
            table_lines.append(row)
        
        table_lines.append(header_separator)
        
        return "\n".join(table_lines)

    
    @staticmethod
    def get_model_summary(model: nn.Module) -> Dict:
        """获取模型摘要信息

        Args:
            model (nn.Module): 模型实例

        Returns:
            dict: 包含 total_parameters, trainable_parameters,
                  non_trainable_parameters, parameter_memory_MB 的字典
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'parameter_memory_MB': total_params * 4 / (1024 ** 2)  # 假设float32，4字节/参数
        }














class RunnerLogger:
    """训练运行日志记录器

    提供日志打印、文件记录、TensorBoard 写入和模型信息打印功能
    """

    def __init__(
        self,
        mode: str,
        log_dir: str,
        log_interval: int,
        eval_interval: int,
        batch_num: int,
        total_epoch: int
    ):
        """初始化 RunnerLogger

        Args:
            mode (str): 运行模式，可选值：'train', 'eval', 'train_ddp'
            log_dir (str): 日志文件保存目录
            log_interval (int): 日志打印间隔，每隔多少个 iteration 打印一次
            eval_interval (int): 评估间隔，每隔多少个 epoch 评估一次
            batch_num (int): 一个 epoch 包含的 batch 数量
            total_epoch (int): 总训练 epoch 数
        """
        self.mode = mode
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.logger = logging.getLogger('runer')
        self.batch_num = batch_num
        self.logger.setLevel(level=logging.DEBUG)
        self.log_dir = log_dir
        self.total_iters = total_epoch * self.batch_num
        self.last_time = time.time()
        self.interval_time = 0
        # 日志格式
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        # 写入文件的日志
        self.log_dir = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{mode}")
        # 日志文件保存路径
        self.log_save_path = os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{mode}.log")
        if not os.path.isdir(self.log_dir):os.makedirs(self.log_dir)
        # NOTE:多卡
        # 只在 local_rank 为 0 的进程中设置日志记录
        if mode in ['train', 'eval'] or (mode == 'train_ddp' and dist.get_rank() == 0):
            # 记录到log文件中的日志
            file_handler = logging.FileHandler(self.log_save_path, encoding="utf-8", mode="a")
            file_handler.setLevel(level=logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            # 终端输出的日志
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        # 对于非主进程，可以设置一个空的日志处理器来忽略日志记录
        else:
            self.logger.addHandler(logging.NullHandler())

        self.argsHistory = ArgsHistory(self.log_dir, self.mode)
        self.model_info_logger = ModelInfoLogger()



    def log_model_info(self, model: nn.Module, optimizer: optim.Optimizer):
        """打印模型详细信息表格

        输出模型参数量、形状、值范围、学习率和权重衰减等信息

        Args:
            model (nn.Module): 模型实例
            optimizer: 优化器实例（用于获取真实的学习率和权重衰减）
        """
        # 保存原 formatter
        handlers = self.logger.handlers
        original_formatters = [h.formatter for h in handlers]
        # 临时替换为无 asctime 的 formatter
        simple_formatter = logging.Formatter('%(message)s')
        for h in handlers:
            h.setFormatter(simple_formatter)
        try:
            # 获取参数信息
            param_info = self.model_info_logger.get_parameter_info(model, optimizer)
            model_table = self.model_info_logger.create_model_table(param_info)
            summary = self.model_info_logger.get_model_summary(model)

            for line in model_table.split('\n'):
                self.logger.info(line)
            self.logger.info(f"🧩 Model Info:")
            self.logger.info(f"  ➤ Total Parameters: {summary['total_parameters']:,}")
            self.logger.info(f"  ➤ Trainable Parameters: {summary['trainable_parameters']:,}")
            self.logger.info(f"  ➤ Non-trainable Parameters: {summary['non_trainable_parameters']:,}")
            self.logger.info(f"  ➤ Estimated Memory Usage: {summary['parameter_memory_MB']:.2f} MB \n")
        finally:
            # 恢复原 formatter
            for h, f in zip(handlers, original_formatters):
                h.setFormatter(f)



    def train_iter_log_printer(self, step: int, epoch: int, optimizer, losses: dict):
        """训练迭代过程中记录/打印日志

        Args:
            step (int): 当前 iteration 在当前 epoch 中的索引
            epoch (int): 当前 epoch 数
            optimizer: 优化器实例（用于打印当前学习率）
            losses (dict): 当前 batch 的损失字典，包含各损失项

        打印效果示例:
            2025-09-22 22:54:39: Epoch(train) [2][50/59]  lr: 0.000990  total_loss: 1.14722
        """   
        # 记录每次iter耗时
        self.interval_time = time.time() - self.last_time
        self.last_time = time.time()
        eta_hours = (self.total_iters - step - (epoch-1) * self.batch_num) * self.interval_time / 3600
        eta_mins = (eta_hours - int(eta_hours)) * 60
        '''记录'''
        current_lr = optimizer.param_groups[0]['lr']
        self.argsHistory.record('lr', current_lr)
        # 记录所有损失:
        for loss_name, loss_value in losses.items():
            # 处理可能的非tensor类型
            if torch.is_tensor(loss_value):
                self.argsHistory.record(loss_name, loss_value.item())
            else:
                try:
                    self.argsHistory.record(loss_name, float(loss_value))
                except (TypeError, ValueError):
                    pass  # 跳过无法转换的值
        '''打印'''
        # 每间隔log_interval个iter才打印一次
        if step % self.log_interval != 0:
            return
        # 右对齐, 打印更美观
        batch_idx = '{:>{}}'.format(step, len(f"{self.batch_num}"))
        log = ("Epoch(train) [%d][%s/%d] eta: %02d:%02d  lr: %8f  ") % (epoch, batch_idx, self.batch_num, int(eta_hours), int(eta_mins), current_lr)
        for loss_name, loss_value in losses.items():
            # 处理可能的非tensor类型
            if torch.is_tensor(loss_value):
                loss_value = loss_value.item()
            try:
                loss_log = (loss_name + ": %.5f  " % float(loss_value))
            except (TypeError, ValueError):
                loss_log = (loss_name + ": N/A  ")
            log += loss_log
        self.logger.info(log)



    def train_epoch_log_printer(self, epoch, evaluations, flag_metric_name):
        """epoch 结束时记录/打印评估日志

        打印评估指标、当前最优 epoch 和最优指标值

        Args:
            epoch (int): 当前 epoch 数
            evaluations (dict): 评估指标字典
            flag_metric_name (str): 用于判断最优模型的指标名称

        打印效果示例:
            ==================================================================
            Epoch(valid) [1]  val_acc: 0.50286  best_val_epoch: 1  best_val_acc: 0.50286
            ==================================================================
        """        
        '''记录评估指标'''
        for metric_name, metric_value in evaluations.items():
            self.argsHistory.record(metric_name, metric_value)
        # 以json格式保存args
        self.argsHistory.saveRecord()

        '''打印'''
        # 找到最高准确率对应的epoch
        flag_metric_list = self.argsHistory.args_history_dict[flag_metric_name]
        best_epoch = flag_metric_list.index(max(flag_metric_list)) + 1
        best_flag_metric_val = max(flag_metric_list)
        # 打印指标
        self.logger.info('=' * 150)
        log = ("Epoch(valid) [%d]  ") % (epoch)
        for metric_name, metric_value in evaluations.items():
            loss_log = (metric_name+": %.5f  " % (metric_value))
            log += loss_log
        log += (f"best_val_epoch: {best_epoch * self.eval_interval}  best_{flag_metric_name}: {best_flag_metric_val:.5f}")
        self.logger.info(log)
        self.logger.info('=' * 150)

