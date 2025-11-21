# 需要import才能注册
from llm import *

from heltonx.utils.utils import get_args, dynamic_import_class
from heltonx.tools.train_accelerate import *




if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = get_args()
    config_path = args.config
    # 使用动态导入模块导入参数文件
    cargs = dynamic_import_class(config_path, get_class=False)
    # 初始化runner
    runner = Trainer(cargs.mode, cargs.epoch, cargs.seed, cargs.log_dir, cargs.log_interval, cargs.eval_interval, cargs.resume, 
                     cargs.model_cfgs, cargs.dataset_cfgs, cargs.optimizer_cfgs, cargs.scheduler_cfgs, cargs.grad_accumulate, cargs.grad_clip)
    # 注册 Hook
    # 任务特定的评估pipeline
    eval_pipeline = None
    hook = NecessaryHook(eval_pipeline)
    runner.register_hook("after_batch", hook.hook_after_batch)
    runner.register_hook("after_epoch", hook.hook_after_epoch)

    # 拷贝一份当前训练对应的config文件(方便之后查看细节)
    if runner.accelerator.is_main_process:
        shutil.copy(config_path, os.path.join(runner.log_dir, os.path.basename(config_path)))
    runner.fit()

