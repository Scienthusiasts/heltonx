# 需要import才能注册
from generation import * 

from heltonx.utils.utils import get_args, dynamic_import_class
from heltonx.tools.eval import *




if __name__ == '__main__':
    args = get_args()
    config_path = args.config
    # 使用动态导入模块导入参数文件
    cargs = dynamic_import_class(config_path, get_class=False)
    # 初始化runner
    runner = Evaler(cargs.seed, cargs.log_dir, cargs.model_cfgs, cargs.dataset_cfgs)
    # 注册 Hook
    # 任务特定的评估pipeline
    eval_pipeline = EVALPIPELINES.build_from_cfg(cargs.eval_pipeline_cfgs)
    hook = NecessaryHook(eval_pipeline)
    runner.register_hook("after_eval", hook.hook_after_eval)
    runner.eval()




