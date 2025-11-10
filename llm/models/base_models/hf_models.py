from .blocks import *
# 注册机制
from heltonx.utils.register import MODELS
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


# AutoModelForCausalLM.from_pretrained(...) 返回的是实例对象, 因此注册一个注册构造函数
@MODELS.register("AutoModelForCausalLM")
def AutoModelForCausalLM_builder(weight_dir):
    """
        weight_dir: huggingface 模型权重所在dir
    """
    return AutoModelForCausalLM.from_pretrained(weight_dir, trust_remote_code=True)


@MODELS.register("AutoTokenizer")
def AutoTokenizer_builder(weight_dir):
    """
        weight_dir: huggingface 模型权重所在dir
    """
    return AutoTokenizer.from_pretrained(weight_dir)