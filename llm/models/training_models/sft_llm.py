import torch
from torch import nn
# 注册机制
from heltonx.utils.register import MODELS
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix



@MODELS.register
class SFTLLM(nn.Module):
    def __init__(self, load_ckpt, llm:nn.Module, loss:nn.Module):
        """有监督微调LLM(用于实现针对LLM的SFT逻辑, 本质只是一个套壳, 核心的模型还是self.llm)
           Instruct-Tuning, CoT-Tuning本质都是SFT, 均适用该类
        """
        super(SFTLLM, self).__init__()
        # 模型
        self.llm = llm
        # 损失
        self.loss = loss
        # 是否导入预训练权重
        if load_ckpt: 
            self = load_state_dict_with_prefix(self, load_ckpt)

    
    def forward(self, batch_datas, return_loss=True):
        """sft pipeline (和pretrain pipeline完全一致)
        """
        # X, Y, loss_mask = [bs, seq_lens]
        X, Y, loss_mask = batch_datas[0], batch_datas[1], batch_datas[2]
        # out = CausalLMOutputWithPast(loss=..., logits=..., past_key_values=..., hidden_states=..., attentions=...)
        out = self.llm(X)
        # [bs, seq_lens, vocab_size] -> [bs, vocab_size, seq_lens]
        logits = out.logits.transpose(1, 2) 
        # [bs, seq_lens]
        loss = self.loss(logits, Y)            
        # 不计算PAD部分的loss
        gen_loss = (loss * loss_mask).sum() / loss_mask.sum()
        # aux_loss可能是MoE的负载均衡损失
        aux_loss = torch.tensor(0.) if out.aux_loss == 0 else out.aux_loss

        '''损失以字典形式组织'''
        losses = dict(
            gen_loss=gen_loss,
            aux_loss=aux_loss
        )
        # TODO: 梯度裁剪, 梯度累加
        return losses


    def state_dict(self, *args, **kwargs):
        """保存权重时只保存 self.llm 的参数
        """
        return self.llm.state_dict(*args, **kwargs)


    def load_state_dict(self, state_dict, strict=True):
        """加载权重时只加载 self.llm 的参数
        """
        return self.llm.load_state_dict(state_dict, strict=strict)