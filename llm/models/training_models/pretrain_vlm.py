import torch
from torch import nn
# 注册机制
from heltonx.utils.register import MODELS



@MODELS.register
class PretrainVLM(nn.Module):
    def __init__(self, vlm:nn.Module, loss:nn.Module):
        """预训练VLM(用于实现针对VLM的预训练逻辑, 本质只是一个套壳, 核心的模型还是self.llm)
           Pretrain本质是在训练模型的文本补全能力, 训练出来的模型不具备对话能力, 还得经过Instruct-Tuning
        """
        super(PretrainVLM, self).__init__()
        # 模型
        self.vlm = vlm
        # 损失
        self.loss = loss

    
    def forward(self, batch_datas, return_loss=True):
        """预训练pipeline
        """
        # X, Y, loss_mask = [bs, seq_lens], pixel_values = [bs, num_img, 3, h, w]
        X, Y, loss_mask, image = batch_datas[0], batch_datas[1], batch_datas[2], batch_datas[3]
        out = self.vlm(X, pixel_values=image)
        # [bs, seq_lens, vocab_size] -> [bs, vocab_size, seq_lens]
        logits = out.logits.transpose(1, 2) 
        # [bs, seq_lens]
        loss = self.loss(logits, Y)            
        # 不计算PAD部分的loss
        gen_loss = (loss * loss_mask).sum() / loss_mask.sum()
        # aux_loss可能是MoE的负载均衡损失
        aux_loss = getattr(out, "aux_loss", torch.tensor(0.0, device=out.logits.device))
        aux_loss = torch.tensor(0.0, device=out.logits.device) if aux_loss == 0.0 else aux_loss

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
        return self.vlm.state_dict(*args, **kwargs)


    def load_state_dict(self, state_dict, strict=True):
        """加载权重时只加载 self.llm 的参数
        """
        return self.vlm.load_state_dict(state_dict, strict=strict)