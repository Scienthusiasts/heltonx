import torch
from torch import nn
import copy
import torch.nn.functional as F
# 注册机制
from heltonx.utils.register import MODELS
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix



@MODELS.register
class DPOLLM(nn.Module):
    def __init__(self, llm:nn.Module, loss:nn.Module, beta=0.1):
        """DPO微调LLM(用于实现针对LLM的DPO逻辑, 本质只是一个套壳, 核心的模型还是self.llm)
        """
        super(DPOLLM, self).__init__()
        self.beta = beta
        # 模型
        self.llm = llm
        self.ref_llm = copy.deepcopy(llm)  # 深拷贝
        self.ref_llm.eval()
        self.ref_llm.requires_grad_(False)
        # 损失
        self.loss = loss


    
    def forward(self, batch_datas, return_loss=True):
        """sft pipeline (和pretrain pipeline完全一致)
        """
        # X, Y, loss_mask = [bs, seq_lens]
        X_chosen, Y_chosen, mask_chosen, X_reject, Y_reject, mask_reject = batch_datas[0], batch_datas[1], batch_datas[2], batch_datas[3], batch_datas[4], batch_datas[5]
        # 只是batch维度拼接, 因此不会有问题
        X = torch.cat([X_chosen, X_reject], dim=0)
        Y = torch.cat([Y_chosen, Y_reject], dim=0)
        loss_mask = torch.cat([mask_chosen, mask_reject], dim=0)

        with torch.no_grad():
            ref_outputs = self.ref_llm(X)
            ref_logits = ref_outputs.logits

        policy_outputs = self.llm(X)
        policy_logits = policy_outputs.logits

        dpo_loss = self.dpo_loss(ref_logits, policy_logits, loss_mask, Y)

        '''损失以字典形式组织'''
        losses = dict(
            dpo_gen_loss=dpo_loss,
        )
        return losses


    def logits_to_log_probs(self, logits, labels):
        # logits shape: (batch_size, seq_len, vocab_size)
        # labels shape: (batch_size, seq_len)
        # log_probs shape: (batch_size, seq_len)
        log_probs = F.log_softmax(logits, dim=2)
        log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
        return log_probs_per_token


    def dpo_loss(self, ref_logits, policy_logits, mask, Y):
        ref_log_probs = self.logits_to_log_probs(ref_logits, Y)
        policy_log_probs = self.logits_to_log_probs(policy_logits, Y)
        # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
        # https://github.com/jingyaogong/minimind/issues/298
        seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8).squeeze()  # 防止零长度mask导致除零NaN
        ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths 
        policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths

        # 将 chosen 和 rejected 数据分开
        batch_size = ref_log_probs.shape[0]
        chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
        reject_ref_log_probs = ref_log_probs[batch_size // 2:]
        chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
        reject_policy_log_probs = policy_log_probs[batch_size // 2:]

        pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
        ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(self.beta * logits)
        return loss.mean()

    
    def state_dict(self, *args, **kwargs):
        """保存权重时只保存 self.llm 的参数
        """
        return self.llm.state_dict(*args, **kwargs)


    # def load_state_dict(self, state_dict, strict=True):
    #     """加载权重时只加载 self.llm 的参数
    #     """
    #     return self.llm.load_state_dict(state_dict, strict=strict)