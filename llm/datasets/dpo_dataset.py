import json
import random
import re
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import os
from heltonx.utils.utils import seed_everything, worker_init_fn
from heltonx.utils.register import DATASETS
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"



@DATASETS.register
class DPODataset(Dataset):
    def __init__(self, json_data_path, tokenizer_cfg_dir, max_length=1024, has_CoT=False, special_tokens_weight=10):
        """DPO数据集(直接偏好优化, 从两个回答中优化模型的回答接近更优回答, 远离次优回答), 从 jsonl 文件中读取每一行的问答数据
            Args:
                json_data_path:          数据集json文件
                tokenizer_cfg_dir:       模型权重(hf格式)
                max_length:              数据的最大序列长度, 超过会截断, 不足会填充 PAD
                has_CoT:                 =False是Instruct-Tuning, =True是CoT-Tuning
                special_tokens_weight:   CoT时会包含<think></think>和<answer></answer>, 增加这些tokens的权重保证, 计算损失时对格式的约束更强
        """
        super().__init__()
        self.has_CoT = has_CoT
        self.special_tokens_weight = special_tokens_weight
        # 加载训练好的 HuggingFace 格式的 tokenizer，用于把文本转成 token ids
        # tokenizer 内部包含词表 / 特殊 token / 编码规则等元数据
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg_dir)
        # 思考标签占位符(训练CoT-Tuning时用到)
        self.sot = self.tokenizer('<think>').input_ids
        self.eot = self.tokenizer('</think>').input_ids
        self.soa = self.tokenizer('<answer>').input_ids
        self.eoa = self.tokenizer('</answer>').input_ids
        # 特殊字符对应的id:
        self.special_token_ids = torch.tensor(self.sot + self.eot + self.soa + self.eoa) 
        self.max_length = max_length
        # 将文件中每一行 jsonl 加载到内存
        self.samples = self.load_data(json_data_path)
        self.padding = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        # 开始和结束tokens; bos_token + "assistant" 用于检测 assistant 段落起始, eos_token 用于检测 assistant 段落结束
        self.bos_id = self.tokenizer(f'{self.tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = self.tokenizer(f'{self.tokenizer.eos_token}', add_special_tokens=False).input_ids


    def load_data(self, path):
        """
            从 .jsonl 文件中逐行读取数据
            每一行应是 {"text": "..."} 格式
            返回一个样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data = json.loads(line.strip())
                samples.append(data)
        return samples


    def __len__(self):
        """返回数据集样本数量
        """
        return len(self.samples)


    def __getitem__(self, index):
        """使用预训练分词器将文本转成 token ids
           构造自回归训练输入 (X) 和标签 (Y)
        """
        sample = self.samples[index]
        chosen = sample['chosen']  
        reject = sample['rejected'] 

        # 构建对话提示
        # prompt格式: "<|im_start|>user user的提问<|im_end|> <|im_start|>assistant <think>...</think> assistant的回答<|im_end|>"
        chosen_prompt = self._create_chat_prompt(chosen)
        reject_prompt = self._create_chat_prompt(reject)
        # 进行tokenize + 截断/PAD
        chosen_input_ids = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length').input_ids
        reject_input_ids = self.tokenizer(reject_prompt, truncation=True, max_length=self.max_length, padding='max_length').input_ids
        # 根据 <bos>assistant ... <eos> 生成 loss mask(只有assistant回答的部分用于计算损失)
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
        reject_loss_mask = self._generate_loss_mask(reject_input_ids)

        # X: 去掉最后一个 token,  Y: X的下一个tokens
        # 模型学: 输入X的基础上预测X的下一个词(Y)
        X_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long).unsqueeze(0)
        Y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long).unsqueeze(0)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long).unsqueeze(0)
        X_reject = torch.tensor(reject_input_ids[:-1], dtype=torch.long).unsqueeze(0)
        Y_reject = torch.tensor(reject_input_ids[1:], dtype=torch.long).unsqueeze(0)
        mask_reject = torch.tensor(reject_loss_mask[1:], dtype=torch.long).unsqueeze(0)

        return X_chosen, Y_chosen, mask_chosen, X_reject, Y_reject, mask_reject
    

    def _create_chat_prompt(self, cs):
        """
            根据对话结构应用 Chat Template
            - 支持 system message + function/tool use
            - 返回拼接完格式化 prompt 文本（不 token 化）
        """
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        # 把对话结构 → 模型预训练 format
        # 注意, 这里最后和推理时严格一致, 反正可能因为分布不一致导致推理效果不佳
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            # =True时会额外添加 <|im_start|>assistant (训练时=False, 推理时=True)
            add_generation_prompt=False,
            tools=tools
        )


    def _generate_loss_mask(self, input_ids):
        """
        自动适配两种模板：
        1. <bos>assistant ... <eos> （旧模板）
        2. <|im_start|>assistant ... <|im_end|> （Qwen模板）
        """
        loss_mask = [0] * len(input_ids)

        # 两种模板的标记
        im_start = self.tokenizer("<|im_start|>", add_special_tokens=False).input_ids
        im_end = self.tokenizer("<|im_end|>", add_special_tokens=False).input_ids
        assistant = self.tokenizer("assistant", add_special_tokens=False).input_ids
        bos_assistant = self.tokenizer(f'{self.tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        eos = self.tokenizer(f'{self.tokenizer.eos_token}', add_special_tokens=False).input_ids

        i = 0
        while i < len(input_ids):
            # ① 匹配 Qwen 模板
            if input_ids[i:i + len(im_start + assistant)] == im_start + assistant:
                start = i + len(im_start + assistant)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(im_end)] == im_end:
                        break
                    end += 1
                for j in range(start, min(end, len(loss_mask))):
                    loss_mask[j] = 1
                i = end + len(im_end)

            # ② 匹配minimind模板 (<bos>assistant ... <eos>)
            elif input_ids[i:i + len(bos_assistant)] == bos_assistant:
                start = i + len(bos_assistant)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(eos)] == eos:
                        break
                    end += 1
                for j in range(start, min(end, len(loss_mask))):
                    loss_mask[j] = 1
                i = end + len(eos)

            else:
                i += 1

        # 防止全0，避免除零导致 NaN
        if sum(loss_mask) == 0:
            loss_mask[0] = 1

        return loss_mask



    def dataset_collate(self, batch_datas):
        """
        """
        X_chosen, Y_chosen, mask_chosen, X_reject, Y_reject, mask_reject = [], [], [], [], [], []
        for data in batch_datas:
            x_c, y_c, mask_c, x_r, y_r, mask_r = data[0], data[1], data[2], data[3], data[4], data[5]
            X_chosen.append(x_c)
            Y_chosen.append(y_c)
            mask_chosen.append(mask_c)
            X_reject.append(x_r)
            Y_reject.append(y_r)
            mask_reject.append(mask_r)

        X_chosen = torch.cat(X_chosen)
        Y_chosen= torch.cat(Y_chosen)
        mask_chosen = torch.cat(mask_chosen)
        X_reject = torch.cat(X_reject)
        Y_reject= torch.cat(Y_reject)
        mask_reject = torch.cat(mask_reject)

        if self.has_CoT:
            # 直接在整个Y张量中查找特殊token
            # NOTE:特殊token可能是许多一般字符组合成的, 这样会不会造成一些一般字符也被增加权重?
            chosen_sp_ids = torch.isin(Y_chosen, self.special_token_ids)
            reject_sp_ids = torch.isin(Y_reject, self.special_token_ids)
            # 对思考标签增加10倍权重
            mask_chosen[chosen_sp_ids] = self.special_tokens_weight
            mask_reject[reject_sp_ids] = self.special_tokens_weight

        return X_chosen, Y_chosen, mask_chosen, X_reject, Y_reject, mask_reject









if __name__ == "__main__":
    json_data_path = '/data/yht/data/llm/dpo.jsonl'
    tokenizer_cfg_dir = '/data/yht/code/HeltonPretrain/llm/tokenizer_configs/minimind2'

    dataset = DPODataset(json_data_path, tokenizer_cfg_dir, has_CoT=True, special_tokens_weight=10)
    train_data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=dataset.dataset_collate, worker_init_fn=partial(worker_init_fn, seed=42))
    # 输出数据格式
    for epoch in range(1):
        for step, batch in enumerate(train_data_loader):
            X_chosen, Y_chosen, mask_chosen, X_reject, Y_reject, mask_reject = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
            print(X_chosen.shape, X_reject.shape)
            
