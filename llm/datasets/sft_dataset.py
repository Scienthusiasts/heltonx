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
class SFTDataset(Dataset):
    def __init__(self, json_data_path, huggingface_weights_dir, max_length=1024):
        """SFT数据集, 从 jsonl 文件中读取每一行的问答数据;
           使用预训练分词器将文本转成 token ids
           构造自回归训练输入 (X) 和标签 (Y)
            Args:
                json_data_path:          数据集json文件
                huggingface_weights_dir: 模型权重(hf格式)
                max_length:              数据的最大序列长度, 超过会截断, 不足会填充 PAD
        """
        super().__init__()
        # 加载训练好的 HuggingFace 格式的 tokenizer，用于把文本转成 token ids
        # tokenizer 内部包含词表 / 特殊 token / 编码规则等元数据
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_weights_dir)
        self.max_length = max_length
        # 将文件中每一行 jsonl 加载到内存
        self.samples = self.load_data(json_data_path)
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
        """
        """
        sample = self.samples[index]
        # 构建对话提示
        # prompt格式: "<|im_start|>user user的提问<|im_end|> <|im_start|>assistant <think>(空)</think> assistant的回答<|im_end|>"
        prompt = self._create_chat_prompt(sample['conversations'])
        # 进行tokenize + 截断
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # padding补齐
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 根据 <bos>assistant ... <eos> 生成 loss mask(只有assistant回答的部分用于计算损失)
        loss_mask = self._generate_loss_mask(input_ids)

        # X: 去掉最后一个 token,  Y: X的下一个tokens
        # 模型学: 输入X的基础上预测X的下一个词(Y)
        X = torch.tensor(input_ids[:-1], dtype=torch.long).unsqueeze(0)
        Y = torch.tensor(input_ids[1:], dtype=torch.long).unsqueeze(0)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long).unsqueeze(0)

        return X, Y, loss_mask
    

    def _create_chat_prompt(self, cs):
        """
            根据对话结构应用 Chat Template
            - 支持 system message + function/tool use
            - 返回拼接完格式化 prompt 文本（不 token 化）
        """
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        # 把对话结构 → 模型预训练 format
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False, # 不加 assistant 起始提示符
            tools=tools
        )


    def _generate_loss_mask(self, input_ids):
        """
            根据 token 序列标记 loss mask:
            - 模型只计算 assistant 回复段落的 loss
            - 用户输入部分 mask=0, assistant输出 mask=1

            原理:
                找到 "bos + assistant" 开始 → 到 "eos" 结束
                此范围内 token = 1, 否则 = 0
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 检测 assistant 段起始
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 内容开始位置
                start = i + len(self.bos_id)
                end = start
                # 找该段落的 eos 结束位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # mask [start+1, eos] 之间的 token
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 跳到 eos 后继续
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


    def dataset_collate(self, batch_datas):
        """
        """
        X, Y, loss_mask = [], [], []
        for data in batch_datas:
            x, y, mask = data[0], data[1], data[2]
            X.append(x)
            Y.append(y)
            loss_mask.append(mask)

        X = torch.cat(X)
        Y = torch.cat(Y)
        loss_mask = torch.cat(loss_mask)
        return X, Y, loss_mask









if __name__ == "__main__":
    json_data_path = '/data/yht/data/llm/sft_512.jsonl'
    huggingface_weights_dir = 'ckpts/hugging_face/Qwen-0.6B'

    dataset = SFTDataset(json_data_path, huggingface_weights_dir)
    train_data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=dataset.dataset_collate, worker_init_fn=partial(worker_init_fn, seed=42))
    # 输出数据格式
    for epoch in range(1):
        for step, batch in enumerate(train_data_loader):
            X, Y, loss_mask = batch[0], batch[1], batch[2]
            print(X.shape)
            
