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
class PretrainDataset(Dataset):
    def __init__(self, json_data_path, tokenizer_cfg_dir, max_length=512):
        """预训练数据集, 从 jsonl 文件中读取每一行的 {"text": "..."} 数据;
            Args:
                json_data_path:          数据集json文件
                tokenizer_cfg_dir: 模型权重(hf格式)
                max_length:              数据的最大序列长度, 超过会截断, 不足会填充 PAD
        """
        super().__init__()
        # 加载训练好的 HuggingFace 格式的 tokenizer，用于把文本转成 token ids
        # tokenizer 内部包含词表 / 特殊 token / 编码规则等元数据
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg_dir)
        self.max_length = max_length
        # 读取所有 json 数据行（每行是 {"text": "..."}）
        self.samples = self.load_data(json_data_path)


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
        # 一行text格式: "<|im_start|> 文本1 <|im_end|>...<|im_start|> 文本n <|im_end|>"
        text = str(self.samples[index]["text"])
        # 构建输入文本, 并将文本进行分词
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = enc.input_ids.squeeze()
        # loss_mask 标记非 PAD token，PAD 不计算 loss
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        # X: 去掉最后一个 token,  Y: X的下一个tokens
        # 模型学: 输入X的基础上预测X的下一个词(Y)
        X = input_ids[:-1].long().unsqueeze(0)
        Y = input_ids[1:].long().unsqueeze(0)
        loss_mask = loss_mask[1:].long().unsqueeze(0)

        return X, Y, loss_mask
    

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
    json_data_path = '/data/yht/data/llm/pretrain_hq.jsonl'
    tokenizer_cfg_dir = 'ckpts/hugging_face/Qwen-0.6B'

    dataset = PretrainDataset(json_data_path, tokenizer_cfg_dir)
    train_data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=dataset.dataset_collate, worker_init_fn=partial(worker_init_fn, seed=42))
    # 输出数据格式
    for epoch in range(1):
        for step, batch in enumerate(train_data_loader):
            X, Y, loss_mask = batch[0], batch[1], batch[2]
            print(X.shape)
            
