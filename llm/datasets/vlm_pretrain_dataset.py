import json
import random
import re
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from PIL import Image
import os
from heltonx.utils.utils import seed_everything, worker_init_fn
from heltonx.utils.register import DATASETS
from functools import partial
from llm.datasets.preprocess import Transforms
os.environ["TOKENIZERS_PARALLELISM"] = "false"



@DATASETS.register
class VLMPretrainDataset(Dataset):
    def __init__(self, json_data_path, imgs_dir, tokenizer_cfg_dir, img_size, max_length=1024, image_special_token='@' * 196):
        """SFT数据集, 从 jsonl 文件中读取每一行的问答数据
           Instruct-Tuning, CoT-Tuning本质都是SFT, 均适用该Dataset
            Args:
                json_data_path:          数据集json文件
                tokenizer_cfg_dir: 模型权重(hf格式)
                max_length:              数据的最大序列长度, 超过会截断, 不足会填充 PAD
                has_CoT:                 =False是Instruct-Tuning, =True是CoT-Tuning
                special_tokens_weight:   CoT时会包含<think></think>和<answer></answer>, 增加这些tokens的权重保证, 计算损失时对格式的约束更强
        """
        super().__init__()
        self.imgs_dir = imgs_dir
        self.transform = Transforms(img_size=img_size)

        self.image_token = image_special_token
        # 加载训练好的 HuggingFace 格式的 tokenizer，用于把文本转成 token ids
        # tokenizer 内部包含词表 / 特殊 token / 编码规则等元数据
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg_dir)
        self.bos_id = self.tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = self.tokenizer('<|im_end|>', add_special_tokens=False).input_ids

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
        """使用预训练分词器将文本转成 token ids
           构造自回归训练输入 (image + X) 和标签 (Y) 
        """
        sample = self.samples[index]
        image_paths = sample['image']
        # 为了避免长度过长, 只提取回答中的第一段话
        sample['conversations'][1]['content'] = sample['conversations'][1]['content'].split('\n')[0]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        # 进行tokenize + 截断
        input_ids = self.tokenizer(prompt).input_ids
        input_ids = input_ids[:self.max_length]
        # padding补齐
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 根据 <bos>assistant ... <eos> 生成 loss mask(只有assistant回答的部分用于计算损失)
        loss_mask = self._generate_loss_mask(input_ids)

        # X: 去掉最后一个 token,  Y: X的下一个tokens
        # 模型学: 输入X的基础上预测X的下一个词(Y)
        X = torch.tensor(input_ids[:-1], dtype=torch.long).unsqueeze(0)
        Y = torch.tensor(input_ids[1:], dtype=torch.long).unsqueeze(0)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long).unsqueeze(0)

        # 可以处理多图
        image_tensors = []
        for image_name in image_paths.split(','):
            image_name = image_name.strip()
            image = Image.open(f'{self.imgs_dir}/{image_name}').convert('RGB')  
            image = np.array(image)
            # 图像预处理
            image = self.albumAug(image).transpose(2,0,1)    
            image_tensors.append(image)
        images = np.stack(image_tensors, axis=0)
        return X, Y, loss_mask, images
    

    def albumAug(self, img):
        """基于albumentations库的基础数据预处理
        """
        trans = self.transform.transform(image=img)          
        img = trans['image']   
        return img
    

    def _create_chat_prompt(self, cs):
        """
            根据对话结构应用 Chat Template
            - 支持 system message + function/tool use
            - 返回拼接完格式化 prompt 文本（不 token 化）
        """
        messages = []
        for i, turn in enumerate(cs):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        # 把对话结构 → 模型预训练 format
        # 注意, 这里最后和推理时严格一致, 否则可能因为分布不一致导致推理效果不佳
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            # =True时会额外添加 <|im_start|>assistant (训练时=False, 推理时=True)
            add_generation_prompt=False
        )




    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask




    def dataset_collate(self, batch_datas):
        """
        """
        X, Y, loss_mask, images = [], [], [], []
        for data in batch_datas:
            x, y, mask, imgs = data[0], data[1], data[2], data[3]
            X.append(x)
            Y.append(y)
            loss_mask.append(mask)
            images.append(imgs)

        X = torch.cat(X)
        Y = torch.cat(Y)
        loss_mask = torch.cat(loss_mask)
        # np -> tensor
        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)

        return X, Y, loss_mask, images









if __name__ == "__main__":
    # json_data_path = '/data/yht/data/llm/sft_512.jsonl'
    json_data_path = '/mnt/yht/data/vlm/pretrain_data_qwen3vlflash_.jsonl'
    imgs_dir = '/mnt/yht/data/vlm/pretrain_images'
    tokenizer_cfg_dir = '/mnt/yht/code/HeltonPretrain/llm/tokenizer_configs/minimind2'
    img_size = [224, 224]
    dataset = VLMPretrainDataset(json_data_path, imgs_dir, tokenizer_cfg_dir, img_size)
    train_data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=dataset.dataset_collate, worker_init_fn=partial(worker_init_fn, seed=42))
    # 输出数据格式
    for epoch in range(1):
        for step, batch in enumerate(train_data_loader):
            X, Y, loss_mask, images = batch[0], batch[1], batch[2], batch[3]
            # images.shape = [bs, num_img, 3, h, w]
            # print(images.shape)
            
