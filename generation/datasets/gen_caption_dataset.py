import numpy as np
import torch
import json
from PIL import Image, ImageFile
import torch.utils.data.dataset as data
from torch.utils.data import DataLoader
from torchvision.io import read_image
from functools import partial
import matplotlib.pyplot as plt
import os
import torch.distributed as dist
from transformers import AutoTokenizer
# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 自定义
from heltonx.utils.register import DATASETS
from heltonx.utils.utils import seed_everything, worker_init_fn
from generation.datasets.preprocess import Transforms
os.environ["TOKENIZERS_PARALLELISM"] = "false"







@DATASETS.register
class GenCaptionDataset(data.Dataset):      
    '''有监督分类任务对应的数据集读取方式
    '''
    def __init__(self, img_dir, json_data_path, img_size, max_length, tokenizer_cfg_dir, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225]):    
        '''__init__() 为默认构造函数，传入数据集类别（训练或测试），以及数据集路径

        Args:
            dir:               图像数据集的根目录
            mode:              模式(train/valid)
            img_size:          网络要求输入的图像尺寸
            json_data_path:    captions json文件路径(包含对应图像名)
            max_length:        数据的最大序列长度, 超过会截断, 不足会填充 PAD
            tokenizer_cfg_dir: 分词模型权重(hf格式)
            img_mean:          归一化的图像均值
            img_std:           归一化的图像标准差

        Returns:
            precision, recall
        '''      
        self.img_mean = img_mean
        self.img_std = img_std
        self.img_dir = img_dir
        self.transform = Transforms(img_size, self.img_mean, self.img_std)
        # 加载训练好的 HuggingFace 格式的 tokenizer，用于把文本转成 token ids
        # tokenizer 内部包含词表 / 特殊 token / 编码规则等元数据
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg_dir)
        self.max_length = max_length
        # 读取所有 json 数据行（每行是 {"text": "..."}）
        self.samples = self.load_data(json_data_path)
        # 记录数据集大小
        self.dataSize = len(self.samples)
        # 打印数据集信息
        use_ddp = dist.is_initialized()
        if not use_ddp or use_ddp and dist.get_rank() == 0:
            print(f'📄  dataset info: 图像数:{self.__len__()}')


    def load_data(self, path):
        """
        从 .jsonl 文件中逐行读取数据
        每一行应是 {"text": "..."} 格式
        返回一个样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    

    def __getitem__(self, item):  
        '''重载data.Dataset父类方法, 获取数据集中数据内容
        '''   
        sample = self.samples[item]
        '''处理图像'''
        image_path = os.path.join(self.img_dir, sample['image'])
        # 读取图片 (用torchvision.io.read_image读取速度会快一些)
        img = read_image(image_path).permute(1,2,0)
        img = np.array(img)
        # 数据增强
        img = self.albumAug(img).transpose(2,0,1) 
        '''处理文本'''
        captions = sample['conversations'][1]['content']
        # 进行tokenize + 截断
        input_ids = self.tokenizer(captions).input_ids
        input_ids = input_ids[:self.max_length]
        # padding补齐
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        caption_tokens = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        return img, caption_tokens
    

    def albumAug(self, img):
        """基于albumentations库的基础数据预处理
        """
        trans = self.transform.transform(image=img)          
        img = trans['image']   
        return img


    def __len__(self):
        '''重载data.Dataset父类方法, 返回数据集大小
        '''
        return self.dataSize


    # DataLoader中collate_fn参数使用
    # 由于检测数据集每张图像上的目标数量不一
    # 因此需要自定义的如何组织一个batch里输出的内容
    def dataset_collate(self, batch):
        images, caption_tokens = [], []
        for img, caption_token in batch:
            images.append(img)
            caption_tokens.append(caption_token)
        # np -> tensor
        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        caption_tokens = torch.cat(caption_tokens)
        return images, caption_tokens
    

    # for debug only:
    def _vis_GenDataset_batch(self, epoch, step, batch):
        '''可视化训练集一个batch
        Args:
        Retuens:
            None     
        '''
        # 图像均值 标准差
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([[0.229, 0.224, 0.225]]) 

        imgs = batch
        plt.figure(figsize = (8,8))
        for idx, img in enumerate(imgs):
            img = img.numpy().transpose((1,2,0))
            img = img * std + mean
            plt.subplot(8,8,idx+1)
            plt.imshow(img)
            plt.axis("off")
            # 微调行间距
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0.01, hspace=0.2)

        plt.savefig(f'./epoch{epoch}_step{step}_.jpg', dpi=300)







# for test only
if __name__ == '__main__':

    # 配置字典
    cfg = {
        "dataset_cfg": {
            "type": "GenCaptionDataset",
            "img_dir": r'/mnt/yht/data/celeba_256/train',
            "img_size": [256, 256],
            "json_data_path": r'/mnt/yht/data/celeba_256/celeba256_captions_qwen3vlflash_structure.jsonl', 
            "max_length":768, 
            "tokenizer_cfg_dir": r'/mnt/yht/code/HeltonPretrain/llm/tokenizer_configs/minimind2'
        },
        "bs": 64,
        "seed": 42,
        "shuffle": True
    }

    dataset_cfg = cfg["dataset_cfg"]
    seed_everything(cfg["seed"])
    train_dataset = DATASETS.build_from_cfg(dataset_cfg)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=cfg["bs"], shuffle=cfg["shuffle"], num_workers=8, collate_fn=train_dataset.dataset_collate, worker_init_fn=partial(worker_init_fn, seed=cfg["seed"]))
    # 输出数据格式
    for epoch in range(1, 10):
        for step, batch in enumerate(train_data_loader):
            batch_imgs, batch_captions = batch[0], batch[1]
            if step == 0:
                # 可视化一个batch里的图像
                train_dataset._vis_GenDataset_batch(epoch, step, batch_imgs)
