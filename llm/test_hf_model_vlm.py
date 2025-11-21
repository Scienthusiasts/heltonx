import argparse
import random
import warnings
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from llm.models.base_models.minimindv import MiniMindForCausalVLM
from llm.datasets.preprocess import Transforms
from heltonx.utils.utils import seed_everything
from heltonx.utils.register import MODELS




def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_from = 'ckpts/hugging_face/MiniMind2-V'
    img_path = '/mnt/yht/data/vlm/pretrain_images/GCC_train_000000072.jpg'
    historys = 0
    llm_cfg = dict(
        type='AutoModelForCausalLM',
        weight_dir = load_from,
    )
    tokenizer_cfg = dict(
        type='AutoTokenizer',
        weight_dir = load_from        
    )
    # 初始化图像处理transform
    img_size=[224, 224]
    transform = Transforms(img_size=img_size)

    # 初始化对话存储列表，用于存储上下文历史
    conversation = []
    tokenizer = MODELS.build_from_cfg(tokenizer_cfg)
    model = MODELS.build_from_cfg(llm_cfg)
    model.vision_encoder, model.processor = model.get_vision_model("/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/clip-vit-base-patch16")

    print(f'模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    preprocess = model.processor
    model.eval().to(device)
    # 获取图像
    image = Image.open(img_path).convert('RGB')
    # 图像预处理
    process_img = transform.transform(image=np.array(image))['image'].transpose(2,0,1)      
    pixel_values = torch.tensor(process_img).to(device).unsqueeze(0).unsqueeze(0)     
    print(pixel_values.shape)

    # 创建流式输出器（边生成边打印）
    # skip_prompt=True 表示不重复打印用户输入, skip_special_tokens=True 表示跳过 <bos>、<eos>、<pad> 等特殊符号
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # <image> 请你描述一下图像中所展示的内容.
    prompt_iter = iter(lambda: input('👶: '), '')
    for prompt in prompt_iter:
        # 保留最近的历史对话（如果设置了historys）
        conversation = conversation[-historys:] if historys else []
        # 将用户当前输入加入到对话上下文
        conversation.append({"role": "user", "content": prompt.replace('<image>', model.params.image_special_token)})
        templates = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        print(templates)
        inputs = tokenizer(templates, return_tensors="pt", truncation=True).to(device)

        # 提示输出
        print('🤖️: ', end='')
        # 使用模型的generate()接口生成文本（即自回归生成）
        generated_ids = model.generate(
            inputs=inputs["input_ids"],               # 输入的token ids
            attention_mask=inputs["attention_mask"],  # 注意力掩码（padding部分为0）
            max_new_tokens=512,                      # 最大生成长度
            do_sample=True,                           # 开启随机采样（非贪心搜索）
            streamer=streamer,                        # 实时打印输出
            pad_token_id=tokenizer.pad_token_id,      # 填充token id 
            eos_token_id=tokenizer.eos_token_id,      # 结束token id
            top_p=0.85,                               # nucleus采样概率阈值
            temperature=0.65,                         # 温度采样系数（越高越随机）
            repetition_penalty=1.0,
            pixel_values=pixel_values
        )
        # 将生成的结果解码成文本（跳过特殊符号）
        # 只保留新生成的部分（去掉输入长度）
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        # 将AI的回答加入到对话历史
        conversation.append({"role": "assistant", "content": response})
        # 打印空行分隔下一轮对话
        print('\n\n')





if __name__ == "__main__":
    main()