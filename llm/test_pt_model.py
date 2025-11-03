import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from .models.base_models.minimind import MiniMindConfig, MiniMindForCausalLM
from heltonx.utils.utils import seed_everything
from heltonx.utils.ckpts_utils import load_state_dict_with_prefix



def init_model(device, tokenizer_dir, weight_path, hidden_size, num_hidden_layers, use_moe, inference_rope_scaling):
    """加载模型与分词器(加载huggingface, transformer库的格式的模型)
    """
    # 加载训练好的分词器, 会根据模型名称自动选择正确的分词规则（例如BPE、SentencePiece）
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    # 加载语言模型
    configs = dict(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
        inference_rope_scaling=inference_rope_scaling,
        vocab_size=6400
    )
    model = MiniMindForCausalLM(configs)
    # model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    model = load_state_dict_with_prefix(model, weight_path)
    print(f'模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(device), tokenizer



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # weight_path = 'log/llm/minimind_sft512_cot/2025-11-03-15-30-10_train_ddp/last.pt'
    weight_path = 'log/llm/minimind_sft512/2025-11-02-13-46-32_train_ddp/last.pt'
    tokenizer_dir = 'llm/tokenizer_configs/minimind2'
    instruct_model = True
    historys = 2
    # seed_everything(42) 

    # 初始化对话存储列表，用于存储上下文历史
    conversation = []
    model, tokenizer = init_model(device, tokenizer_dir, weight_path, 768, 16, False, False)
    # 创建流式输出器（边生成边打印）
    # skip_prompt=True 表示不重复打印用户输入
    # skip_special_tokens=True 表示跳过 <bos>、<eos>、<pad> 等特殊符号
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = iter(lambda: input('👶: '), '')
    for prompt in prompt_iter:
        # 保留最近的历史对话（如果设置了historys）
        conversation = conversation[-historys:] if historys else []
        # 将用户当前输入加入到对话上下文
        conversation.append({"role": "user", "content": prompt})
        # 构造对话模板（Hugging Face 的chat模板，用于自动拼接system+user+assistant格式）
        templates = {
            "conversation": conversation,
            "tokenize": False, 
            "add_generation_prompt": True,
            # 是否开启CoT(本质是是否加<think></think>字段, =True不加, =False加)
            "enable_thinking": True
        }
        # 将模板转换成可供模型输入的字符串
        inputs = tokenizer.apply_chat_template(**templates) if instruct_model else tokenizer.bos_token + prompt
        # 使用tokenizer将文本转换为模型输入（token ids、attention mask）
        # truncation=True 保证输入不过长
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(device)
        # 提示输出
        print('🤖️: ', end='')
        # 使用模型的generate()接口生成文本（即自回归生成）
        generated_ids = model.generate(
            inputs=inputs["input_ids"],               # 输入的token ids
            attention_mask=inputs["attention_mask"],  # 注意力掩码（padding部分为0）
            max_new_tokens=8192,                      # 最大生成长度
            do_sample=True,                           # 开启随机采样（不再选择概率最大的 token（贪心），而是对概率分布进行采样）
            streamer=streamer,                        # 实时打印输出
            pad_token_id=tokenizer.pad_token_id,      # 填充token id 
            eos_token_id=tokenizer.eos_token_id,      # 结束token id
            top_p=0.85,                               # nucleus采样概率阈值
            temperature=0.85,                         # 温度采样系数（越高越随机）
            repetition_penalty=1.0                    # 重复惩罚系数（>1抑制重复）
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