import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from heltonx.utils.utils import seed_everything



def init_model(device, weight_dir):
    """加载模型与分词器(加载huggingface, transformer库的格式的模型)
    """
    # AutoTokenizer 会根据模型名称自动选择正确的分词规则（例如BPE、SentencePiece）
    tokenizer = AutoTokenizer.from_pretrained(weight_dir)
    # # 加载语言模型, 使用transformer库自动进行配置(符合现有开源大模型的标准, 可以直接导入huggingfave上的llm模型)
    # trust_remote_code=True 表示允许加载远程仓库中自定义的模型代码（有时非官方模型需要）
    model = AutoModelForCausalLM.from_pretrained(weight_dir, trust_remote_code=True)
    print(f'模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(device), tokenizer



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load_from = 'F:\Desktop\git\minimind\Qwen-4B-Instruct'
    load_from = 'F:\Desktop\git\minimind\Qwen-0.6B'
    # load_from = './ckpts/hugging_face/MiniMind2-R1'
    historys = 0
    # seed_everything(42) 

    # 初始化对话存储列表，用于存储上下文历史
    conversation = []
    model, tokenizer = init_model(device, load_from)
    # 创建流式输出器（边生成边打印）
    # skip_prompt=True 表示不重复打印用户输入
    # skip_special_tokens=True 表示跳过 <bos>、<eos>、<pad> 等特殊符号
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = iter(lambda: input('👶: '), '')
    for prompt in prompt_iter:
        # 保留最近的历史对话（如果设置了historys）
        conversation = conversation[-historys:] if historys else []
        # 在每次构造 templates 前都保证 system 存在一次
        system_prompt = {
            "role": "system", 
            "content": 
                    # """
                    #     你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

                    #     # 可用工具:
                    #     - `get_weather(city: str)`: 查询指定城市的实时天气。
                    #     - `get_attraction(city: str)`: 根据城市搜索推荐的旅游景点。
                    #     - `get_time(city: str, weather: str)`: 根据城市查询当前时间(包含年月日小时分钟秒)。
                    #     - `get_path(start: str, end: str)`: 查询起始地到目的地的导航路线
                        
                    #     # 行动格式:
                    #     你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动。
                    #     Thought: [这里是你的思考过程和下一步计划]
                    #     Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

                    #     # 任务完成:
                    #     当你收集到足够的信息，能够回答用户的最终问题时，你必须使用 `finish(answer="...")` 来输出最终答案。

                    #     请开始吧！
                    # """
                    """
                        你是一个可靠的智能助手, 请尽量以直白且生动形象的语言回答用户提出的问题, 在必要的时候, 你可以举个例子
                    """
        }
        if not conversation or conversation[0].get("role") != "system":
            conversation.insert(0, system_prompt)
        # 将用户当前输入加入到对话上下文
        conversation.append({"role": "user", "content": prompt})
        # 构造对话模板（Hugging Face 的chat模板，用于自动拼接system+user+assistant格式）
        templates = {
            "conversation": conversation,
            "tokenize": False, 
            "add_generation_prompt": True,
            # 是否开启CoT
            "enable_thinking": False
        }
        # 根据模板的参数将模板转换成可供模型输入的字符串
        inputs = tokenizer.apply_chat_template(**templates)
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
            do_sample=True,                           # 开启随机采样（非贪心搜索）
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