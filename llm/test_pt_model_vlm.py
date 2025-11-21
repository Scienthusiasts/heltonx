import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, TextStreamer
from llm.datasets.preprocess import Transforms
from heltonx.utils.register import MODELS





def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # weight_path = 'ckpts/minimind2/minimindv_sft_vlm_768.pth'
    weight_path = 'log/llm/minimind_pretrain/2025-11-19-19-35-20_train_ddp/last.pt'
    tokenizer_dir = 'llm/tokenizer_configs/minimind2'
    # img_path = '/mnt/yht/data/vlm/pretrain_images/GCC_train_002672738.jpg'
    img_path = 'llm/demo/P0252__1024__860___350.png'
    instruct_model = True
    historys = 0
    llm_cfgs=dict(
        type="MiniMindForCausalVLM",
        vision_encoder = dict(
            type="OpenAICLIPImgEncoder",
            weight_dir="/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/clip-vit-base-patch16"
            # type='DINOv3',
            # weight_dir = '/mnt/yht/code/HeltonPretrain/ckpts/hugging_face/DINOv3s'
        ),
        load_ckpt=weight_path, 
        config=dict(
            v_hidden_size=768,    # 视觉tokens初始维度
            hidden_size=768,      # 模型tokens维度
            num_hidden_layers=16, # transformer 堆叠层数
            vocab_size=6400,      # 使用的词表的大小(单词数)
            use_moe=False, 
            inference_rope_scaling=False,
        )
    )
    tokenizer_cfg = dict(
        type='AutoTokenizer',
        weight_dir = tokenizer_dir   
    )
    # 初始化图像处理transform
    img_size=[224, 224]
    transform = Transforms(img_size=img_size)

    # 加载训练好的分词器, 会根据模型名称自动选择正确的分词规则（例如BPE、SentencePiece）
    tokenizer = MODELS.build_from_cfg(tokenizer_cfg)
    model = MODELS.build_from_cfg(llm_cfgs).eval().to(device)
    model.eval().to(device)
    print(f'模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    # 获取图像
    image = Image.open(img_path).convert('RGB')
    # 图像预处理
    process_img = transform.transform(image=np.array(image))['image'].transpose(2,0,1)      
    pixel_values = torch.tensor(process_img).to(device).unsqueeze(0).unsqueeze(0)     
    print(pixel_values.shape)


    # 初始化对话存储列表，用于存储上下文历史
    conversation = []
    # 创建流式输出器（边生成边打印）
    # skip_prompt=True 表示不重复打印用户输入, skip_special_tokens=True 表示跳过 <bos>、<eos>、<pad> 等特殊符号
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # <image> 请你描述一下图像中所展示的内容.
    prompt_iter = iter(lambda: input('👶: '), '')
    for prompt in prompt_iter:
        # 保留最近的历史对话（如果设置了historys）
        conversation = conversation[-historys:] if historys else []
        # 将用户当前输入加入到对话上下文
        conversation.append({"role": "user", "content": prompt.replace('<image>', model.llm_config.image_special_token)})
        # print(conversation)
        # 构造对话模板（Hugging Face 的chat模板，用于自动拼接system+user+assistant格式）
        templates = {
            "conversation": conversation,
            "tokenize": False, 
            "add_generation_prompt": True,
            # 是否开启CoT(本质是是否加<think></think>字段, =True不加, =False加)
            "enable_thinking": False
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
            temperature=0.65,                         # 温度采样系数（越高越随机）
            repetition_penalty=1.0,                   # 重复惩罚系数（>1抑制重复）
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