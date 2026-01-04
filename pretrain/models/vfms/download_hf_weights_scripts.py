import os
from huggingface_hub import snapshot_download

def download_hf_model():
    # ---------------- 配置区域 ----------------
    # 1. 模型 ID (在 HuggingFace 网页复制)
    repo_id = "LiheYoung/pixio-vith16"
    
    # 2. 本地保存路径 (会自动创建文件夹)
    local_dir = r"F:\Desktop\git\heltonx\ckpts\hugging_face\pixio-vith16"
    
    # 3. Token (如果是公开模型填 None，私有模型/Llama3 等需填 Token)
    token = None 
    
    # 4. (可选) 国内镜像加速：如果下载速度慢，请取消下面这行的注释
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # ----------------------------------------

    print(f"准备下载模型: {repo_id}")
    print(f"保存位置: {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 关键：设为 False 才能得到真实的物理文件，而不是快捷方式
            resume_download=True,          # 关键：支持断点续传，网络断了重跑即可
            token=token,
            # allow_patterns=["*.safetensors", "*.json", "*.txt"], # 可选：只下载特定后缀的文件
            # ignore_patterns=["*.h5", "*.msgpack"],               # 可选：忽略不需要的文件
        )
        print("\n✅ 下载成功！")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")

if __name__ == "__main__":
    # 确保安装了库: pip install huggingface_hub
    download_hf_model()