import torch.nn as nn
import clip

from heltonx.utils.utils import *
from pretrain.datasets.preprocess import Transforms
# 注册机制
from heltonx.utils.register import MODELS



@MODELS.register
class OpenAICLIP(nn.Module):
    """Learning Transferable Visual Models From Natural Language Supervision(CLIP): https://arxiv.org/abs/2103.00020
    """
    def __init__(self, pretrain_path):
        """初始化
            Args:
                img_size:      输入图像尺寸
                pretrain_path: CLIP的权重路径
        """
        super(OpenAICLIP, self).__init__()
        # model
        self.clip_model, self.preprocess = clip.load(pretrain_path)
        # half转全精度
        self.clip_model.eval().float()

        for param in self.clip_model.parameters():
            param.requires_grad_(False)

    def forward(self, x, type='image', device='cpu', *args, **kwargs):
        '''前向, 调用openai-clip图像和文本编码器(API接口, 外部调用此方法)
        '''
        self.device = device
        if type == 'image':
            embs = self._forward_img(x)
        if type == 'text':
            embs = self._forward_text(x)
        return embs

    
    def _forward_img(self, x):
        """前向, 调用openai-clip图像编码器
            Args:
                x: [B, 3, H, W]
            Returns:
                img_embs: [B, dim]
        """
        with torch.no_grad():
            img_embs = self.clip_model.encode_image(x)
            return img_embs
        

    def _forward_text(self, x):
        """前向, 调用openai-clip文本编码器
            Args: 
                device: 当前使用的设备
                x: list[str1, ..., strB]
            Returns:
                text_embs: [B, dim]
        """
        with torch.no_grad():
            # 文本先得转tokens
            tokens = clip.tokenize(x).to(self.device)
            text_embs = self.clip_model.encode_text(tokens)
            return text_embs












# for test only:
if __name__ == '__main__':
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # "clip_vitl14_768", "clip_vitb32_512"
    model_name = "clip_vitl14_768"
    img_size = [224, 224]
    img_dir = '/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/train'
    ckpt_path = f"/mnt/yht/code/HeltonPretrain/ckpts/{model_name}.pt"
    model = OpenAICLIP(ckpt_path).to(device)
    prompts = [
        "A cat with a happy expression.",
        "A photo of a friendly cat.",
        "A lovely little cat.",
        "a photo showing a cat in the scene",
        "a color picture of a cat, it is cute",
        "a photograph of a nice cat.",
        "a cropped photo of a cat, it is playful.",
        "I own a cat and I really like it.",
        "a picture of a cat taken long time ago.",
        "the picture showing a cat in the center.",
        "a picture of one cat in the scene.", # 
        "I adopted this cat several years ago.",
        "I took a picture of my cat.",
        "I love my cat and it loves me too.",
        "The cat in the picture is my friend's.",
        "This cat was a birthday present from my best friend.",
        "I accidentally snapped a picture of this cat.",
        "I petted my cat and she seemed to enjoy it.",
        "I called out to the cat and it rushed to me.",
        "My cat looking at the camera. It's the best memory ever.",
        "this cat used to be my best mate. Now it's gone.",
        "You're the best, my good cat.",
        "the cat is staring at me, want something to eat.",
        "My neighbour's cat, it looks funny.",
    ]
    img_dir = r'/mnt/yht/data/The_Oxford_IIIT_Pet_Dataset/images/valid'
    img_path = rf"{img_dir}/Maine_Coon/Maine_Coon_6.jpg"
    image = np.array(Image.open(img_path).convert('RGB'))

    # 图像预处理
    transform = Transforms(img_size)
    tensor_img = torch.tensor(transform.valid_transform(image=image)['image']).permute(2,0,1).unsqueeze(0).to(device)
    img_embs = model(tensor_img, 'image', device)
    text_embs = model(prompts, 'text', device)
    print(img_embs.shape, text_embs.shape)
    print(text_embs)
