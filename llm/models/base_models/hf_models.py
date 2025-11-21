from .blocks import *
# 注册机制
from heltonx.utils.register import MODELS
from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel


# AutoModelForCausalLM.from_pretrained(...) 返回的是实例对象, 因此注册一个注册构造函数
@MODELS.register("AutoModelForCausalLM")
def AutoModelForCausalLM_builder(weight_dir, *args, **kwargs):
    """
        weight_dir: huggingface 模型权重所在dir
    """
    return AutoModelForCausalLM.from_pretrained(weight_dir, trust_remote_code=True, *args, **kwargs)


@MODELS.register("AutoTokenizer")
def AutoTokenizer_builder(weight_dir, *args, **kwargs):
    """
        weight_dir: huggingface 模型权重所在dir
    """
    return AutoTokenizer.from_pretrained(weight_dir, *args, **kwargs)




@MODELS.register
class OpenAICLIPImgEncoder(nn.Module):
    """Learning Transferable Visual Models From Natural Language Supervision(CLIP): https://arxiv.org/abs/2103.00020
    """
    def __init__(self, weight_dir):
        """初始化
            Args:
                img_size:      输入图像尺寸
                pretrain_path: CLIP的权重路径
        """
        super(OpenAICLIPImgEncoder, self).__init__()
        self.model = CLIPModel.from_pretrained(weight_dir)


    def forward(self, x):
        '''前向, 调用openai-clip图像编码器
        '''
        embeddings = self.model.vision_model(pixel_values=x)
        # 从索引1开始是为了忽略cls_token
        img_embedding = embeddings.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding
    


@MODELS.register
class DINOv3(nn.Module):
    """Learning Transferable Visual Models From Natural Language Supervision(CLIP): https://arxiv.org/abs/2103.00020
    """
    def __init__(self, weight_dir):
        """初始化
            Args:
                img_size:      输入图像尺寸
                pretrain_path: CLIP的权重路径
        """
        super(DINOv3, self).__init__()
        self.model = AutoModel.from_pretrained(weight_dir)


    def forward(self, x):
        '''前向, 调用openai-clip图像编码器
        '''
        embeddings = self.model(x)
        # 从索引5开始是为了忽略之前的special_tokens
        img_embedding = embeddings.last_hidden_state[:, 5:, :].squeeze()
        return img_embedding
    



if __name__ == '__main__':
    from heltonx.utils.register import MODELS
    cfg = dict(
        type='OpenAICLIPImgEncoder',
        weight_dir = 'ckpts/hugging_face/clip-vit-base-patch16'
    )
    # cfg = dict(
    #     type='DINOv3',
    #     weight_dir = 'ckpts/hugging_face/DINOv3s'
    # )
    vision_encoder = MODELS.build_from_cfg(cfg)

    img = torch.randn(1, 3, 224, 224)
    output = vision_encoder(img)
    print(output.shape)


