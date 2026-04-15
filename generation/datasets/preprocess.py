import cv2
import albumentations as A







class Transforms():
    '''数据预处理/数据增强(基于albumentations库)
       https://albumentations.ai/docs/api_reference/full_reference/
    '''
    def __init__(self, img_size, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)):
        self.img_mean = img_mean
        self.img_std = img_std        
        # VAE(vae-kl-f8-d16):
        # self.img_mean = (0.5, 0.5, 0.5)
        # self.img_std = (0.5, 0.5, 0.5)
        # OpenAI-CLIP:
        # self.img_mean = (0.48145466, 0.4578275, 0.40821073)
        # self.img_std = (0.26862954, 0.26130258, 0.27577711)
        # DINOv3 LVD-1689M:
        # self.img_mean = (0.485, 0.456, 0.406)
        # self.img_std = (0.229, 0.224, 0.225)
        # DINOv3 SAT-493M:
        # self.img_mean = (0.430, 0.411, 0.296)
        # self.img_std = (0.213, 0.156, 0.143)

        self.pad_value = [128,128,128]
        # 训练时增强
        self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                # 随机尺寸裁剪并缩放到固定尺寸
                A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.99, 1), ratio=(0.99, 1), p=1, interpolation=cv2.INTER_AREA),    
                # 最长边限制为img_size
                A.LongestMaxSize(max_size=img_size[0]),
                A.PadIfNeeded(img_size[0], img_size[1], border_mode=cv2.BORDER_ISOLATED, value=self.pad_value),
                # 图像归一化
                A.Normalize(mean=self.img_mean, std=self.img_std),
            ])
        
        self.mask_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # 图像归一化
                A.Normalize(mean=self.img_mean, std=self.img_std),
            ],
            is_check_shapes=False)
