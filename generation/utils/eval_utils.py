# coding=utf-8
import os
import json
import torch
import random
import numpy as np

from pretrain.utils.metrics import *
from matplotlib.gridspec import GridSpec
# 多卡并行训练:
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from heltonx.utils.register import EVALPIPELINES


@EVALPIPELINES.register
class GenerationEvalPipeline():
    '''一个epoch的评估(基于验证集)
    '''
    def __call__(self, runner, model=None):
        # 直接从传入的类中获取参数(避免每个任务的特殊化):
        if model==None:
            model = runner.model
        epoch = runner.cur_epoch
        log_dir = runner.log_dir

        # 图像均值 标准差
        # mean = np.array([0.485, 0.456, 0.406]) 
        # std = np.array([[0.229, 0.224, 0.225]]) 
        mean = np.array(runner.train_dataset.img_mean)
        std = np.array(runner.train_dataset.img_std).reshape(1, -1)

        model.eval()
        # 图像生成
        samples = model(bs=64, return_loss=False)

        # 可视化
        generate_images = samples
        fig, axes = plt.subplots(8, 8, figsize=(10, 10))  # Create an 8x8 grid of subplots
        for i, ax in enumerate(axes.flat):
            gen_img_norm = generate_images[i].transpose((1,2,0))
            gen_img = gen_img_norm * std + mean
            ax.imshow(gen_img) 
            ax.axis("off")  

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"epoch_{epoch}.png"), dpi=300)

        # TODO:
        # 评估结果以字典形式返回(统一格式, key的前缀一定有'val_')
        evaluations = dict(
            ssim=epoch
        )
        # 后续保存best_ckpt以val_flag_metric为参考
        flag_metric_name = "ssim"
        return evaluations, flag_metric_name
    






@EVALPIPELINES.register
class ConditionGenerationEvalPipeline():
    '''一个epoch的评估(基于验证集)
    '''
    def __call__(self, runner, model=None):
        bs = 64
        # 直接从传入的类中获取参数(避免每个任务的特殊化):
        if model==None:
            model = runner.model
        epoch = runner.cur_epoch
        log_dir = runner.log_dir

        '''永远读取第2条数据的caption'''
        prompts = runner.train_dataset.samples[1]['conversations'][1]['content']
        tokenizer = runner.train_dataset.tokenizer
        max_len = runner.train_dataset.max_length
        # 进行tokenize + 截断
        input_ids = tokenizer(prompts).input_ids
        input_ids = input_ids[:max_len]
        # padding补齐
        input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
        prompts_tokens = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).repeat(bs, 1)

        # 图像均值 标准差
        mean = np.array([0.485, 0.456, 0.406]) 
        std = np.array([[0.229, 0.224, 0.225]]) 

        model.eval()
        # 图像生成
        samples = model(bs=bs, c_tokens=prompts_tokens, return_loss=False)

        # 可视化
        generate_images = samples
        B, C, H, W = generate_images.shape
        print(generate_images.shape)
        fig, axes = plt.subplots(8, 8, figsize=(10, 10))  # Create an 8x8 grid of subplots
        for i, ax in enumerate(axes.flat):
            gen_img_norm = generate_images[i].reshape(C, H, W).transpose((1,2,0))
            # figtest = reverse_transform(torch.from_numpy(generate_image))
            gen_img = gen_img_norm * std + mean
            ax.imshow(gen_img) 
            ax.axis("off")  

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"epoch_{epoch}.png"), dpi=300)

        # TODO:
        # 评估结果以字典形式返回(统一格式, key的前缀一定有'val_')
        evaluations = dict(
            ssim=epoch
        )
        # 后续保存best_ckpt以val_flag_metric为参考
        flag_metric_name = "ssim"
        return evaluations, flag_metric_name
    





@EVALPIPELINES.register
class MaskGenerationEvalPipeline():
    '''一个epoch的评估(基于验证集)，将生成的 16 通道 Mask 合并为彩色图并保留所有原始噪声
    '''
    def __call__(self, runner, model=None):
        bs = 64
        if model is None:
            model = runner.model
        epoch = runner.cur_epoch
        log_dir = runner.log_dir

        model.eval()
        samples = model(bs=bs, return_loss=False)

        # 2. 数据后处理：从 [-1, 1] 线性映射回 [0, 255]
        # 注意：这里只做线性放缩和边界裁剪，完全保留了模型产生的所有中间值和高斯噪声
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        generate_masks = np.clip((samples + 1.0) / 2.0 * 255.0, 0, 255)

        B = generate_masks.shape[0]

        # 3. 显式设置整个大画板的背景色为白色
        fig = plt.figure(figsize=(12, 12), facecolor='white')
        grid_size = int(np.ceil(np.sqrt(B)))
        cmap = plt.get_cmap('tab20')

        for idx in range(B):
            mask_tensor = generate_masks[idx] # [16, H, W]
            
            # 提取前景掩膜 (忽略索引0的背景)
            fg_masks = mask_tensor[1:] # [15, H, W]

            # 获取每个像素所属的类别索引 (1~15)
            # 在有噪声的情况下，这一步会把每个像素分配给噪声值最大的那个类别
            cls_idx = np.argmax(fg_masks, axis=0) + 1
            
            # 获取该像素的掩膜强度 (0.0 ~ 1.0)
            # 噪声越大的地方强度越高，越亮；没有目标的干净背景趋近于 0，呈黑色
            intensity = np.max(fg_masks, axis=0) / 255.0

            # 映射到 RGB 颜色，并乘以强度
            rgba = cmap((cls_idx - 1) / 14.0)
            rgb = rgba[..., :3]
            final_vis = rgb * intensity[..., np.newaxis]
            
            ax = plt.subplot(grid_size, grid_size, idx + 1)
            ax.imshow(final_vis)

            # 隐藏坐标刻度和文字，但不使用 axis("off") 这样就能保留物理边框
            ax.set_xticks([])
            ax.set_yticks([])

        # 稍微调大 wspace 和 hspace，利用白色的 Figure 背景形成物理上的“白边”隔离
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.05, hspace=0.05)
        
        # 确保保存目录存在
        os.makedirs(log_dir, exist_ok=True)
        save_name = os.path.join(log_dir, f'epoch{epoch}.jpg')
        
        # 保存时强制指定背景为白色 (facecolor='white')
        plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # 评估结果以字典形式返回(统一格式, key的前缀一定有'val_')
        evaluations = dict(
            ssim=epoch
        )
        # 后续保存best_ckpt以val_flag_metric为参考
        flag_metric_name = "ssim"
        return evaluations, flag_metric_name
    




@EVALPIPELINES.register
class ControlGenerationEvalPipeline():
    '''一个epoch的评估(基于验证集)，提取真实的 txt 标注作为条件生成图像，并做对比可视化
    '''
    def __call__(self, runner, model=None):
        if model == None:
            model = runner.model
        epoch = runner.cur_epoch
        log_dir = runner.log_dir
        
        device = next(model.parameters()).device
        dataset = runner.train_dataset

        # 图像均值 标准差 (用于生成图像的反归一化可视化)
        mean = np.array(dataset.img_mean)
        std = np.array(dataset.img_std).reshape(1, -1)
        model.eval()
        
        # ==========================================
        # 1. 随机选取一批 txt 标注文件
        # ==========================================
        eval_bs = 25 # 建议为 16 (4x4网格)，太大显存和画板放不下
        total_data = len(dataset.data_list)
        # 防止数据集总数小于 eval_bs
        sample_indices = random.sample(range(total_data), min(eval_bs, total_data))
        sampled_data = [dataset.data_list[i] for i in sample_indices]
        
        # ==========================================
        # 2. obb2mask：将 txt 转化为模型所需的条件 Mask
        # ==========================================
        mask_H, mask_W = dataset.mask_size
        ori_H, ori_W = dataset.ori_img_size
        num_classes = dataset.num_classes
        class2idx = dataset.class2idx
        
        batch_masks = []
        fnames = []
        
        for fname, _, txt_path in sampled_data:
            fnames.append(fname)
            # 在原图尺寸上绘制掩膜，保留几何精度
            obj_masks_ori = np.zeros((num_classes, ori_H, ori_W), dtype=np.uint8)
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9: continue
                cls_name = parts[8]
                if cls_name not in class2idx: continue
                
                channel_idx = class2idx[cls_name] - 1 
                try:
                    pts = [float(p) for p in parts[:8]]
                except ValueError: continue 
                
                pts_x, pts_y = np.array(pts[0::2]), np.array(pts[1::2])
                poly_pts = np.stack([pts_x, pts_y], axis=1).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(obj_masks_ori[channel_idx], [poly_pts], color=255)

            # 下采样至潜空间尺寸 (mask_size) 并产生软边缘
            obj_masks_resized = np.zeros((num_classes, mask_H, mask_W), dtype=np.float32)
            for i in range(num_classes):
                if np.any(obj_masks_ori[i]): 
                    obj_masks_resized[i] = cv2.resize(
                        obj_masks_ori[i], (mask_W, mask_H), interpolation=cv2.INTER_AREA
                    )

            # 计算背景
            fg_max = np.max(obj_masks_resized, axis=0)
            bg_mask = np.expand_dims(255.0 - fg_max, axis=0)
            
            # 拼接并直接转换为张量，归一化到 [-1, 1]
            # 💡 注意：评估阶段不需要应用 mask_transform 数据增强，保证与真实标注完全对齐
            mask = np.concatenate([bg_mask, obj_masks_resized], axis=0)
            mask_tensor = torch.from_numpy(mask).type(torch.FloatTensor) / 255. * 2. - 1.
            batch_masks.append(mask_tensor)

        # 组合为 Batch [B, 16, h, w] 并送入相应显卡
        mask_conditions = torch.stack(batch_masks, dim=0).to(device)

        # ==========================================
        # 3. 执行条件生成 (推理)
        # ==========================================
        # 组装 batch_data，第一项原图设为 None，第二项为条件 Mask
        batch_data = [None, mask_conditions]
        
        with torch.no_grad():
            # 调用 LDM 的 forward，返回的样本通常是 numpy 数组或需要转换
            samples = model(batch_data=batch_data, return_loss=False)
            if isinstance(samples, torch.Tensor):
                samples = samples.cpu().numpy()

        # ==========================================
        # 4. 左右对照可视化：左侧 Condition Mask，右侧 Generated Image
        # ==========================================
        B = mask_conditions.shape[0]
        masks_np = mask_conditions.cpu().numpy()
        grid_n = int(np.ceil(np.sqrt(B)))
        
        fig = plt.figure(figsize=(grid_n * 5, grid_n * 2.5), facecolor='white')
        gs = GridSpec(1, 2, figure=fig, wspace=0.1)
        
        gs_left = gs[0].subgridspec(grid_n, grid_n, wspace=0.05, hspace=0.2)
        gs_right = gs[1].subgridspec(grid_n, grid_n, wspace=0.05, hspace=0.2)
        cmap = plt.get_cmap('tab20')
        
        for idx in range(B):
            row = idx // grid_n
            col = idx % grid_n
            fname = fnames[idx]
            
            # --------- 左侧：渲染条件 Mask ---------
            ax_mask = fig.add_subplot(gs_left[row, col])
            mask_tensor = masks_np[idx] 
            fg_masks = mask_tensor[1:] 
            cls_idx = np.argmax(fg_masks, axis=0) + 1
            intensity = (np.max(fg_masks, axis=0) + 1) / 2.0  # [-1, 1] 映射回 [0, 1]
            
            rgba = cmap((cls_idx - 1) / 14.0)
            rgb = rgba[..., :3]
            mask_vis = rgb * intensity[..., np.newaxis]
            
            ax_mask.imshow(mask_vis)
            ax_mask.set_xticks([])
            ax_mask.set_yticks([])
            ax_mask.set_title(f"Mask: {fname}", fontsize=8, pad=3)
            for spine in ax_mask.spines.values():
                spine.set_edgecolor('lightgray')

            # --------- 右侧：渲染模型生成的图像 ---------
            ax_img = fig.add_subplot(gs_right[row, col])
            # 根据您提供的反归一化逻辑
            gen_img_norm = samples[idx].transpose((1, 2, 0))
            gen_img = gen_img_norm * std + mean 
            
            ax_img.imshow(gen_img)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title(f"Gen: {fname}", fontsize=8, pad=3)
            for spine in ax_img.spines.values():
                spine.set_edgecolor('lightgray')

        # 增加主标题
        # fig.text(0.25, 0.98, f"Condition Masks (Epoch {epoch})", ha='center', fontsize=14, fontweight='bold')
        # fig.text(0.75, 0.98, f"Generated Images (Epoch {epoch})", ha='center', fontsize=14, fontweight='bold')

        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, f"epoch_{epoch}_eval.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # ==========================================
        # 5. 返回评估指标
        # ==========================================
        evaluations = dict(
            ssim=epoch # 占位符，可以之后替换为真实的 FID / SSIM 计算逻辑
        )
        flag_metric_name = "ssim"
        
        return evaluations, flag_metric_name
    






@EVALPIPELINES.register
class ClassGenerationEvalPipeline():
    '''一个epoch的评估(基于验证集)，随机采样类别 ID 作为条件生成图像，并做网格可视化
    '''
    def __call__(self, runner, model=None):
        if model == None:
            model = runner.model
        epoch = runner.cur_epoch
        log_dir = runner.log_dir
        
        device = next(model.parameters()).device
        dataset = runner.train_dataset

        # 图像均值 标准差 (用于生成图像的反归一化可视化)
        mean = np.array(dataset.img_mean)
        std = np.array(dataset.img_std).reshape(1, -1)
        model.eval()
        
        # ==========================================
        # 1. 随机生成一批类别 ID 条件
        # ==========================================
        eval_bs = 64 # 建议为 16 或 25 (网格大小)
        num_classes = dataset.num_classes
        
        # 随机采样 eval_bs 个类别 ID，范围从 0 到 num_classes - 1
        random_labels = torch.randint(0, num_classes, (eval_bs,), device=device, dtype=torch.long)

        # 尝试获取类别名称映射，如果 dataset 提供了 classes 列表，可视化会更直观
        if hasattr(dataset, 'classes'):
            idx2class = {i: cls_name for i, cls_name in enumerate(dataset.classes)}
        else:
            idx2class = {i: str(i) for i in range(num_classes)}

        # ==========================================
        # 2. 执行条件生成 (推理)
        # ==========================================
        # 组装 batch_data，第一项原图设为 None，第二项为类别条件 labels
        batch_data = [None, random_labels]
        
        with torch.no_grad():
            # 调用 ClassLDM 的 forward，返回的样本通常是 numpy 数组或需要转换
            samples = model(batch_data=batch_data, return_loss=False)
            if isinstance(samples, torch.Tensor):
                samples = samples.cpu().numpy()

        # ==========================================
        # 3. 网格可视化：显示生成的图像及对应的类别
        # ==========================================
        labels_np = random_labels.cpu().numpy()
        grid_n = int(np.ceil(np.sqrt(eval_bs)))
        
        fig, axes = plt.subplots(grid_n, grid_n, figsize=(grid_n * 3, grid_n * 3), facecolor='white')
        
        # 防止因 eval_bs=1 导致 axes 不是二维数组
        if eval_bs == 1:
            axes = np.array([axes])
            
        for i, ax in enumerate(axes.flat):
            if i < eval_bs:
                # 反归一化
                gen_img_norm = samples[i].transpose((1, 2, 0))
                gen_img = (gen_img_norm * std + mean).clip(0, 1) 
                
                ax.imshow(gen_img)
                ax.axis("off")
                
                # 获取类别名称并设置为标题
                cls_id = labels_np[i]
                cls_name = idx2class.get(cls_id, str(cls_id))
                ax.set_title(f"Class: {cls_name}", fontsize=10, pad=4, fontweight='bold')
            else:
                # 隐藏多余的空子图
                ax.axis("off")

        # 增加主标题
        fig.suptitle(f"Class-Conditioned Generation (Epoch {epoch})", fontsize=16, fontweight='bold', y=0.98)

        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, f"epoch_{epoch}_eval.png")
        plt.tight_layout()
        plt.subplots_adjust(top=0.92) # 给主标题留出空间
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # ==========================================
        # 4. 返回评估指标
        # ==========================================
        evaluations = dict(
            ssim=epoch # 占位符，可以之后替换为真实的 FID / IS 计算逻辑
        )
        flag_metric_name = "ssim"
        
        return evaluations, flag_metric_name