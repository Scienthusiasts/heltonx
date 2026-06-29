import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageFile
import os
import cv2
import matplotlib.pyplot as plt
from heltonx.utils.register import DATASETS
from heltonx.utils.utils import seed_everything, worker_init_fn
from heltonx.utils.wrappers import DDPSafeDataset
from detection.datasets.base_dataset import BaseDetDataset
from detectionobb.datasets.preprocess_obb import OBBTransforms
from detectionobb.utils.obb_ops import xyxyxyxy2xywhr, xywhr2xyxyxyxy

ImageFile.LOAD_TRUNCATED_IMAGES = True


@DATASETS.register
class DOTADataset(BaseDetDataset):
    """DOTA / YOLO-OBB 数据集

    支持两种标注格式:
    1. label_format='yolo_obb' (默认): YOLO OBB 格式, 归一化坐标
       class_idx x1 y1 x2 y2 x3 y3 x4 y4
    2. label_format='dota_raw': DOTA 原始格式, 绝对像素坐标
       x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty

    输出旋转框格式: xywhr (cx, cy, w, h, angle_radians)

    类别映射根据 cat_names 自动生成: cat_names[i] → i

    Args:
        nc (int): 类别数
        cat_names (list): 类别名称列表 (按索引排序, 同时用于生成 class_mapping)
        img_dir (str): 图像目录
        label_dir (str): 标签目录
        img_size (list): [H, W] 网络输入尺寸
        mode (str): 'train' / 'valid'
        label_format (str): 'yolo_obb' (归一化) 或 'dota_raw' (绝对像素)
        mosaic_p (float): mosaic 增强概率
        mixup_p (float): mixup 增强概率
        filter_no_obb (bool): 是否滤除无 OBB 标注的图像 (True=滤除, False=保留全部)
    """

    def __init__(self, nc, cat_names, img_dir, label_dir, img_size, mode,
                 label_format='yolo_obb', mosaic_p=0, mixup_p=0, filter_no_obb=True):
        assert label_format in ('yolo_obb', 'dota_raw'), \
            f"label_format 必须是 'yolo_obb' 或 'dota_raw', 得到 '{label_format}'"
        self.mode = mode
        self.nc = nc
        self.cat_names = cat_names
        self.img_size = img_size
        self.tf = OBBTransforms(img_size)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.label_format = label_format
        self.mosaic_p = mosaic_p
        self.mixup_p = mixup_p
        self.filter_no_obb = filter_no_obb
        # 根据 cat_names 自动生成类别名→索引映射 (cat_names[i] → i)
        self.class_mapping = {name: idx for idx, name in enumerate(cat_names)}

        # 加载并过滤数据集文件列表
        self.img_files, self.dataset_num = self._load_and_filter_data()

    def _load_and_filter_data(self):
        """加载图像文件列表, 根据 filter_no_obb 决定是否过滤无标注的图像

        DDP 模式下各 rank 共享同一文件系统, 独立执行即可.
        不做 broadcast, 保证 dataset_num 始终等于 len(img_files).

        Returns:
            img_files (list[str]): 有效图像路径列表
            dataset_num (int): 有效图像数量
        """
        if self.filter_no_obb:
            img_files = self._scan_and_filter_img_files()
        else:
            img_files = self._scan_img_files()
        dataset_num = len(img_files)
        return img_files, dataset_num

    def _scan_img_files(self):
        """扫描图像目录, 只保留有对应 label 文件的图像 (不检查标注内容)

        当 filter_no_obb=False 时使用: 不验证标注是否包含有效 OBB,
        但仍需确保 label 文件存在, 否则 __getitem__ 无法读取标注.

        Returns:
            list[str]: 有对应 label 文件的图像路径列表
        """
        img_files = sorted([
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ], key=lambda x: os.path.basename(x))

        valid_img_files = []
        for img_path in img_files:
            label_path = os.path.join(self.label_dir,
                                      os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            if os.path.exists(label_path):
                valid_img_files.append(img_path)

        is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if is_main and len(valid_img_files) != len(img_files):
            print(f'[DOTADataset({self.mode})] label 存在性过滤: 从 {len(img_files)} 张图像中滤除 '
                  f'{len(img_files) - len(valid_img_files)} 张无 label 文件的图像, '
                  f'保留 {len(valid_img_files)} 张')

        return valid_img_files

    def _scan_and_filter_img_files(self):
        """扫描图像目录, 过滤掉没有有效 OBB 标注的图像

        Returns:
            list[str]: 有效图像路径列表
        """
        img_files = sorted([
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ], key=lambda x: os.path.basename(x))

        valid_img_files = []
        filtered_num = 0
        for img_path in img_files:
            if self._has_valid_obb(img_path):
                valid_img_files.append(img_path)
            else:
                filtered_num += 1

        if filtered_num > 0:
            # DDP 模式下只在 rank 0 打印, 避免重复日志
            is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)
            if is_main:
                print(f'[DOTADataset({self.mode})] 前置过滤: 从 {len(img_files)} 张图像中滤除 '
                      f'{filtered_num} 张无 OBB 标注的图像, 保留 {len(valid_img_files)} 张')

        return valid_img_files

    def _has_valid_obb(self, img_path):
        """检查图像是否有至少一个有效的 OBB 标注

        Args:
            img_path (str): 图像文件路径

        Returns:
            bool: 是否有有效 OBB 标注
        """
        label_path = os.path.join(self.label_dir,
                                  os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        if not os.path.exists(label_path):
            return False

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                # dota_raw 格式还需检查类别是否在 class_mapping 中
                if self.label_format == 'dota_raw':
                    class_name = parts[8]
                    if class_name not in self.class_mapping:
                        continue
                return True  # 至少一个有效标注即可

        return False

    def __getitem__(self, index):
        img_path = self.img_files[index]
        img_id = os.path.splitext(os.path.basename(img_path))[0]

        image, rboxes, labels = self.get_data_by_index(index)
        raw_size = [image.shape[0], image.shape[1]]  # [H, W] 从已加载图像获取
        image, rboxes, labels = self.augment(image, rboxes, labels)

        rboxes = np.array(rboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        return image.transpose(2, 0, 1), rboxes, labels, img_id, raw_size

    def get_data_by_index(self, index):
        """读取图像和标注, 转换为 xywhr 格式

        根据 self.label_format 选择读取方式:
        - 'yolo_obb': 读取归一化坐标, 需反归一化
        - 'dota_raw': 读取绝对像素坐标 + 类名, 需通过 self.class_mapping 映射类别
        """
        img_path = self.img_files[index]
        image = Image.open(img_path)
        image = np.array(image.convert('RGB'))
        H, W = image.shape[:2]

        label_path = os.path.join(self.label_dir,
                                   os.path.splitext(os.path.basename(img_path))[0] + '.txt')

        labels, rboxes = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue

                    if self.label_format == 'yolo_obb':
                        # YOLO OBB 格式: class_idx x1 y1 x2 y2 x3 y3 x4 y4 (归一化)
                        cls_idx = int(parts[0])
                        coords = [float(p) for p in parts[1:9]]
                        # 反归一化
                        coords[0::2] = [x * W for x in coords[0::2]]  # x 坐标
                        coords[1::2] = [y * H for y in coords[1::2]]  # y 坐标
                    else:
                        # DOTA 原始格式: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                        coords = [float(p) for p in parts[:8]]
                        class_name = parts[8]
                        if class_name not in self.class_mapping:
                            continue
                        cls_idx = self.class_mapping[class_name]

                    # 8 顶点 → xywhr
                    corners = np.array(coords, dtype=np.float32).reshape(1, 4, 2)
                    xywhr = xyxyxyxy2xywhr(torch.from_numpy(corners.reshape(1, 8))).numpy()[0]
                    labels.append(cls_idx)
                    rboxes.append(xywhr)

        labels = np.array(labels, dtype=np.int64) if labels else np.array([], dtype=np.int64)
        rboxes = np.array(rboxes, dtype=np.float32) if rboxes else np.zeros((0, 5), dtype=np.float32)

        return image, rboxes, labels

    def augment(self, image, rboxes, labels):
        """OBB 专用数据增强"""
        if self.mode == 'train':
            image, rboxes, labels = self.tf.train_aug(image, rboxes, labels)
            image, rboxes, labels = self._mosaic4(image, rboxes, labels, p=self.mosaic_p)
        image, rboxes, labels = self.tf.normal_aug(image, rboxes, labels)
        if self.mode == 'train':
            image, rboxes, labels = self._mixUp(image, rboxes, labels, p=self.mixup_p)
        return image, rboxes, labels

    def _mosaic4(self, image1, rboxes1, labels1, jitter=0.2, scale=0.5, p=0.5):
        if np.random.rand() >= p:
            return image1, rboxes1, labels1

        indexes = np.random.randint(self.dataset_num, size=3)
        images = [image1]
        rboxes_list = [rboxes1]
        labels_list = [labels1]
        for idx in indexes:
            img2, rb2, lb2 = self.get_data_by_index(idx)
            img2, rb2, lb2 = self.tf.train_aug(img2, rb2, lb2)
            images.append(img2)
            rboxes_list.append(rb2)
            labels_list.append(lb2)

        return self.tf.mosaic4(images, rboxes_list, labels_list, jitter, scale)

    def _mixUp(self, image1, rboxes1, labels1, p=0.5):
        if np.random.rand() >= p:
            return image1, rboxes1, labels1

        index2 = np.random.randint(self.dataset_num)
        image2, rboxes2, labels2 = self.get_data_by_index(index2)
        image2, rboxes2, labels2 = self.tf.train_aug(image2, rboxes2, labels2)
        image2, rboxes2, labels2 = self.tf.normal_aug(image2, rboxes2, labels2)

        images = [image1, image2]
        rboxes_list = [rboxes1, rboxes2]
        labels_list = [labels1, labels2]
        return self.tf.mixUp(images, rboxes_list, labels_list)

    @staticmethod
    def dataset_collate(batch):
        """OBB batch collate: rboxes 为 5 维 (xywhr)

        包含数据校验:
        - 过滤标签为负数的样本
        - 过滤 NaN/Inf 的 bbox
        - 过滤退化框 (w 或 h < 1)
        """
        images = []
        rboxes = []
        labels = []
        img_ids = []
        raw_sizes = []
        for img, rb, label, img_id, raw_size in batch:
            # 校验 bbox: 过滤 NaN/Inf 和退化框
            if len(rb) > 0:
                rb_np = np.asarray(rb, dtype=np.float32)
                if np.isnan(rb_np).any() or np.isinf(rb_np).any():
                    valid = ~(np.isnan(rb_np).any(axis=-1) | np.isinf(rb_np).any(axis=-1))
                    rb_np = rb_np[valid]
                    label = np.asarray(label, dtype=np.int64)[valid]
                if len(rb_np) > 0:
                    small_mask = (rb_np[:, 2] < 1) | (rb_np[:, 3] < 1)
                    if small_mask.any():
                        rb_np = rb_np[~small_mask]
                        label = np.asarray(label, dtype=np.int64)[~small_mask]
                rb = rb_np  # ★ 关键修复: 将过滤结果写回 rb
            # 校验标签: 过滤负数
            if len(label) > 0:
                label_np = np.asarray(label, dtype=np.int64)
                if label_np.min() < 0:
                    label = label_np.clip(0)  # ★ 写回label变量

            images.append(img)
            rboxes.append(torch.from_numpy(np.asarray(rb, dtype=np.float32)).type(torch.FloatTensor) if len(rb) > 0 else torch.zeros((0, 5), dtype=torch.float32))
            labels.append(torch.from_numpy(np.asarray(label, dtype=np.int64)).type(torch.LongTensor) if len(label) > 0 else torch.zeros((0,), dtype=torch.long))
            img_ids.append(img_id)
            raw_sizes.append(raw_size)

        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        return images, rboxes, labels, img_ids, raw_sizes

    def _vis_dota_batch(self, epoch, step, batch, save_dir='./vis_obb'):
        """可视化训练集一个 batch 的旋转框

        Args:
            epoch (int): 当前 epoch
            step (int): 当前 step
            batch: DataLoader 输出的 batch (images, rboxes, labels, img_ids, raw_sizes)
            save_dir (str): 可视化结果保存目录
        """
        images, rboxes, labels, img_ids, raw_sizes = batch
        # 图像均值标准差 (与 OBBTransforms 一致)
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])

        os.makedirs(save_dir, exist_ok=True)
        bs = len(images)
        cols = min(4, bs)
        rows = (bs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        if bs == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]

        for idx in range(bs):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]

            img = images[idx].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            img = np.clip(img * std + mean, 0, 1)
            rb = rboxes[idx].cpu().numpy()  # [N, 5] xywhr
            lb = labels[idx].cpu().numpy()  # [N]

            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'img_id: {img_ids[idx]}\n{len(rb)} objects', fontsize=10)

            if len(rb) > 0:
                # xywhr → 4 顶点
                rb_tensor = torch.from_numpy(rb)
                corners = xywhr2xyxyxyxy(rb_tensor).numpy()  # [N, 4, 2]
                for box, cls_idx, corner in zip(rb, lb, corners):
                    # 绘制旋转框 (4 条边)
                    poly = plt.Polygon(corner, fill=False, edgecolor='lime', linewidth=1.5)
                    ax.add_patch(poly)
                    # 标注类别名
                    name = self.cat_names[cls_idx] if cls_idx < len(self.cat_names) else str(cls_idx)
                    ax.text(corner[0, 0], corner[0, 1] - 2, name,
                            color='white', fontsize=7,
                            bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

        # 隐藏多余的子图
        for idx in range(bs, rows * cols):
            axes[idx // cols, idx % cols].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'epoch{epoch}_step{step}.jpg')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[VIS] 保存可视化结果: {save_path}')




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from functools import partial

    # DOTA 1.0 类别 (15 类) — 顺序即为类别索引
    cat_names = [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]

    # ============ 配置 (请修改为实际路径) ============
    # 方式1: 读取 YOLO OBB 格式 (归一化坐标, 需先用 convert_dota_to_yolo_obb.py 转换)
    # img_dir = r'/mnt/yht/data/DOTA/images/train'
    # label_dir = r'/mnt/yht/data/DOTA/labels/train'
    # label_format = 'yolo_obb'

    # 方式2: 读取 DOTA 原始格式 (绝对像素坐标, 无需转换)
    img_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/images'
    label_dir = r'/mnt/yht/data/DOTA-1.0-1.5_ss_size-1024_gap-200/1.0/trainval/annfiles'
    label_format = 'dota_raw'

    cfg = dict(
        dataset_cfg=dict(
            type="DOTADataset",
            nc=len(cat_names),
            cat_names=cat_names,
            img_dir=img_dir,
            label_dir=label_dir,
            img_size=[1024, 1024],
            mode='train',
            label_format=label_format,  # 'yolo_obb' 或 'dota_raw'
            mosaic_p=0.0,
            mixup_p=0.0,
            filter_no_obb=True
        ),
        bs=16,
        seed=42,
        shuffle=True,
    )

    dataset_cfg = cfg["dataset_cfg"]
    seed_everything(cfg["seed"])
    train_dataset = DATASETS.build_from_cfg(dataset_cfg)
    print(f'数据集大小: {train_dataset.__len__()}')
    print(f'标注格式: {train_dataset.label_format}')
    print(f'类别映射: {train_dataset.class_mapping}')

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["bs"],
        shuffle=cfg["shuffle"],
        num_workers=4,
        collate_fn=train_dataset.dataset_collate,
        worker_init_fn=partial(worker_init_fn, seed=cfg["seed"]),
    )

    # 输出数据格式 & 可视化
    for epoch in range(1, 3):
        for step, batch in enumerate(train_data_loader):
            images, rboxes, labels, img_ids, raw_sizes = batch
            print(f"epoch:{epoch}, batch:{step}, ")
            # print(f"images: {images.shape}, "
            #       f"rboxes: {[rb.shape for rb in rboxes]}, "
            #       f"labels: {[lb.shape for lb in labels]}")

            # if step % 5 == 0:
            #     # 可视化一个 batch 的旋转框
            #     train_dataset._vis_dota_batch(epoch, step, batch)
