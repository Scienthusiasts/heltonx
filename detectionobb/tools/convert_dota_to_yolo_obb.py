"""DOTA 格式转 YOLO OBB 格式工具

DOTA 原始格式:  x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
YOLO OBB 格式:  class_idx x1_norm y1_norm x2_norm y2_norm x3_norm y3_norm x4_norm y4_norm

类别映射根据 cat_names 自动生成: cat_names[i] → i

目录结构:
    DOTA/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train_original/
        └── val_original/

执行后生成:
    DOTA/
    └── labels/
        ├── train/
        └── val/
"""

import cv2
from pathlib import Path
from tqdm import tqdm


def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir, cat_names):
    """转换单张图像的 DOTA 标注为 YOLO OBB 格式

    Args:
        cat_names (list): 类别名称列表, cat_names[i] 对应索引 i
    """
    class_mapping = {name: idx for idx, name in enumerate(cat_names)}

    orig_label_path = orig_label_dir / f"{image_name}.txt"
    save_path = save_dir / f"{image_name}.txt"

    with orig_label_path.open("r") as f, save_path.open("w") as g:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            class_name = parts[8]
            if class_name not in class_mapping:
                continue
            class_idx = class_mapping[class_name]
            coords = [float(p) for p in parts[:8]]
            normalized_coords = [
                coords[i] / image_width if i % 2 == 0 else coords[i] / image_height
                for i in range(8)
            ]
            formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]
            g.write(f"{class_idx} {' '.join(formatted_coords)}\n")


def convert_dota_to_yolo_obb(dota_root_path, cat_names):
    """将 DOTA 数据集标注转换为 YOLO OBB 格式

    Args:
        dota_root_path (str): DOTA 数据集根目录
        cat_names (list): 类别名称列表, cat_names[i] 对应索引 i
    """
    dota_root_path = Path(dota_root_path)

    for phase in ["train", "val"]:
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        if not image_dir.exists():
            print(f"[WARN] {image_dir} 不存在, 跳过 {phase}")
            continue

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in tqdm(image_paths, desc=f"Processing {phase} images"):
            if image_path.suffix not in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"[WARN] 无法读取图像: {image_path}")
                continue
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir, cat_names)

    print(f"DOTA -> YOLO OBB 转换完成: {dota_root_path}")


if __name__ == "__main__":
    import argparse

    # DOTA 1.0 类别 (15 类) — 顺序即为类别索引
    default_cat_names = [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]

    parser = argparse.ArgumentParser(description="DOTA -> YOLO OBB 格式转换")
    parser.add_argument("--dota_root", type=str, required=True, help="DOTA 数据集根目录")
    parser.add_argument("--cat_names", nargs='+', default=default_cat_names,
                        help="类别名称列表 (空格分隔, 顺序即为索引)")
    args = parser.parse_args()
    convert_dota_to_yolo_obb(args.dota_root, args.cat_names)
