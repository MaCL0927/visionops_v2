"""
Stage 1: 数据预处理
- 支持分类、目标检测等视觉任务
- 数据增强（离线增强，提升多样性）
- 划分训练集/验证集
- 生成数据集统计报告
"""
import os
import json
import yaml
import shutil
import random
import numpy as np
from pathlib import Path
from datetime import datetime


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collect_image_files(raw_dir: str) -> list:
    """递归收集所有图像文件"""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for p in Path(raw_dir).rglob("*"):
        if p.suffix.lower() in extensions:
            files.append(str(p))
    return files


def split_dataset(files: list, val_split: float, seed: int = 42) -> tuple:
    """按比例划分训练集和验证集"""
    random.seed(seed)
    random.shuffle(files)
    n_val = max(1, int(len(files) * val_split))
    return files[n_val:], files[:n_val]


def process_image(src_path: str, dst_path: str, img_size: list, augment: bool = False):
    """处理单张图像：resize + 可选增强"""
    import cv2

    img = cv2.imread(src_path)
    if img is None:
        return False

    # Resize
    h, w = img_size[0], img_size[1]
    img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # 简单增强（训练集）
    if augment:
        # 随机水平翻转
        if random.random() > 0.5:
            img_resized = cv2.flip(img_resized, 1)
        # 随机亮度调整
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            img_resized = np.clip(img_resized.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, img_resized)
    return True


def main():
    cfg = load_config("pipeline/configs/data.yaml")
    preprocess_cfg = cfg.get("preprocess", {})
    img_size = preprocess_cfg.get("img_size", [640, 640])
    val_split = preprocess_cfg.get("val_split", 0.2)
    augment = preprocess_cfg.get("augment", True)

    raw_dir = cfg["paths"]["raw_data"]
    processed_dir = cfg["paths"]["processed"]
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")

    print(f"扫描原始数据: {raw_dir}")
    if not os.path.exists(raw_dir):
        print(f"原始数据目录不存在: {raw_dir}，创建空目录")
        os.makedirs(raw_dir, exist_ok=True)
        # 生成dummy统计文件
        stats = {"total_images": 0, "train": 0, "val": 0, "note": "无原始数据"}
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        with open("data/processed/dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print("生成空数据集统计，请将原始数据放入 data/raw/ 目录")
        return

    all_files = collect_image_files(raw_dir)
    print(f"发现 {len(all_files)} 张图像")

    if not all_files:
        print("没有找到图像文件！请检查 data/raw/ 目录")
        return

    train_files, val_files = split_dataset(all_files, val_split)
    print(f"划分: 训练集 {len(train_files)} | 验证集 {len(val_files)}")

    # 检查是否有子目录结构（分类任务）
    has_class_dirs = any(
        os.path.dirname(f) != raw_dir
        for f in all_files[:5]
    )

    success_count = 0
    fail_count = 0

    for split_name, files, do_augment in [
        ("train", train_files, augment),
        ("val", val_files, False)
    ]:
        split_dir = os.path.join(processed_dir, split_name)
        print(f"\n处理 {split_name} 集...")

        for src_path in files:
            if has_class_dirs:
                # 保留类别子目录结构
                rel_path = os.path.relpath(src_path, raw_dir)
                dst_path = os.path.join(split_dir, rel_path)
            else:
                dst_path = os.path.join(split_dir, os.path.basename(src_path))

            if process_image(src_path, dst_path, img_size, do_augment):
                success_count += 1
            else:
                fail_count += 1

        print(f"  {split_name}: 完成 {success_count} 张")

    # 生成数据集统计
    stats = {
        "total_images": len(all_files),
        "train": len(train_files),
        "val": len(val_files),
        "val_split": val_split,
        "img_size": img_size,
        "augment": augment,
        "has_class_structure": has_class_dirs,
        "success": success_count,
        "failed": fail_count,
        "processed_at": datetime.now().isoformat(),
    }

    with open("data/processed/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 数据预处理完成!")
    print(f"  总图像: {len(all_files)} | 成功: {success_count} | 失败: {fail_count}")
    print(f"  统计报告: data/processed/dataset_stats.json")


if __name__ == "__main__":
    main()
