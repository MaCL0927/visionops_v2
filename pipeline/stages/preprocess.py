"""
Stage 1: 分类数据预处理

支持两种分类数据目录格式：

格式 A：未划分数据集
data/raw/
├── class_a/
│   ├── xxx.jpg
│   └── ...
└── class_b/
    ├── xxx.jpg
    └── ...

格式 B：已划分数据集
data/raw/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/

输出格式：
data/processed/
├── train/
│   ├── class_a/
│   └── class_b/
├── val/
│   ├── class_a/
│   └── class_b/
├── dataset_stats.json
└── class_names.yaml
"""

import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def detect_raw_format(raw_dir: Path) -> str:
    """
    判断原始数据格式。

    返回：
    - split: data/raw/train/class_x + data/raw/val/class_x
    - class_dirs: data/raw/class_x
    - flat: data/raw/*.jpg，不推荐用于分类
    - empty: 没有图片
    """
    if not raw_dir.exists():
        return "empty"

    train_dir = raw_dir / "train"
    val_dir = raw_dir / "val"

    has_train_images = train_dir.exists() and any(is_image_file(p) for p in train_dir.rglob("*"))
    has_val_images = val_dir.exists() and any(is_image_file(p) for p in val_dir.rglob("*"))

    if has_train_images and has_val_images:
        return "split"

    direct_class_dirs = [
        p for p in raw_dir.iterdir()
        if p.is_dir() and p.name not in {"train", "val", "test"}
    ]
    if direct_class_dirs:
        has_images_in_class_dirs = any(
            is_image_file(img)
            for class_dir in direct_class_dirs
            for img in class_dir.rglob("*")
        )
        if has_images_in_class_dirs:
            return "class_dirs"

    flat_images = [p for p in raw_dir.iterdir() if is_image_file(p)]
    if flat_images:
        return "flat"

    return "empty"


def collect_split_dataset(raw_dir: Path) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """
    收集已划分好的分类数据：
    data/raw/train/class_x/*.jpg
    data/raw/val/class_x/*.jpg
    """
    train_by_class = collect_class_dirs(raw_dir / "train")
    val_by_class = collect_class_dirs(raw_dir / "val")
    return train_by_class, val_by_class


def collect_class_dirs(root_dir: Path) -> Dict[str, List[Path]]:
    """
    收集 root_dir/class_name/*.jpg。
    返回：
    {
        "class_a": [Path(...), ...],
        "class_b": [Path(...), ...]
    }
    """
    by_class: Dict[str, List[Path]] = defaultdict(list)

    if not root_dir.exists():
        return by_class

    for class_dir in sorted(root_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        for img_path in sorted(class_dir.rglob("*")):
            if is_image_file(img_path):
                by_class[class_name].append(img_path)

    return dict(by_class)


def split_class_dirs_dataset(
    raw_dir: Path,
    val_split: float,
    seed: int = 42,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """
    对 data/raw/class_name/*.jpg 格式进行按类别分层划分。
    """
    all_by_class = collect_class_dirs(raw_dir)

    rng = random.Random(seed)
    train_by_class: Dict[str, List[Path]] = {}
    val_by_class: Dict[str, List[Path]] = {}

    for class_name, files in all_by_class.items():
        files = list(files)
        rng.shuffle(files)

        if len(files) <= 1:
            # 只有 1 张图时无法合理划分，先放入训练集
            n_val = 0
        else:
            n_val = max(1, int(len(files) * val_split))

        val_by_class[class_name] = files[:n_val]
        train_by_class[class_name] = files[n_val:]

    return train_by_class, val_by_class


def copy_or_resize_image(
    src_path: Path,
    dst_path: Path,
    img_size: List[int],
    resize: bool = True,
) -> bool:
    """
    复制或 resize 单张图像。

    分类训练通常可以在 Dataset transform 中 resize。
    但为了让 processed 数据完全标准化，这里默认 resize 到配置尺寸。
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not resize:
        shutil.copy2(src_path, dst_path)
        return True

    try:
        import cv2
    except ImportError:
        # 如果没有 cv2，退化为直接复制
        shutil.copy2(src_path, dst_path)
        return True

    img = cv2.imread(str(src_path))
    if img is None:
        return False

    h, w = int(img_size[0]), int(img_size[1])
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return bool(cv2.imwrite(str(dst_path), img))


def export_split(
    split_name: str,
    by_class: Dict[str, List[Path]],
    processed_dir: Path,
    img_size: List[int],
    resize: bool,
) -> Tuple[int, int, Dict[str, int]]:
    """
    导出 train 或 val。
    返回：success_count, fail_count, class_counts
    """
    success_count = 0
    fail_count = 0
    class_counts: Dict[str, int] = {}

    split_dir = processed_dir / split_name

    for class_name, files in sorted(by_class.items()):
        class_counts[class_name] = 0

        for idx, src_path in enumerate(files):
            # 避免不同子目录中同名文件覆盖
            dst_name = f"{src_path.stem}_{idx:06d}{src_path.suffix.lower()}"
            dst_path = split_dir / class_name / dst_name

            ok = copy_or_resize_image(
                src_path=src_path,
                dst_path=dst_path,
                img_size=img_size,
                resize=resize,
            )

            if ok:
                success_count += 1
                class_counts[class_name] += 1
            else:
                fail_count += 1

    return success_count, fail_count, class_counts


def write_class_names(processed_dir: Path, class_names: List[str]) -> None:
    payload = {
        "task": "classification",
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    with open(processed_dir / "class_names.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def write_edge_class_names(class_names: List[str]) -> None:
    """
    同步写一份给边缘端后续使用。
    目前只是生成文件，不改 engine.py。
    """
    edge_dir = Path("edge/runtime")
    edge_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "task": "classification",
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    with open(edge_dir / "class_names.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    cfg = load_config("pipeline/configs/data.yaml")

    preprocess_cfg = cfg.get("preprocess", {})
    paths_cfg = cfg.get("paths", {})

    img_size = preprocess_cfg.get("img_size", [224, 224])
    val_split = float(preprocess_cfg.get("val_split", 0.2))
    resize = bool(preprocess_cfg.get("resize", True))

    raw_dir = Path(paths_cfg.get("raw_data", "data/raw/"))
    processed_dir = Path(paths_cfg.get("processed", "data/processed/"))

    print("=" * 60)
    print("分类数据预处理开始")
    print("=" * 60)
    print(f"原始数据目录: {raw_dir}")
    print(f"输出数据目录: {processed_dir}")
    print(f"图像尺寸: {img_size}")
    print(f"验证集比例: {val_split}")
    print(f"是否 resize: {resize}")

    raw_format = detect_raw_format(raw_dir)
    print(f"识别到数据格式: {raw_format}")

    reset_dir(processed_dir / "train")
    reset_dir(processed_dir / "val")
    processed_dir.mkdir(parents=True, exist_ok=True)

    if raw_format == "empty":
        raw_dir.mkdir(parents=True, exist_ok=True)
        stats = {
            "task": "classification",
            "status": "empty",
            "note": "未找到分类图像，请将数据放入 data/raw/class_name/ 或 data/raw/train|val/class_name/。",
            "total_images": 0,
            "num_classes": 0,
            "class_names": [],
            "processed_at": datetime.now().isoformat(),
        }

        with open(processed_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print("未找到原始数据，已生成空统计文件。")
        return

    if raw_format == "flat":
        raise ValueError(
            "当前 data/raw/ 下是扁平图片结构，无法推断分类标签。\n"
            "请改成 data/raw/class_name/*.jpg，或者 data/raw/train/class_name/*.jpg + data/raw/val/class_name/*.jpg。"
        )

    if raw_format == "split":
        train_by_class, val_by_class = collect_split_dataset(raw_dir)
    elif raw_format == "class_dirs":
        train_by_class, val_by_class = split_class_dirs_dataset(
            raw_dir=raw_dir,
            val_split=val_split,
            seed=42,
        )
    else:
        raise RuntimeError(f"未知数据格式: {raw_format}")

    class_names = sorted(set(train_by_class.keys()) | set(val_by_class.keys()))

    if not class_names:
        raise RuntimeError("没有识别到任何类别目录，请检查 data/raw/ 结构。")

    train_success, train_fail, train_counts = export_split(
        split_name="train",
        by_class=train_by_class,
        processed_dir=processed_dir,
        img_size=img_size,
        resize=resize,
    )

    val_success, val_fail, val_counts = export_split(
        split_name="val",
        by_class=val_by_class,
        processed_dir=processed_dir,
        img_size=img_size,
        resize=resize,
    )

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_images = total_train + total_val

    write_class_names(processed_dir, class_names)
    write_edge_class_names(class_names)

    stats = {
        "task": "classification",
        "status": "success",
        "raw_format": raw_format,
        "raw_data": str(raw_dir),
        "processed": str(processed_dir),
        "num_classes": len(class_names),
        "class_names": class_names,
        "total_images": total_images,
        "train": total_train,
        "val": total_val,
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
        "img_size": img_size,
        "resize": resize,
        "val_split": val_split,
        "success": train_success + val_success,
        "failed": train_fail + val_fail,
        "processed_at": datetime.now().isoformat(),
    }

    with open(processed_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("✓ 分类数据预处理完成")
    print("=" * 60)
    print(f"类别数: {len(class_names)}")
    print(f"类别名: {class_names}")
    print(f"训练集: {total_train}")
    print(f"验证集: {total_val}")
    print(f"成功处理: {train_success + val_success}")
    print(f"失败图像: {train_fail + val_fail}")
    print(f"统计文件: {processed_dir / 'dataset_stats.json'}")
    print(f"类别文件: {processed_dir / 'class_names.yaml'}")
    print(f"边缘类别文件: edge/runtime/class_names.yaml")


if __name__ == "__main__":
    main()
