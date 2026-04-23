from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def collect_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and is_image_file(p)])


def corresponding_label_path(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def validate_label_file(
    label_path: Path,
    num_classes: int,
    class_counter: Counter,
) -> int:
    """
    校验单个 YOLO 标签文件。
    返回该文件中的目标框数量。
    """
    if not label_path.exists():
        raise FileNotFoundError(f"缺少标签文件: {label_path}")

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        # 允许空标签，表示该图中无目标
        return 0

    box_count = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path} 第 {line_no} 行格式错误，应为 5 列，实际为 {len(parts)} 列: {line}"
            )

        try:
            cls_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError as e:
            raise ValueError(
                f"{label_path} 第 {line_no} 行存在无法解析的数字: {line}"
            ) from e

        if not (0 <= cls_id < num_classes):
            raise ValueError(f"{label_path} 第 {line_no} 行类别 id 越界: {cls_id}")

        for c in coords:
            if not (0.0 <= c <= 1.0):
                raise ValueError(
                    f"{label_path} 第 {line_no} 行坐标未归一化到 [0,1]: {line}"
                )

        class_counter[cls_id] += 1
        box_count += 1

    return box_count


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_pairs(
    image_files: List[Path],
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
) -> None:
    for img_path in image_files:
        rel = img_path.relative_to(src_images_dir)
        src_label = corresponding_label_path(img_path, src_images_dir, src_labels_dir)

        dst_img = dst_images_dir / rel
        dst_label = (dst_labels_dir / rel).with_suffix(".txt")

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, dst_img)
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            # 只有在关闭标签严格检查时，才可能走到这里
            dst_label.write_text("", encoding="utf-8")


def parse_names(names_obj) -> List[str]:
    """
    兼容两种写法：
    names:
      0: person
      1: smoke

    或

    names: [person, smoke]
    """
    if isinstance(names_obj, dict):
        return [names_obj[k] for k in sorted(names_obj.keys(), key=lambda x: int(x))]
    if isinstance(names_obj, list):
        return names_obj
    raise ValueError("data.yaml 中 names 字段格式不正确，应为 dict 或 list")


def build_processed_data_yaml(processed_root: Path, class_names: List[str]) -> dict:
    return {
        "path": processed_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }


def collect_split_info(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
    check_images: bool = True,
    check_labels: bool = True,
) -> Dict:
    """
    按开关收集一个 split 的信息。

    - check_images=True: 要求 split 中必须存在图像
    - check_labels=True: 严格检查每张图对应标签是否存在，且每个标签文件格式合法
    - check_labels=False: 不做严格标签校验；若标签文件存在，则仅粗略统计行数
    """
    image_files = collect_images(images_dir)

    if check_images and not image_files:
        raise ValueError(f"{split_name} 集没有找到任何图像: {images_dir}")

    missing_labels: List[str] = []
    class_counter: Counter = Counter()
    total_boxes = 0
    empty_label_files = 0

    for img_path in image_files:
        label_path = corresponding_label_path(img_path, images_dir, labels_dir)

        if not label_path.exists():
            if check_labels:
                missing_labels.append(str(label_path))
            continue

        if check_labels:
            n_boxes = validate_label_file(label_path, num_classes, class_counter)
        else:
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                n_boxes = 0
            else:
                # 仅做粗略统计，不校验每行内容格式
                n_boxes = len(text.splitlines())

        total_boxes += n_boxes
        if n_boxes == 0:
            empty_label_files += 1

    if check_labels and missing_labels:
        preview = "\n".join(missing_labels[:10])
        raise FileNotFoundError(
            f"{split_name} 集有 {len(missing_labels)} 个标签文件缺失，前 10 个如下：\n{preview}"
        )

    return {
        "num_images": len(image_files),
        "num_boxes": total_boxes,
        "empty_label_files": empty_label_files,
        "class_counter": dict(class_counter),
        "image_files": image_files,
        "missing_label_files": missing_labels,
    }


def main() -> None:
    cfg_path = Path("pipeline/configs/detection_data.generated.yaml")
    if not cfg_path.exists():
        cfg_path = Path("pipeline/configs/detection_data.yaml")
    cfg = load_yaml(cfg_path)

    dataset_cfg = cfg["dataset"]
    preprocess_cfg = cfg.get("preprocess", {})

    raw_root = Path(dataset_cfg["root"])
    raw_yaml_path = Path(dataset_cfg["yaml_path"])
    processed_root = Path(preprocess_cfg.get("processed_root", "data/processed_detection"))

    check_images = preprocess_cfg.get("check_images", True)
    check_labels = preprocess_cfg.get("check_labels", True)
    copy_to_processed = preprocess_cfg.get("copy_to_processed", True)

    if not raw_root.exists():
        raise FileNotFoundError(f"原始检测数据根目录不存在: {raw_root}")
    if not raw_yaml_path.exists():
        raise FileNotFoundError(f"原始 data.yaml 不存在: {raw_yaml_path}")

    raw_data_yaml = load_yaml(raw_yaml_path)
    class_names = parse_names(raw_data_yaml["names"])
    num_classes = dataset_cfg.get("num_classes", len(class_names))

    if len(class_names) != num_classes:
        raise ValueError(
            f"配置不一致：detection_data.yaml 中 num_classes={num_classes}，"
            f"但 data.yaml 中 names 数量为 {len(class_names)}"
        )

    train_images_dir = raw_root / "images" / "train"
    val_images_dir = raw_root / "images" / "val"
    train_labels_dir = raw_root / "labels" / "train"
    val_labels_dir = raw_root / "labels" / "val"

    print("=" * 60)
    print("检测数据预处理开始")
    print("=" * 60)
    print(f"原始数据根目录: {raw_root}")
    print(f"原始 data.yaml: {raw_yaml_path}")
    print(f"输出目录: {processed_root}")
    print(f"类别: {class_names}")
    print("=" * 60)

    if not check_images and not check_labels:
        print("警告: check_images 和 check_labels 都为 False，将跳过严格校验。")
    elif not check_labels:
        print("警告: 已关闭标签严格校验，将只做粗略统计。")

    train_info = collect_split_info(
        "train",
        train_images_dir,
        train_labels_dir,
        num_classes,
        check_images=check_images,
        check_labels=check_labels,
    )
    val_info = collect_split_info(
        "val",
        val_images_dir,
        val_labels_dir,
        num_classes,
        check_images=check_images,
        check_labels=check_labels,
    )

    # 无论是否 copy_to_processed，都保证 processed_root/data.yaml 与 dataset_stats.json 存在，
    # 与 dvc.yaml 的输出约定保持一致。
    if copy_to_processed:
        ensure_clean_dir(processed_root)
        (processed_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (processed_root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (processed_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (processed_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

        copy_pairs(
            train_info["image_files"],
            train_images_dir,
            train_labels_dir,
            processed_root / "images" / "train",
            processed_root / "labels" / "train",
        )
        copy_pairs(
            val_info["image_files"],
            val_images_dir,
            val_labels_dir,
            processed_root / "images" / "val",
            processed_root / "labels" / "val",
        )
    else:
        processed_root.mkdir(parents=True, exist_ok=True)

    processed_data_yaml = build_processed_data_yaml(processed_root, class_names)
    save_yaml(processed_data_yaml, processed_root / "data.yaml")

    total_class_counter = Counter()
    total_class_counter.update(train_info["class_counter"])
    total_class_counter.update(val_info["class_counter"])

    stats = {
        "task": "detection",
        "dataset_root": raw_root.as_posix(),
        "processed_root": processed_root.as_posix(),
        "raw_yaml_path": raw_yaml_path.as_posix(),
        "processed_yaml_path": (processed_root / "data.yaml").as_posix(),
        "class_names": class_names,
        "num_classes": num_classes,
        "train": {
            "num_images": train_info["num_images"],
            "num_boxes": train_info["num_boxes"],
            "empty_label_files": train_info["empty_label_files"],
            "missing_label_files": len(train_info["missing_label_files"]),
            "instances_per_class": {
                class_names[int(k)]: v for k, v in train_info["class_counter"].items()
            },
        },
        "val": {
            "num_images": val_info["num_images"],
            "num_boxes": val_info["num_boxes"],
            "empty_label_files": val_info["empty_label_files"],
            "missing_label_files": len(val_info["missing_label_files"]),
            "instances_per_class": {
                class_names[int(k)]: v for k, v in val_info["class_counter"].items()
            },
        },
        "total": {
            "num_images": train_info["num_images"] + val_info["num_images"],
            "num_boxes": train_info["num_boxes"] + val_info["num_boxes"],
            "instances_per_class": {
                class_names[int(k)]: v for k, v in total_class_counter.items()
            },
        },
        "checked": {
            "images": check_images,
            "labels": check_labels,
            "copy_to_processed": copy_to_processed,
            "strict_mode": bool(check_images or check_labels),
        },
        "processed_at": datetime.now().isoformat(),
    }

    save_json(stats, processed_root / "dataset_stats.json")

    print("✓ 检测数据预处理完成")
    print(f"train 图像数: {train_info['num_images']}")
    print(f"val 图像数: {val_info['num_images']}")
    print(f"train 框数: {train_info['num_boxes']}")
    print(f"val 框数: {val_info['num_boxes']}")
    print(f"输出 YAML: {processed_root / 'data.yaml'}")
    print(f"统计文件: {processed_root / 'dataset_stats.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
