from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pipeline.core.config import load_stage_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def parse_names(names_obj: Any) -> list[str]:
    """
    兼容两种 data.yaml 写法：

    names:
      0: green_cube
      1: yellow_cube

    或：

    names: [green_cube, yellow_cube]
    """
    if isinstance(names_obj, dict):
        return [
            str(names_obj[k])
            for k in sorted(names_obj.keys(), key=lambda x: int(x))
        ]

    if isinstance(names_obj, list):
        return [str(x) for x in names_obj]

    raise ValueError("data.yaml 中 names 字段格式不正确，应为 dict 或 list")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")

    return sorted(
        [p for p in images_dir.rglob("*") if p.is_file() and is_image_file(p)]
    )


def corresponding_label_path(
    image_path: Path,
    images_dir: Path,
    labels_dir: Path,
) -> Path:
    rel = image_path.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def validate_obb_label_file(
    label_path: Path,
    num_classes: int,
    class_counter: Counter,
) -> int:
    """
    校验 YOLO-OBB 标签文件。

    标准格式：
        class_id x1 y1 x2 y2 x3 y3 x4 y4

    一共 9 列：
        1 个类别 id + 8 个归一化坐标

    返回：
        当前标签文件中的 OBB 目标数量
    """
    if not label_path.exists():
        raise FileNotFoundError(f"缺少标签文件: {label_path}")

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        # 允许空标签，表示该图没有目标
        return 0

    box_count = 0

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        if len(parts) != 9:
            raise ValueError(
                f"{label_path} 第 {line_no} 行格式错误。"
                f"YOLO-OBB 应为 9 列：class x1 y1 x2 y2 x3 y3 x4 y4，"
                f"实际为 {len(parts)} 列: {line}"
            )

        try:
            cls_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError as e:
            raise ValueError(
                f"{label_path} 第 {line_no} 行存在无法解析的数字: {line}"
            ) from e

        if not (0 <= cls_id < num_classes):
            raise ValueError(
                f"{label_path} 第 {line_no} 行类别 id 越界: {cls_id}，"
                f"num_classes={num_classes}"
            )

        if len(coords) != 8:
            raise ValueError(
                f"{label_path} 第 {line_no} 行 OBB 坐标数量错误，"
                f"应为 8 个坐标，实际为 {len(coords)}: {line}"
            )

        for coord in coords:
            if not (0.0 <= coord <= 1.0):
                raise ValueError(
                    f"{label_path} 第 {line_no} 行坐标未归一化到 [0, 1]: {line}"
                )

        # 简单检查四个点不能全部重合
        xs = coords[0::2]
        ys = coords[1::2]
        if max(xs) - min(xs) <= 1e-8 or max(ys) - min(ys) <= 1e-8:
            raise ValueError(
                f"{label_path} 第 {line_no} 行 OBB 框退化，四点范围过小: {line}"
            )

        class_counter[cls_id] += 1
        box_count += 1

    return box_count


def collect_split_info(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
    check_images: bool = True,
    check_labels: bool = True,
) -> dict[str, Any]:
    """
    收集并校验一个 split 的信息。

    check_images=True:
        要求 split 中必须存在图像。

    check_labels=True:
        要求每张图都有对应标签文件，且标签必须符合 YOLO-OBB 9 字段格式。
    """
    image_files = collect_images(images_dir)

    if check_images and not image_files:
        raise ValueError(f"{split_name} 集没有找到任何图像: {images_dir}")

    missing_labels: list[str] = []
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
            n_boxes = validate_obb_label_file(
                label_path=label_path,
                num_classes=num_classes,
                class_counter=class_counter,
            )
        else:
            text = label_path.read_text(encoding="utf-8").strip()
            n_boxes = 0 if not text else len(text.splitlines())

        total_boxes += n_boxes

        if n_boxes == 0:
            empty_label_files += 1

    if check_labels and missing_labels:
        preview = "\n".join(missing_labels[:10])
        raise FileNotFoundError(
            f"{split_name} 集有 {len(missing_labels)} 个标签文件缺失，"
            f"前 10 个如下：\n{preview}"
        )

    return {
        "num_images": len(image_files),
        "num_boxes": total_boxes,
        "empty_label_files": empty_label_files,
        "class_counter": dict(class_counter),
        "image_files": image_files,
        "missing_label_files": missing_labels,
    }


def copy_pairs(
    image_files: list[Path],
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
            # 正常严格检查下不会走到这里。
            # 只有 check_labels=False 时，允许补空标签。
            dst_label.write_text("", encoding="utf-8")


def build_processed_data_yaml(
    processed_root: Path,
    class_names: list[str],
    has_test: bool,
) -> dict[str, Any]:
    data = {
        "path": processed_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    if has_test:
        data["test"] = "images/test"

    return data


def split_dirs(raw_root: Path, split: str) -> tuple[Path, Path]:
    """
    统一约定 raw_obb 目录结构为：

    data/raw_obb/
      images/train
      labels/train
      images/val
      labels/val
      images/test
      labels/test
    """
    return raw_root / "images" / split, raw_root / "labels" / split


def main() -> None:
    cfg = load_stage_config("preprocess")

    dataset_cfg = cfg["dataset"]
    preprocess_cfg = cfg.get("preprocess", {})

    raw_root = Path(dataset_cfg["root"])
    raw_yaml_path = Path(dataset_cfg["yaml_path"])
    processed_root = Path(preprocess_cfg.get("processed_root", "data/processed_obb"))
    processed_yaml_path = Path(
        preprocess_cfg.get("processed_yaml", processed_root / "data.yaml")
    )

    check_images = bool(preprocess_cfg.get("check_images", True))
    check_labels = bool(preprocess_cfg.get("check_labels", True))
    copy_to_processed = bool(preprocess_cfg.get("copy_to_processed", True))

    expected_label_fields = int(preprocess_cfg.get("expected_label_fields", 9))
    if expected_label_fields != 9:
        raise ValueError(
            f"OBB 标签应为 9 列，但配置 expected_label_fields={expected_label_fields}"
        )

    if not raw_root.exists():
        raise FileNotFoundError(f"原始 OBB 数据根目录不存在: {raw_root}")

    if not raw_yaml_path.exists():
        raise FileNotFoundError(f"原始 OBB data.yaml 不存在: {raw_yaml_path}")

    raw_data_yaml = load_yaml(raw_yaml_path)
    if "names" not in raw_data_yaml:
        raise KeyError(f"{raw_yaml_path} 中缺少 names 字段")

    class_names = parse_names(raw_data_yaml["names"])
    num_classes = int(dataset_cfg.get("num_classes", len(class_names)))

    if len(class_names) != num_classes:
        raise ValueError(
            f"配置不一致：generated/task.generated.yaml 中 num_classes={num_classes}，"
            f"但 data.yaml 中 names 数量为 {len(class_names)}"
        )

    train_images_dir, train_labels_dir = split_dirs(raw_root, "train")
    val_images_dir, val_labels_dir = split_dirs(raw_root, "val")
    test_images_dir, test_labels_dir = split_dirs(raw_root, "test")

    has_test = test_images_dir.exists() and test_labels_dir.exists()

    print("=" * 60)
    print("OBB 旋转框数据预处理开始")
    print("=" * 60)
    print(f"原始数据根目录: {raw_root}")
    print(f"原始 data.yaml: {raw_yaml_path}")
    print(f"输出目录: {processed_root}")
    print(f"输出 data.yaml: {processed_yaml_path}")
    print(f"类别数: {num_classes}")
    print(f"类别: {class_names}")
    print("标签格式: YOLO-OBB 9列，class x1 y1 x2 y2 x3 y3 x4 y4")
    print("=" * 60)

    train_info = collect_split_info(
        split_name="train",
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        num_classes=num_classes,
        check_images=check_images,
        check_labels=check_labels,
    )

    val_info = collect_split_info(
        split_name="val",
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        num_classes=num_classes,
        check_images=check_images,
        check_labels=check_labels,
    )

    test_info: dict[str, Any] | None = None
    if has_test:
        test_info = collect_split_info(
            split_name="test",
            images_dir=test_images_dir,
            labels_dir=test_labels_dir,
            num_classes=num_classes,
            check_images=False,
            check_labels=check_labels,
        )

    if copy_to_processed:
        ensure_clean_dir(processed_root)

        for split in ["train", "val"]:
            (processed_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (processed_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        copy_pairs(
            image_files=train_info["image_files"],
            src_images_dir=train_images_dir,
            src_labels_dir=train_labels_dir,
            dst_images_dir=processed_root / "images" / "train",
            dst_labels_dir=processed_root / "labels" / "train",
        )

        copy_pairs(
            image_files=val_info["image_files"],
            src_images_dir=val_images_dir,
            src_labels_dir=val_labels_dir,
            dst_images_dir=processed_root / "images" / "val",
            dst_labels_dir=processed_root / "labels" / "val",
        )

        if has_test and test_info is not None:
            (processed_root / "images" / "test").mkdir(parents=True, exist_ok=True)
            (processed_root / "labels" / "test").mkdir(parents=True, exist_ok=True)

            copy_pairs(
                image_files=test_info["image_files"],
                src_images_dir=test_images_dir,
                src_labels_dir=test_labels_dir,
                dst_images_dir=processed_root / "images" / "test",
                dst_labels_dir=processed_root / "labels" / "test",
            )
    else:
        processed_root.mkdir(parents=True, exist_ok=True)

    processed_data_yaml = build_processed_data_yaml(
        processed_root=processed_root,
        class_names=class_names,
        has_test=has_test,
    )
    save_yaml(processed_data_yaml, processed_yaml_path)

    total_class_counter: Counter = Counter()
    total_class_counter.update(train_info["class_counter"])
    total_class_counter.update(val_info["class_counter"])

    if test_info is not None:
        total_class_counter.update(test_info["class_counter"])

    def instances_per_class(info: dict[str, Any]) -> dict[str, int]:
        return {
            class_names[int(k)]: int(v)
            for k, v in info["class_counter"].items()
        }

    total_images = train_info["num_images"] + val_info["num_images"]
    total_boxes = train_info["num_boxes"] + val_info["num_boxes"]

    if test_info is not None:
        total_images += test_info["num_images"]
        total_boxes += test_info["num_boxes"]

    stats: dict[str, Any] = {
        "task": "obb_detection",
        "label_format": "yolo_obb_8points",
        "expected_label_fields": 9,
        "dataset_root": raw_root.as_posix(),
        "processed_root": processed_root.as_posix(),
        "raw_yaml_path": raw_yaml_path.as_posix(),
        "processed_yaml_path": processed_yaml_path.as_posix(),
        "class_names": class_names,
        "num_classes": num_classes,
        "train": {
            "num_images": train_info["num_images"],
            "num_boxes": train_info["num_boxes"],
            "empty_label_files": train_info["empty_label_files"],
            "missing_label_files": len(train_info["missing_label_files"]),
            "instances_per_class": instances_per_class(train_info),
        },
        "val": {
            "num_images": val_info["num_images"],
            "num_boxes": val_info["num_boxes"],
            "empty_label_files": val_info["empty_label_files"],
            "missing_label_files": len(val_info["missing_label_files"]),
            "instances_per_class": instances_per_class(val_info),
        },
        "total": {
            "num_images": total_images,
            "num_boxes": total_boxes,
            "instances_per_class": {
                class_names[int(k)]: int(v)
                for k, v in total_class_counter.items()
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

    if test_info is not None:
        stats["test"] = {
            "num_images": test_info["num_images"],
            "num_boxes": test_info["num_boxes"],
            "empty_label_files": test_info["empty_label_files"],
            "missing_label_files": len(test_info["missing_label_files"]),
            "instances_per_class": instances_per_class(test_info),
        }

    save_json(stats, processed_root / "dataset_stats.json")

    print("✓ OBB 旋转框数据预处理完成")
    print(f"train 图像数: {train_info['num_images']}")
    print(f"val 图像数:   {val_info['num_images']}")

    if test_info is not None:
        print(f"test 图像数:  {test_info['num_images']}")

    print(f"train OBB框数: {train_info['num_boxes']}")
    print(f"val OBB框数:   {val_info['num_boxes']}")

    if test_info is not None:
        print(f"test OBB框数:  {test_info['num_boxes']}")

    print(f"总图像数: {total_images}")
    print(f"总OBB框数: {total_boxes}")
    print(f"输出 YAML: {processed_yaml_path}")
    print(f"统计文件: {processed_root / 'dataset_stats.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
