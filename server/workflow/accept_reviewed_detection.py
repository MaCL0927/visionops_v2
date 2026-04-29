#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ["-", "_", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("._") or "unknown"


def load_current_batch_dir(model_context_path: Path) -> Path:
    manifest = read_json(model_context_path, default={}) or {}

    source_manifest = manifest.get("source_manifest")
    if source_manifest:
        p = Path(source_manifest)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.parent

    package_name = manifest.get("package_name") or manifest.get("batch_id")
    if package_name:
        return PROJECT_ROOT / "data" / "raw_collected" / package_name

    raise RuntimeError(
        f"无法从 {model_context_path} 推断当前 batch，请先运行 make ingest-collected"
    )


def load_class_names() -> list[str]:
    """
    第一版尽量兼容：
    1. data/model_context/manifest.json 里的 class_names/classes/names
    2. edge/runtime/class_names.yaml
    3. 默认 names: ['object']
    """
    manifest_path = PROJECT_ROOT / "data" / "model_context" / "manifest.json"
    manifest = read_json(manifest_path, default={}) or {}

    for key in ["class_names", "classes", "names"]:
        value = manifest.get(key)
        if isinstance(value, list) and value:
            return [str(x) for x in value]
        if isinstance(value, dict) and value:
            return [str(value[k]) for k in sorted(value.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]

    yaml_path = PROJECT_ROOT / "edge" / "runtime" / "class_names.yaml"
    if yaml_path.exists():
        try:
            import yaml
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            for key in ["class_names", "classes", "names"]:
                value = data.get(key)
                if isinstance(value, list) and value:
                    return [str(x) for x in value]
                if isinstance(value, dict) and value:
                    return [str(value[k]) for k in sorted(value.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
        except Exception:
            pass

    return ["object"]


def find_image_label_pairs(
    all_images_dir: Path,
    labels_dir: Path,
    allow_empty_labels: bool,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

    images = sorted([
        p for p in all_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])

    labels_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        label = labels_dir / f"{img.stem}.txt"

        if label.exists():
            pairs.append((img, label))
        elif allow_empty_labels:
            label.write_text("", encoding="utf-8")
            pairs.append((img, label))

    return pairs


def infer_label_format(label_paths: list[Path]) -> str:
    """
    简单识别标注格式：
    YOLO HBB: class x y w h -> 5列
    YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4 -> 9列
    """
    max_cols = 0

    for path in label_paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        for line in text.splitlines():
            parts = line.strip().split()
            max_cols = max(max_cols, len(parts))

    if max_cols >= 9:
        return "obb"
    return "detection"

def reset_output_root(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

def ensure_raw_detection_dirs(output_root: Path) -> None:
    for sub in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)


def move_or_copy(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def write_data_yaml(output_root: Path, class_names: list[str]) -> None:
    names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(class_names)])
    content = f"""# Auto-generated by VisionOps accept_reviewed_detection.py
path: {output_root.as_posix()}
train: images/train
val: images/val

nc: {len(class_names)}
names:
{names_lines}
"""
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Accept reviewed labels into raw_detection dataset.")
    parser.add_argument(
        "--model-context",
        type=str,
        default="data/model_context/manifest.json",
        help="当前数据闭环上下文 manifest。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/raw_detection",
        help="检测任务输出数据集目录。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="数据划分随机种子。",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="剪切图片和标签。默认是复制。",
    )
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        default=True,
        help="允许无目标图片，自动创建空 txt。",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()

    model_context_path = PROJECT_ROOT / args.model_context
    batch_dir = load_current_batch_dir(model_context_path)
    batch_id = batch_dir.name
    all_images_dir = batch_dir / "all_images"
    labels_dir = batch_dir / "labels"

    if not all_images_dir.exists():
        raise FileNotFoundError(f"未找到 all_images 目录: {all_images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(
            f"未找到 labels 目录: {labels_dir}。请先在 X-AnyLabeling 中导出 YOLO 格式标注到该目录。"
        )

    pairs = find_image_label_pairs(
        all_images_dir=all_images_dir,
        labels_dir=labels_dir,
        allow_empty_labels=args.allow_empty_labels,
    )

    if not pairs:
        raise RuntimeError(
            f"未找到可接收的图片/标签。请确认 X-AnyLabeling 已在 {all_images_dir} 保存同名 txt。"
        )

    label_format = infer_label_format([label for _, label in pairs])

    if label_format == "obb":
        output_root = PROJECT_ROOT / "data" / "raw_obb"
        print("[INFO] 检测到 OBB 标注格式，输出到 data/raw_obb")
    else:
        output_root = PROJECT_ROOT / args.output_root
        print("[INFO] 检测到普通 detection 标注格式，输出到 data/raw_detection")

    print(f"[INFO] 清空输出目录: {output_root.relative_to(PROJECT_ROOT)}")
    reset_output_root(output_root)
    ensure_raw_detection_dirs(output_root)

    random.seed(args.seed)
    random.shuffle(pairs)

    val_count = max(1, int(len(pairs) * args.val_ratio)) if len(pairs) > 1 else 0
    val_pairs = set([p[0] for p in pairs[:val_count]])

    class_names = load_class_names()

    moved_train = 0
    moved_val = 0

    for img, label in pairs:
        split = "val" if img in val_pairs else "train"

        # 加 batch 前缀，避免不同批次同名图片覆盖。
        new_stem = safe_name(f"{batch_id}_{img.stem}")
        img_dst = output_root / "images" / split / f"{new_stem}{img.suffix.lower()}"
        label_dst = output_root / "labels" / split / f"{new_stem}.txt"

        move_or_copy(img, img_dst, move=args.move)
        move_or_copy(label, label_dst, move=args.move)

        if split == "val":
            moved_val += 1
        else:
            moved_train += 1

    write_data_yaml(output_root, class_names)

    review_record = {
        "batch_id": batch_id,
        "source_all_images_dir": str(all_images_dir.relative_to(PROJECT_ROOT)),
        "source_labels_dir": str(labels_dir.relative_to(PROJECT_ROOT)),
        "output_root": str(output_root.relative_to(PROJECT_ROOT)),
        "label_format": label_format,
        "mode": "move" if args.move else "copy",
        "train_count": moved_train,
        "val_count": moved_val,
        "class_names": class_names,
        "accepted_at": now_str(),
    }

    write_json(output_root / "review_acceptance.json", review_record)

    print("============================================================")
    print("[OK] 审核数据已接入训练数据目录")
    print(f"batch_id:      {batch_id}")
    print(f"label_format:  {label_format}")
    print(f"output_root:   {output_root.relative_to(PROJECT_ROOT)}")
    print(f"train_count:   {moved_train}")
    print(f"val_count:     {moved_val}")
    print(f"data_yaml:     {(output_root / 'data.yaml').relative_to(PROJECT_ROOT)}")
    print("============================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
