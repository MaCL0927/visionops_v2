from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def resolve_device(device_cfg: str) -> str | int:
    device_cfg = str(device_cfg).strip().lower()
    if device_cfg == "cuda":
        return 0
    if device_cfg == "cpu":
        return "cpu"
    return device_cfg


def extract_class_names(data_yaml: dict) -> List[str]:
    names = data_yaml.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, list):
        return names
    return []


def main() -> None:
    from ultralytics import YOLO

    train_cfg_path = Path("pipeline/configs/detection_train.generated.yaml")
    if not train_cfg_path.exists():
        train_cfg_path = Path("pipeline/configs/detection_train.yaml")
    train_cfg = load_yaml(train_cfg_path)
    print(f"评估配置文件: {train_cfg_path}")

    model_path = Path("models/checkpoints_detection/best.pt")
    dataset_yaml = Path(train_cfg["dataset"]["yaml_path"])
    metrics_dir = Path(train_cfg.get("output", {}).get("metrics_dir", "models/metrics_detection"))

    if not model_path.exists():
        raise FileNotFoundError(f"检测模型不存在: {model_path}")
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"检测数据集 YAML 不存在: {dataset_yaml}")

    data_yaml_obj = load_yaml(dataset_yaml)
    class_names = extract_class_names(data_yaml_obj)

    device = resolve_device(str(train_cfg.get("train", {}).get("device", "cpu")))
    img_size = int(train_cfg.get("train", {}).get("img_size", 640))

    print("=" * 60)
    print("YOLOv8 检测评估开始")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"数据集 YAML: {dataset_yaml}")
    print(f"类别: {class_names}")
    print(f"device: {device}")
    print(f"img_size: {img_size}")
    print("=" * 60)

    model = YOLO(str(model_path))

    results = model.val(
        data=str(dataset_yaml),
        imgsz=img_size,
        device=device,
        split="val",
        verbose=True,
        save_json=False,
        plots=True,
    )

    # 整体指标
    map50 = safe_float(getattr(results.box, "map50", 0.0))
    map50_95 = safe_float(getattr(results.box, "map", 0.0))
    map75 = safe_float(getattr(results.box, "map75", 0.0))

    # Ultralytics 常见结构：results.results_dict
    results_dict = getattr(results, "results_dict", {}) or {}

    precision = 0.0
    recall = 0.0

    # 尝试优先从 results_dict 中取
    for key in ["metrics/precision(B)", "metrics/precision", "precision"]:
        if key in results_dict:
            precision = safe_float(results_dict[key])
            break

    for key in ["metrics/recall(B)", "metrics/recall", "recall"]:
        if key in results_dict:
            recall = safe_float(results_dict[key])
            break

    # 每类指标
    per_class = []
    maps = getattr(results.box, "maps", None)
    if maps is not None and len(class_names) > 0:
        for i, class_name in enumerate(class_names):
            cls_map50_95 = safe_float(maps[i]) if i < len(maps) else 0.0
            per_class.append(
                {
                    "class_id": i,
                    "class_name": class_name,
                    "map50_95": round(cls_map50_95, 6),
                }
            )

    eval_metrics = {
        "task": "detection",
        "model_path": str(model_path),
        "dataset_yaml": str(dataset_yaml),
        "num_classes": len(class_names),
        "class_names": class_names,
        "map50": round(map50, 6),
        "map50_95": round(map50_95, 6),
        "map75": round(map75, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "per_class": per_class,
        "evaluated_at": datetime.now().isoformat(),
    }

    eval_report = {
        "summary": {
            "task": "detection",
            "model": str(model_path),
            "dataset": str(dataset_yaml),
            "overall_performance": {
                "map50": round(map50, 6),
                "map50_95": round(map50_95, 6),
                "map75": round(map75, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
            },
        },
        "classes": per_class,
        "raw_results_dict": results_dict,
        "generated_at": datetime.now().isoformat(),
    }

    save_json(eval_metrics, metrics_dir / "eval_metrics.json")
    save_json(eval_report, metrics_dir / "eval_report.json")

    print("✓ YOLOv8 检测评估完成")
    print(f"mAP50: {map50:.4f}")
    print(f"mAP50-95: {map50_95:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"评估指标文件: {metrics_dir / 'eval_metrics.json'}")
    print(f"评估报告文件: {metrics_dir / 'eval_report.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
