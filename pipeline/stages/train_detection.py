from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import mlflow


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def pick_metric(metrics_dict: Dict[str, Any], keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in metrics_dict:
            return safe_float(metrics_dict[key], default)
    return default


def resolve_device(device_cfg: str) -> str | int:
    """
    Ultralytics 支持 device='cpu'、device=0、device='0' 等形式。
    这里尽量保持简单：
    - 'cuda' -> 0
    - 'cpu'  -> 'cpu'
    - 其他值原样返回
    """
    device_cfg = str(device_cfg).strip().lower()
    if device_cfg == "cuda":
        return 0
    if device_cfg == "cpu":
        return "cpu"
    return device_cfg


def main() -> None:
    from ultralytics import YOLO

    cfg_path = Path("pipeline/configs/detection_train.yaml")
    cfg = load_yaml(cfg_path)

    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    mlflow_cfg = cfg.get("mlflow", {})
    output_cfg = cfg.get("output", {})

    dataset_yaml = Path(dataset_cfg["yaml_path"])
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"检测数据集 YAML 不存在: {dataset_yaml}")

    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "models/checkpoints_detection"))
    metrics_dir = Path(output_cfg.get("metrics_dir", "models/metrics_detection"))
    ensure_dir(checkpoint_dir)
    ensure_dir(metrics_dir)

    architecture = model_cfg.get("architecture", "yolov8n")
    weights = model_cfg.get("weights", f"{architecture}.pt")
    num_classes = int(model_cfg.get("num_classes", 2))
    pretrained = bool(model_cfg.get("pretrained", True))

    epochs = int(train_cfg.get("epochs", 50))
    batch_size = int(train_cfg.get("batch_size", 16))
    img_size = int(train_cfg.get("img_size", 640))
    lr0 = float(train_cfg.get("lr0", 0.001))
    workers = int(train_cfg.get("workers", 4))
    patience = int(train_cfg.get("patience", 20))
    cache = bool(train_cfg.get("cache", False))
    # project = str(train_cfg.get("project", "runs/detect"))
    project = str(Path(train_cfg.get("project", "models/runs/detect")).resolve())
    name = str(train_cfg.get("name", "visionops_detection"))
    exist_ok = bool(train_cfg.get("exist_ok", True))
    device = resolve_device(str(train_cfg.get("device", "cpu")))

    print("=" * 60)
    print("YOLOv8 检测训练开始")
    print("=" * 60)
    print(f"模型架构: {architecture}")
    print(f"初始权重: {weights}")
    print(f"数据集 YAML: {dataset_yaml}")
    print(f"类别数: {num_classes}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"img_size: {img_size}")
    print(f"device: {device}")
    print("=" * 60)

    # 设置 MLflow
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "visionops-detection"))

    run_name = f"{architecture}-detect-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # 记录基础参数
        mlflow.log_params(
            {
                "task": "detection",
                "architecture": architecture,
                "weights": weights,
                "num_classes": num_classes,
                "pretrained": pretrained,
                "dataset_yaml": str(dataset_yaml),
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
                "lr0": lr0,
                "workers": workers,
                "patience": patience,
                "cache": cache,
                "device": str(device),
                "project": project,
                "name": name,
            }
        )

        (metrics_dir / "mlflow_run_id.txt").write_text(run.info.run_id, encoding="utf-8")

        # 构建 YOLO 模型
        model = YOLO(weights)

        # 正式训练
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            lr0=lr0,
            device=device,
            workers=workers,
            patience=patience,
            cache=cache,
            project=project,
            name=name,
            exist_ok=exist_ok,
            pretrained=pretrained,
            verbose=True,
        )

        # Ultralytics 训练后常见输出目录
        # results.save_dir 通常是本次 run 的目录
        save_dir = Path(getattr(results, "save_dir", Path(project) / name))
        weights_dir = save_dir / "weights"

        best_src = weights_dir / "best.pt"
        last_src = weights_dir / "last.pt"

        best_dst = checkpoint_dir / "best.pt"
        last_dst = checkpoint_dir / "last.pt"

        copied_best = copy_if_exists(best_src, best_dst)
        copied_last = copy_if_exists(last_src, last_dst)

        # 尝试从训练结果中提取关键指标
        results_dict = getattr(results, "results_dict", {}) or {}

        best_map50 = pick_metric(
            results_dict,
            keys=["metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50"],
            default=0.0,
        )
        best_map50_95 = pick_metric(
            results_dict,
            keys=["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "map"],
            default=0.0,
        )
        best_precision = pick_metric(
            results_dict,
            keys=["metrics/precision(B)", "metrics/precision", "precision"],
            default=0.0,
        )
        best_recall = pick_metric(
            results_dict,
            keys=["metrics/recall(B)", "metrics/recall", "recall"],
            default=0.0,
        )
        fitness = pick_metric(
            results_dict,
            keys=["fitness"],
            default=0.0,
        )

        # 写 MLflow 指标
        mlflow.log_metrics(
            {
                "best_map50": best_map50,
                "best_map50_95": best_map50_95,
                "best_precision": best_precision,
                "best_recall": best_recall,
                "fitness": fitness,
            }
        )

        # 记录关键产物
        artifact_logging_error = None

        if mlflow_cfg.get("log_artifacts", True):
            try:
                if copied_best:
                    mlflow.log_artifact(str(best_dst))
                if copied_last:
                    mlflow.log_artifact(str(last_dst))
                    
                # 记录训练目录下可能存在的 results.csv / args.yaml 等
                for maybe_artifact in ["results.csv", "args.yaml"]:
                    fp = save_dir / maybe_artifact
                    if fp.exists():
                        mlflow.log_artifact(str(fp))
            except Exception as e:
                artifact_logging_error = str(e)
                print(f"警告: MLflow artifact 上传失败，将继续完成本地训练流程: {e}")

        metrics_payload = {
            "task": "detection",
            "architecture": architecture,
            "weights": weights,
            "num_classes": num_classes,
            "dataset_yaml": str(dataset_yaml),
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "device": str(device),
            "save_dir": str(save_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "copied_best": copied_best,
            "copied_last": copied_last,
            "best_checkpoint": str(best_dst) if copied_best else None,
            "last_checkpoint": str(last_dst) if copied_last else None,
            "best_metrics": {
                "map50": round(best_map50, 6),
                "map50_95": round(best_map50_95, 6),
                "precision": round(best_precision, 6),
                "recall": round(best_recall, 6),
                "fitness": round(fitness, 6),
            },
            "results_dict": results_dict,
            "mlflow_run_id": run.info.run_id,
            "trained_at": datetime.now().isoformat(),
            "mlflow_artifact_logging_enabled": bool(mlflow_cfg.get("log_artifacts", True)),
            "mlflow_artifact_logging_error": artifact_logging_error,
        }

        save_json(metrics_payload, metrics_dir / "train_metrics.json")

        print("✓ YOLOv8 检测训练完成")
        print(f"训练输出目录: {save_dir}")
        print(f"best.pt 已复制: {copied_best} -> {best_dst}")
        print(f"last.pt 已复制: {copied_last} -> {last_dst}")
        print(f"best mAP50: {best_map50:.4f}")
        print(f"best mAP50-95: {best_map50_95:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("=" * 60)


if __name__ == "__main__":
    main()
