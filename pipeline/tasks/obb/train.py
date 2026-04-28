from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from pipeline.core.config import load_stage_config, project_path, require_file


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def to_builtin(value: Any) -> Any:
    """
    将 numpy / torch / pathlib 等对象转成 JSON 可序列化对象。
    """
    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    return value


def find_weight_file(results: Any, filename: str) -> Path | None:
    """
    Ultralytics 训练完成后通常会输出：
        <save_dir>/weights/best.pt
        <save_dir>/weights/last.pt
    """
    save_dir = getattr(results, "save_dir", None)

    if save_dir is None:
        return None

    candidate = Path(save_dir) / "weights" / filename
    return candidate if candidate.exists() else None


def copy_if_exists(src: Path | None, dst: Path) -> str | None:
    if src is None or not src.exists():
        return None

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.as_posix()


def extract_results_dict(results: Any) -> dict[str, Any]:
    """
    Ultralytics 不同版本的返回对象字段略有差异，这里做兼容。
    """
    data: dict[str, Any] = {}

    if hasattr(results, "results_dict"):
        try:
            data.update(to_builtin(results.results_dict))
        except Exception:
            pass

    if hasattr(results, "save_dir"):
        data["save_dir"] = Path(results.save_dir).as_posix()

    return data


def main() -> None:
    cfg = load_stage_config("train")

    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    output_cfg = cfg.get("output", {})
    mlflow_cfg = cfg.get("mlflow", {})

    weights = project_path(model_cfg.get("weights", "models/pretrained/yolov8n-obb.pt"))
    data_yaml = project_path(dataset_cfg.get("yaml_path", "data/processed_obb/data.yaml"))

    checkpoint_dir = project_path(output_cfg.get("checkpoint_dir", "models/checkpoints_obb"))
    metrics_dir = project_path(output_cfg.get("metrics_dir", "models/metrics_obb"))

    img_size = int(train_cfg.get("img_size", 640))
    epochs = int(train_cfg.get("epochs", 50))
    batch_size = int(train_cfg.get("batch_size", 16))
    device = train_cfg.get("device", "cpu")
    workers = int(train_cfg.get("workers", 4))
    patience = int(train_cfg.get("patience", 20))
    cache = bool(train_cfg.get("cache", False))
    project = train_cfg.get("project", "models/runs/obb")
    name = train_cfg.get("name", "visionops_obb")
    exist_ok = bool(train_cfg.get("exist_ok", True))
    lr0 = float(train_cfg.get("lr0", 0.001))

    require_file(weights, "请确认 task.yaml 中 model.pretrained_weights 指向本地 yolov8n-obb.pt")
    require_file(data_yaml, "请先运行 dvc repro preprocess 生成 data/processed_obb/data.yaml")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OBB 模型训练开始")
    print("=" * 60)
    print(f"权重文件: {weights}")
    print(f"数据配置: {data_yaml}")
    print(f"输入尺寸: {img_size}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"device: {device}")
    print(f"workers: {workers}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"metrics_dir: {metrics_dir}")
    print("=" * 60)

    model = YOLO(str(weights))

    results = model.train(
        task="obb",
        data=str(data_yaml),
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        device=device,
        workers=workers,
        patience=patience,
        cache=cache,
        project=str(project),
        name=str(name),
        exist_ok=exist_ok,
        lr0=lr0,
    )

    best_src = find_weight_file(results, "best.pt")
    last_src = find_weight_file(results, "last.pt")

    best_dst = checkpoint_dir / "best.pt"
    last_dst = checkpoint_dir / "last.pt"

    best_path = copy_if_exists(best_src, best_dst)
    last_path = copy_if_exists(last_src, last_dst)

    if best_path is None:
        raise FileNotFoundError(
            "训练完成但未找到 best.pt。请检查 Ultralytics 输出目录是否正常。"
        )

    result_dict = extract_results_dict(results)

    train_metrics = {
        "task": "obb_detection",
        "stage": "train",
        "status": "success",
        "weights": weights.as_posix(),
        "data_yaml": data_yaml.as_posix(),
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "best_pt": best_path,
        "last_pt": last_path,
        "ultralytics_save_dir": result_dict.get("save_dir"),
        "params": {
            "img_size": img_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "workers": workers,
            "patience": patience,
            "cache": cache,
            "project": str(project),
            "name": str(name),
            "exist_ok": exist_ok,
            "lr0": lr0,
        },
        "mlflow": {
            "experiment_name": mlflow_cfg.get("experiment_name"),
            "tracking_uri": mlflow_cfg.get("tracking_uri"),
            "log_artifacts": mlflow_cfg.get("log_artifacts", True),
            "log_model": mlflow_cfg.get("log_model", False),
        },
        "raw_results": result_dict,
        "trained_at": datetime.now().isoformat(),
    }

    save_json(train_metrics, metrics_dir / "train_metrics.json")

    print("✓ OBB 模型训练完成")
    print(f"best.pt: {best_path}")
    print(f"last.pt:  {last_path}")
    print(f"训练指标: {metrics_dir / 'train_metrics.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
