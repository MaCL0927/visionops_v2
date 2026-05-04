from __future__ import annotations

import json
import re
import shutil
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from pipeline.core.config import load_stage_config, project_path, require_file

try:
    import mlflow
except Exception:  # pragma: no cover - 允许无 MLflow 环境先跑通训练
    mlflow = None  # type: ignore


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


def sanitize_mlflow_key(key: str) -> str:
    """
    MLflow metric/param key 对字符有限制，这里把括号、中文等替换成下划线。
    """
    key = str(key).strip().replace("/", "_")
    key = re.sub(r"[^A-Za-z0-9_. -]+", "_", key)
    key = re.sub(r"\s+", "_", key)
    key = re.sub(r"_+", "_", key).strip("._- ")
    return key or "metric"


def safe_log_param(key: str, value: Any) -> None:
    if mlflow is None or value is None:
        return

    try:
        if isinstance(value, (dict, list, tuple)):
            mlflow.log_param(sanitize_mlflow_key(key), json.dumps(to_builtin(value), ensure_ascii=False))
        else:
            mlflow.log_param(sanitize_mlflow_key(key), str(value))
    except Exception as e:
        print(f"[WARN] MLflow log_param 失败: {key}={value}, err={e}")


def safe_log_metric(key: str, value: Any) -> None:
    if mlflow is None or value is None:
        return

    try:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            mlflow.log_metric(sanitize_mlflow_key(key), float(value))
            return
        if hasattr(value, "item"):
            v = value.item()
            if isinstance(v, (int, float)):
                mlflow.log_metric(sanitize_mlflow_key(key), float(v))
    except Exception as e:
        print(f"[WARN] MLflow log_metric 失败: {key}={value}, err={e}")


def safe_log_artifact(path: Path, artifact_path: str | None = None) -> None:
    if mlflow is None or not path.exists():
        return

    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as e:
        print(f"[WARN] MLflow log_artifact 失败: {path}, err={e}")


def safe_log_artifacts_from_dir(save_dir: Path, artifact_path: str = "train") -> None:
    if mlflow is None or not save_dir.exists() or not save_dir.is_dir():
        return

    important_files = [
        "args.yaml",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "train_batch0.jpg",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
    ]
    for name in important_files:
        p = save_dir / name
        if p.exists():
            safe_log_artifact(p, artifact_path=artifact_path)


def setup_mlflow(mlflow_cfg: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    """
    训练阶段负责创建/切换 MLflow experiment。
    - enabled=false 时跳过
    - MLflow 未安装或服务不可用时不阻塞训练，但会打印 WARN
    """
    enabled = bool(mlflow_cfg.get("enabled", True))
    tracking_uri = str(mlflow_cfg.get("tracking_uri") or "http://localhost:5000")
    experiment_name = str(mlflow_cfg.get("experiment_name") or "visionops-segmentation")

    if not enabled:
        print("[INFO] MLflow logging disabled by config")
        return False, tracking_uri, experiment_name

    if mlflow is None:
        print("[WARN] 未安装 mlflow，跳过 MLflow 实验创建与日志写入")
        return False, tracking_uri, experiment_name

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        print(f"[INFO] MLflow Tracking URI: {tracking_uri}")
        print(f"[INFO] MLflow Experiment: {experiment_name}")
        return True, tracking_uri, experiment_name
    except Exception as e:
        print(f"[WARN] MLflow 初始化失败，训练继续但不会写入 MLflow: {e}")
        return False, tracking_uri, experiment_name


def main() -> None:
    cfg = load_stage_config("train")

    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    output_cfg = cfg.get("output", {})
    mlflow_cfg = cfg.get("mlflow", {})

    weights = project_path(model_cfg.get("weights", "models/pretrained/yolov8n-seg.pt"))
    data_yaml = project_path(dataset_cfg.get("yaml_path", "data/processed_segmentation/data.yaml"))

    checkpoint_dir = project_path(output_cfg.get("checkpoint_dir", "models/checkpoints_segmentation"))
    metrics_dir = project_path(output_cfg.get("metrics_dir", "models/metrics_segmentation"))

    img_size = int(train_cfg.get("img_size", 640))
    epochs = int(train_cfg.get("epochs", 50))
    batch_size = int(train_cfg.get("batch_size", 16))
    device = train_cfg.get("device", "cpu")
    workers = int(train_cfg.get("workers", 4))
    patience = int(train_cfg.get("patience", 20))
    cache = bool(train_cfg.get("cache", False))
    project = train_cfg.get("project", "models/runs/segment")
    name = train_cfg.get("name", "visionops_segmentation")
    exist_ok = bool(train_cfg.get("exist_ok", True))
    lr0 = float(train_cfg.get("lr0", 0.001))

    require_file(weights, "请确认 task.yaml 中 model.pretrained_weights 指向本地 yolov8n-seg.pt")
    require_file(data_yaml, "请先运行 dvc repro preprocess 生成 data/processed_segmentation/data.yaml")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Segmentation 模型训练开始")
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

    mlflow_enabled, tracking_uri, experiment_name = setup_mlflow(mlflow_cfg)
    run_name = str(mlflow_cfg.get("run_name") or f"segmentation-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    run_ctx = mlflow.start_run(run_name=run_name) if (mlflow_enabled and mlflow is not None) else nullcontext()

    run_id: str | None = None
    model = YOLO(str(weights))

    with run_ctx as active_run:
        if active_run is not None:
            run_id = active_run.info.run_id
            print(f"[INFO] MLflow Run ID: {run_id}")

            params = {
                "task": "segmentation",
                "stage": "train",
                "weights": weights.as_posix(),
                "data_yaml": data_yaml.as_posix(),
                "architecture": model_cfg.get("architecture", "yolov8n-seg"),
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
            }
            for k, v in params.items():
                safe_log_param(k, v)

        results = model.train(
            task="segment",
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

        if active_run is not None:
            # 训练阶段 Ultralytics 有时会在 results_dict 里带少量数值；能记则记，失败不影响主流程。
            for k, v in result_dict.items():
                safe_log_metric(f"train_{k}", v)

            safe_log_artifact(best_dst, artifact_path="checkpoints")
            if last_dst.exists():
                safe_log_artifact(last_dst, artifact_path="checkpoints")

            save_dir_raw = result_dict.get("save_dir")
            if save_dir_raw:
                safe_log_artifacts_from_dir(Path(save_dir_raw), artifact_path="train")

        train_metrics = {
            "task": "segmentation",
            "stage": "train",
            "status": "success",
            "mlflow_run_id": run_id,
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
                "enabled": mlflow_enabled,
                "experiment_name": experiment_name,
                "tracking_uri": tracking_uri,
                "run_id": run_id,
                "log_artifacts": mlflow_cfg.get("log_artifacts", True),
                "log_model": mlflow_cfg.get("log_model", False),
            },
            "raw_results": result_dict,
            "trained_at": datetime.now().isoformat(),
        }

        train_metrics_path = metrics_dir / "train_metrics.json"
        save_json(train_metrics, train_metrics_path)

        if run_id:
            (metrics_dir / "mlflow_run_id.txt").write_text(run_id, encoding="utf-8")

        if active_run is not None:
            safe_log_artifact(train_metrics_path, artifact_path="metrics")

    print("✓ Segmentation 模型训练完成")
    print(f"best.pt: {best_path}")
    print(f"last.pt:  {last_path}")
    print(f"训练指标: {metrics_dir / 'train_metrics.json'}")
    if run_id:
        print(f"MLflow run_id: {run_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
