from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_eval_metrics(cfg: dict) -> dict:
    paths_cfg = cfg["paths"]
    eval_metrics_path = Path(paths_cfg["eval_metrics"])
    train_metrics_path = Path(paths_cfg["train_metrics"])

    if eval_metrics_path.exists():
        return load_json(eval_metrics_path)

    # 如果没有 eval_metrics，退回 train_metrics
    if train_metrics_path.exists():
        data = load_json(train_metrics_path)
        best_metrics = data.get("best_metrics", {})
        return {
            "task": "detection",
            "map50": best_metrics.get("map50", 0.0),
            "map50_95": best_metrics.get("map50_95", 0.0),
            "precision": best_metrics.get("precision", 0.0),
            "recall": best_metrics.get("recall", 0.0),
            "source": "train_metrics",
        }

    return {
        "task": "detection",
        "map50": 0.0,
        "map50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "source": "none",
    }


def check_promotion_threshold(metrics: dict, thresholds: dict) -> Tuple[bool, str]:
    map50 = safe_float(metrics.get("map50", 0.0))
    map50_95 = safe_float(metrics.get("map50_95", 0.0))
    latency_ms = metrics.get("latency_ms")

    required_map50 = safe_float(thresholds.get("map50", 0.70))
    required_map50_95 = safe_float(thresholds.get("map50_95", 0.25))
    max_latency = safe_float(thresholds.get("latency_ms", 50))

    if map50 < required_map50:
        return False, f"mAP50 {map50:.4f} < 阈值 {required_map50:.4f}"

    if map50_95 < required_map50_95:
        return False, f"mAP50-95 {map50_95:.4f} < 阈值 {required_map50_95:.4f}"

    if latency_ms is not None and safe_float(latency_ms) > max_latency:
        return False, f"延迟 {latency_ms}ms > 阈值 {max_latency}ms"

    return True, "所有检测指标满足晋升阈值"


def resolve_run_id(paths_cfg: dict) -> str | None:
    run_id_path = Path(paths_cfg["mlflow_run_id"])
    if run_id_path.exists():
        return run_id_path.read_text(encoding="utf-8").strip()
    return None


def ensure_registration_run(
    client: MlflowClient,
    tracking_uri: str,
    paths_cfg: dict,
    metrics: dict,
) -> str:
    """
    如果训练阶段已经有 run_id，则直接复用。
    如果没有，则新建一个 run 并上传 ONNX/RKNN 产物。
    """
    run_id = resolve_run_id(paths_cfg)
    if run_id:
        return run_id

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("visionops-detection-registration")

    onnx_path = Path(paths_cfg["onnx_model"])
    rknn_path = Path(paths_cfg["rknn_model"])

    with mlflow.start_run(
        run_name=f"register-detection-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ) as run:
        run_id = run.info.run_id

        # 上传本地模型产物
        if rknn_path.exists():
            mlflow.log_artifact(str(rknn_path), artifact_path="rknn")
        if onnx_path.exists():
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")

        # 记录关键指标
        log_metrics = {}
        for k in ["map50", "map50_95", "precision", "recall", "latency_ms"]:
            if k in metrics:
                try:
                    log_metrics[k] = float(metrics[k])
                except Exception:
                    pass
        if log_metrics:
            mlflow.log_metrics(log_metrics)

        return run_id


def ensure_registered_model(client: MlflowClient, model_name: str, description: str) -> None:
    try:
        client.get_registered_model(model_name)
        print(f"已存在的注册模型: {model_name}")
    except Exception:
        client.create_registered_model(model_name, description=description)
        print(f"创建新注册模型: {model_name}")


def register_model(cfg: dict, metrics: dict) -> dict:
    registry_cfg = cfg["registry"]
    paths_cfg = cfg["paths"]

    model_name = registry_cfg.get("model_name", "visionops-detector-detection")
    tags = registry_cfg.get("tags", {})
    thresholds = registry_cfg.get("promotion_threshold", {})

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    run_id = ensure_registration_run(client, tracking_uri, paths_cfg, metrics)

    rknn_path = Path(paths_cfg["rknn_model"])
    onnx_path = Path(paths_cfg["onnx_model"])

    print("\n注册 detection 模型到 MLflow Registry...")
    print(f"模型名称: {model_name}")
    print(f"Run ID: {run_id}")

    ensure_registered_model(
        client,
        model_name=model_name,
        description="VisionOps Detection 模型 | YOLOv8 | 目标平台: RK3588",
    )

    # 优先注册 RKNN 产物；如果没有，则退回 ONNX
    if rknn_path.exists():
        source_uri = f"runs:/{run_id}/rknn"
        source_type = "rknn"
    else:
        source_uri = f"runs:/{run_id}/onnx"
        source_type = "onnx"

    mv = client.create_model_version(
        name=model_name,
        source=source_uri,
        run_id=run_id,
        description=(
            f"自动注册 detection 模型 | "
            f"mAP50={metrics.get('map50', 'N/A')} | "
            f"mAP50-95={metrics.get('map50_95', 'N/A')} | "
            f"{datetime.now().isoformat()}"
        ),
    )
    version = mv.version
    print(f"创建版本: {version}")
    print(f"注册源: {source_type} -> {source_uri}")

    can_promote, reason = check_promotion_threshold(metrics, thresholds)

    if can_promote:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        stage = "Production"
        print(f"✓ 模型晋升到 Production！原因: {reason}")
    else:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
        )
        stage = "Staging"
        print(f"⚠ 模型留在 Staging。原因: {reason}")

    # 写标签
    merged_tags = {
        **tags,
        "source_type": source_type,
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for k, v in merged_tags.items():
        client.set_model_version_tag(model_name, version, k, str(v))

    result = {
        "status": "success",
        "task": "detection",
        "model_name": model_name,
        "version": version,
        "stage": stage,
        "promoted": can_promote,
        "promotion_reason": reason,
        "run_id": run_id,
        "source_type": source_type,
        "registered_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    return result


def main() -> None:
    cfg_path = Path("pipeline/configs/detection_mlops.yaml")
    cfg = load_yaml(cfg_path)

    metrics = load_eval_metrics(cfg)

    print("=" * 60)
    print("Detection 模型注册到 MLflow Model Registry")
    print("当前指标:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("=" * 60)

    result = register_model(cfg, metrics)

    registry_result_path = Path(cfg["paths"]["registry_result"])
    save_json(result, registry_result_path)

    print("\n" + "=" * 60)
    print("✓ Detection 注册完成!")
    print(f"模型: {result['model_name']} v{result['version']}")
    print(f"阶段: {result['stage']}")
    print(f"是否晋升 Production: {result['promoted']}")
    print(f"结果文件: {registry_result_path}")
    print("=" * 60)

    # 不把“未晋升”当错误
    sys.exit(0)


if __name__ == "__main__":
    main()
