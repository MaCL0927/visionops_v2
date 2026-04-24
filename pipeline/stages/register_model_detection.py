from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_rknn_report_path(paths_cfg: dict) -> Path:
    # detection_mlops.yaml 目前未显式给出 rknn_report，
    # 这里按 dvc.yaml 的 detection RKNN 输出约定推断：
    # models/export_detection/rknn_perf_report.json
    explicit = paths_cfg.get("rknn_report")
    if explicit:
        return Path(explicit)

    rknn_model = Path(paths_cfg["rknn_model"])
    return rknn_model.parent / "rknn_perf_report.json"


def resolve_rknn_source(paths_cfg: dict) -> Tuple[bool, Dict[str, Any]]:
    """
    判断 RKNN 是否可作为正式注册源：

    可用条件：
    1. rknn 文件存在
    2. rknn perf report 存在
    3. report.status != simulated
    """
    rknn_path = Path(paths_cfg["rknn_model"])
    report_path = infer_rknn_report_path(paths_cfg)

    if not rknn_path.exists():
        return False, {
            "reason": "rknn file not found",
            "report_path": report_path.as_posix(),
        }

    if not report_path.exists():
        return False, {
            "reason": "rknn perf report not found",
            "report_path": report_path.as_posix(),
        }

    try:
        report = load_json(report_path)
    except Exception as e:
        return False, {
            "reason": f"failed to parse rknn perf report: {e}",
            "report_path": report_path.as_posix(),
        }

    status = str(report.get("status", "")).lower()
    deployable = report.get("deployable", None)

    if status == "simulated":
        return False, {
            "reason": "rknn is simulated placeholder",
            "report_path": report_path.as_posix(),
            "status": status,
            "deployable": deployable,
            "report": report,
        }

    if deployable is False:
        return False, {
            "reason": "rknn report marks deployable=false",
            "report_path": report_path.as_posix(),
            "status": status,
            "deployable": deployable,
            "report": report,
        }

    return True, {
        "reason": None,
        "report_path": report_path.as_posix(),
        "status": status or "ok",
        "deployable": deployable,
        "report": report,
    }


def ensure_registered_model_exists(client: MlflowClient, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
        print(f"MLflow Registered Model 已存在: {model_name}")
    except Exception:
        client.create_registered_model(model_name)
        print(f"已创建 MLflow Registered Model: {model_name}")


def log_metrics_if_any(metrics: Dict[str, Any]) -> None:
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, float(v))


def log_params_if_any(params: Dict[str, Any]) -> None:
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            mlflow.log_param(k, json.dumps(v, ensure_ascii=False))
        else:
            mlflow.log_param(k, v)

def resolve_upstream_run_id(
    train_metrics: Dict[str, Any],
    mlflow_run_id_path: Optional[Path],
) -> Tuple[str, str]:
    """
    优先从 train_metrics.json 读取上游训练 run_id；
    若没有，再从可选辅助文件 mlflow_run_id.txt 兜底读取。
    返回: (run_id, source)
    """
    run_id_from_metrics = str(train_metrics.get("mlflow_run_id", "") or "").strip()
    if run_id_from_metrics:
        return run_id_from_metrics, "train_metrics.json"

    if mlflow_run_id_path and mlflow_run_id_path.exists():
        run_id_from_txt = read_text(mlflow_run_id_path)
        if run_id_from_txt:
            return run_id_from_txt, "mlflow_run_id.txt"

    return "", ""

def ensure_registration_run(
    model_name: str,
    run_id: str,
    eval_metrics: dict,
    train_metrics: dict,
    paths_cfg: dict,
    registry_cfg: dict,
) -> str:
    """
    如果上游训练阶段已提供 mlflow_run_id，则直接复用。
    否则新建一个注册专用 run，并补齐必要 artifact。
    """
    if run_id:
        print(f"复用已有 MLflow run_id: {run_id}")
        return run_id

    print("未找到上游 mlflow_run_id，创建注册专用 MLflow Run...")
    with mlflow.start_run(run_name=f"register-{model_name}") as run:
        created_run_id = run.info.run_id

        log_metrics_if_any(eval_metrics)
        log_params_if_any({
            "task": registry_cfg.get("task", "detection"),
            "model_name": model_name,
            "source": "register_model_detection",
            "auto_registered": True,
        })

        onnx_path = Path(paths_cfg["onnx_model"])
        if onnx_path.exists():
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")

        use_rknn, rknn_info = resolve_rknn_source(paths_cfg)
        rknn_path = Path(paths_cfg["rknn_model"])
        if use_rknn and rknn_path.exists():
            mlflow.log_artifact(str(rknn_path), artifact_path="rknn")

        eval_metrics_path = Path(paths_cfg["eval_metrics"])
        if eval_metrics_path.exists():
            mlflow.log_artifact(str(eval_metrics_path), artifact_path="metrics")

        train_metrics_path = Path(paths_cfg.get("train_metrics", ""))
        if train_metrics_path and train_metrics_path.exists():
            mlflow.log_artifact(str(train_metrics_path), artifact_path="metrics")

        report_path = infer_rknn_report_path(paths_cfg)
        if report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="rknn")

        print(f"已创建注册专用 run_id: {created_run_id}")
        return created_run_id


def main() -> None:
    cfg_path = Path("pipeline/configs/detection_mlops.yaml")
    cfg = load_yaml(cfg_path)

    registry_cfg = cfg["registry"]
    paths_cfg = cfg["paths"]

    model_name = registry_cfg["model_name"]
    task = registry_cfg.get("task", "detection")
    tags = registry_cfg.get("tags", {})
    thresholds = registry_cfg.get("promotion_threshold", {})

    eval_metrics_path = Path(paths_cfg["eval_metrics"])
    train_metrics_path = Path(paths_cfg.get("train_metrics", ""))
    mlflow_run_id_raw = paths_cfg.get("mlflow_run_id", "")
    mlflow_run_id_path = Path(mlflow_run_id_raw) if mlflow_run_id_raw else None
    onnx_path = Path(paths_cfg["onnx_model"])
    rknn_path = Path(paths_cfg["rknn_model"])
    registry_result_path = Path(paths_cfg["registry_result"])

    if not eval_metrics_path.exists():
        raise FileNotFoundError(f"缺少评估指标文件: {eval_metrics_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"缺少 ONNX 模型文件: {onnx_path}")

    eval_metrics = load_json(eval_metrics_path)
    train_metrics = load_json(train_metrics_path) if train_metrics_path.exists() else {}

    tracking_uri = mlflow.get_tracking_uri()
    print("=" * 60)
    print("Detection 模型注册开始")
    print("=" * 60)
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"Model Name: {model_name}")
    print(f"Task: {task}")
    print(f"Eval Metrics: {eval_metrics_path}")
    print(f"ONNX Path: {onnx_path}")
    print(f"RKNN Path: {rknn_path}")
    print("=" * 60)

    upstream_run_id, run_id_source = resolve_upstream_run_id(
        train_metrics=train_metrics,
        mlflow_run_id_path=mlflow_run_id_path,
    )

    if upstream_run_id:
        print(f"检测到上游 MLflow run_id，来源: {run_id_source} -> {upstream_run_id}")
    else:
        print("未在 train_metrics.json 或辅助 txt 中找到上游 mlflow_run_id，将创建注册专用 MLflow Run...")

    run_id = ensure_registration_run(
        model_name=model_name,
        run_id=upstream_run_id,
        eval_metrics=eval_metrics,
        train_metrics=train_metrics,
        paths_cfg=paths_cfg,
        registry_cfg=registry_cfg,
    )

    use_rknn, rknn_info = resolve_rknn_source(paths_cfg)
    if use_rknn:
        source_uri = f"runs:/{run_id}/rknn"
        source_type = "rknn"
        print("检测到可用的正式 RKNN 产物，将优先注册 RKNN。")
    else:
        source_uri = f"runs:/{run_id}/onnx"
        source_type = "onnx"
        print(f"RKNN 不可用，回退为 ONNX 注册。原因: {rknn_info.get('reason')}")

    client = MlflowClient()
    ensure_registered_model_exists(client, model_name)

    version = client.create_model_version(
        name=model_name,
        source=source_uri,
        run_id=run_id,
        tags={
            **{k: str(v) for k, v in tags.items()},
            "task": str(task),
            "source_type": source_type,
            "registered_at": datetime.now().isoformat(),
            "auto_fallback_to_onnx": str(not use_rknn).lower(),
        },
    )

    version_number = version.version
    print(f"已创建模型版本: {model_name} v{version_number}")

    # 给版本写说明性标签，方便 UI 检索
    extra_tags = {
        "map50": str(eval_metrics.get("map50", eval_metrics.get("mAP50", ""))),
        "map50_95": str(eval_metrics.get("map50_95", eval_metrics.get("mAP50-95", ""))),
        "precision": str(eval_metrics.get("precision", "")),
        "recall": str(eval_metrics.get("recall", "")),
        "rknn_usable": str(use_rknn).lower(),
        "rknn_status": str(rknn_info.get("status", "")),
    }
    for k, v in extra_tags.items():
        if v != "":
            client.set_model_version_tag(model_name, version_number, k, v)

    # 是否满足自动晋升阈值
    map50 = coerce_float(eval_metrics.get("map50", eval_metrics.get("mAP50")), 0.0)
    map50_95 = coerce_float(eval_metrics.get("map50_95", eval_metrics.get("mAP50-95")), 0.0)
    latency_ms = coerce_float(
        safe_get(rknn_info, "report", "latency_ms"),
        None,
    )

    min_map50 = coerce_float(thresholds.get("map50"), 0.0)
    min_map50_95 = coerce_float(thresholds.get("map50_95"), 0.0)
    max_latency_ms = coerce_float(thresholds.get("latency_ms"), None)

    passed = True
    reasons = []

    if map50 is not None and min_map50 is not None and map50 < min_map50:
        passed = False
        reasons.append(f"map50={map50:.4f} < {min_map50:.4f}")

    if map50_95 is not None and min_map50_95 is not None and map50_95 < min_map50_95:
        passed = False
        reasons.append(f"map50_95={map50_95:.4f} < {min_map50_95:.4f}")

    if (
        max_latency_ms is not None
        and latency_ms is not None
        and latency_ms > max_latency_ms
    ):
        passed = False
        reasons.append(f"latency_ms={latency_ms:.4f} > {max_latency_ms:.4f}")

    stage_to_set = "Staging" if passed else "None"
    if passed:
        client.transition_model_version_stage(
            name=model_name,
            version=version_number,
            stage="Staging",
            archive_existing_versions=False,
        )
        print(f"模型版本 v{version_number} 已晋升到 Staging")
    else:
        print("模型版本未自动晋升到 Staging")
        if reasons:
            print("原因：")
            for r in reasons:
                print(f" - {r}")

    result = {
        "success": True,
        "model_name": model_name,
        "task": task,
        "run_id": run_id,
        "version": version_number,
        "source_uri": source_uri,
        "source_type": source_type,
        "stage": stage_to_set,
        "rknn_usable": use_rknn,
        "rknn_status": rknn_info.get("status"),
        "rknn_report_path": rknn_info.get("report_path"),
        "rknn_fallback_reason": None if use_rknn else rknn_info.get("reason"),
        "thresholds": {
            "map50": min_map50,
            "map50_95": min_map50_95,
            "latency_ms": max_latency_ms,
        },
        "metrics": {
            "map50": map50,
            "map50_95": map50_95,
            "precision": coerce_float(eval_metrics.get("precision")),
            "recall": coerce_float(eval_metrics.get("recall")),
            "latency_ms": latency_ms,
        },
        "promotion_passed": passed,
        "promotion_reasons": reasons,
        "registered_at": datetime.now().isoformat(),
    }

    save_json(result, registry_result_path)

    print("=" * 60)
    print("✓ Detection 模型注册完成")
    print(f"版本: v{version_number}")
    print(f"注册源: {source_type}")
    print(f"结果文件: {registry_result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
