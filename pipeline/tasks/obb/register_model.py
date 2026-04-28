from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.core.config import load_stage_config, project_path, require_file


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def read_metrics(eval_metrics_path: Path) -> dict[str, Any]:
    if not eval_metrics_path.exists():
        raise FileNotFoundError(
            f"评估指标文件不存在: {eval_metrics_path}\n"
            "请先运行 dvc repro evaluate"
        )

    data = load_json(eval_metrics_path)

    metrics = data.get("metrics", {})
    return {
        "precision": safe_float(metrics.get("precision")),
        "recall": safe_float(metrics.get("recall")),
        "map50": safe_float(metrics.get("map50")),
        "map50_95": safe_float(metrics.get("map50_95")),
        "fitness": safe_float(metrics.get("fitness")),
        "raw": data,
    }


def read_train_metrics(train_metrics_path: Path) -> dict[str, Any] | None:
    if not train_metrics_path.exists():
        return None
    return load_json(train_metrics_path)


def read_convert_report(convert_report_path: Path) -> dict[str, Any] | None:
    if not convert_report_path.exists():
        return None
    return load_json(convert_report_path)


def should_promote(
    metrics: dict[str, Any],
    threshold: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    根据 task.yaml / generated 中的 promotion_threshold 判断是否建议推广。

    对 OBB 目前主要看：
    - map50 >= threshold.map50
    - map50_95 >= threshold.map50_95
    - latency_ms <= threshold.latency_ms，如果当前没有 latency_ms，则跳过
    """
    reasons: list[str] = []

    map50 = metrics.get("map50")
    map50_95 = metrics.get("map50_95")
    latency_ms = metrics.get("latency_ms")

    th_map50 = threshold.get("map50")
    th_map50_95 = threshold.get("map50_95")
    th_latency = threshold.get("latency_ms")

    ok = True

    if th_map50 is not None:
        if map50 is None:
            ok = False
            reasons.append("缺少 map50，无法判断是否达到推广阈值")
        elif float(map50) < float(th_map50):
            ok = False
            reasons.append(f"map50={map50:.6f} < 阈值 {float(th_map50):.6f}")
        else:
            reasons.append(f"map50={map50:.6f} >= 阈值 {float(th_map50):.6f}")

    if th_map50_95 is not None:
        if map50_95 is None:
            ok = False
            reasons.append("缺少 map50_95，无法判断是否达到推广阈值")
        elif float(map50_95) < float(th_map50_95):
            ok = False
            reasons.append(f"map50_95={map50_95:.6f} < 阈值 {float(th_map50_95):.6f}")
        else:
            reasons.append(f"map50_95={map50_95:.6f} >= 阈值 {float(th_map50_95):.6f}")

    if th_latency is not None and latency_ms is not None:
        if float(latency_ms) > float(th_latency):
            ok = False
            reasons.append(f"latency_ms={latency_ms:.3f} > 阈值 {float(th_latency):.3f}")
        else:
            reasons.append(f"latency_ms={latency_ms:.3f} <= 阈值 {float(th_latency):.3f}")

    if th_latency is not None and latency_ms is None:
        reasons.append("当前注册阶段没有边缘端 latency_ms，暂不判断延迟阈值")

    return ok, reasons


def maybe_log_to_mlflow(
    registry_cfg: dict[str, Any],
    result: dict[str, Any],
    rknn_model: Path,
    onnx_model: Path,
    eval_metrics_path: Path,
    train_metrics_path: Path,
    convert_report_path: Path,
) -> dict[str, Any]:
    """
    尝试写入 MLflow。

    如果 mlflow 未安装、服务未启动、连接失败，不中断主流程。
    register_model.py 的最低目标是生成 registry_result.json。
    """
    mlflow_result: dict[str, Any] = {
        "enabled": True,
        "status": "not_run",
        "run_id": None,
        "error": None,
    }

    try:
        import mlflow
    except Exception as e:
        mlflow_result.update(
            {
                "enabled": False,
                "status": "skipped",
                "error": f"mlflow 未安装或无法导入: {e}",
            }
        )
        return mlflow_result

    tracking_uri = registry_cfg.get("tracking_uri") or result.get("mlflow", {}).get("tracking_uri")
    experiment_name = registry_cfg.get("experiment_name") or result.get("mlflow", {}).get(
        "experiment_name",
        "visionops-obb",
    )
    model_name = registry_cfg.get("model_name", "visionops-obb-rk3588")

    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id

            mlflow.set_tags(result.get("tags", {}))
            mlflow.set_tag("task", "obb_detection")
            mlflow.set_tag("model_format", "rknn")
            mlflow.set_tag("bbox_type", "oriented_bbox")
            mlflow.set_tag("registered_model_name", model_name)

            metrics = result.get("metrics", {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            files = [
                rknn_model,
                onnx_model,
                eval_metrics_path,
                train_metrics_path,
                convert_report_path,
            ]
            for p in files:
                if p.exists():
                    mlflow.log_artifact(p.as_posix())

            mlflow_result.update(
                {
                    "status": "success",
                    "run_id": run_id,
                    "tracking_uri": tracking_uri,
                    "experiment_name": experiment_name,
                    "model_name": model_name,
                }
            )

    except Exception as e:
        mlflow_result.update(
            {
                "status": "failed_but_ignored",
                "error": str(e),
                "tracking_uri": tracking_uri,
                "experiment_name": experiment_name,
                "model_name": model_name,
            }
        )

    return mlflow_result


def main() -> None:
    cfg = load_stage_config("register_model")

    registry_cfg = cfg.get("registry", {})
    paths_cfg = cfg.get("paths", {})

    model_name = registry_cfg.get("model_name", "visionops-obb-rk3588")
    task = registry_cfg.get("task", "obb_detection")
    promotion_threshold = registry_cfg.get(
        "promotion_threshold",
        {
            "map50": 0.5,
            "map50_95": 0.3,
            "latency_ms": 150.0,
        },
    )
    tags = registry_cfg.get("tags", {})

    eval_metrics_path = project_path(
        paths_cfg.get("eval_metrics", "models/metrics_obb/eval_metrics.json")
    )
    train_metrics_path = project_path(
        paths_cfg.get("train_metrics", "models/metrics_obb/train_metrics.json")
    )
    onnx_model = project_path(
        paths_cfg.get("onnx_model", "models/export_obb/model.onnx")
    )
    rknn_model = project_path(
        paths_cfg.get("rknn_model", "models/export_obb/model.rknn")
    )
    registry_result_path = project_path(
        paths_cfg.get("registry_result", "models/metrics_obb/registry_result.json")
    )

    # 兼容 convert 报告路径：优先从 generated paths 读，没有就用默认 rknn_report.json
    convert_report_path = project_path(
        paths_cfg.get("convert_report", "models/export_obb/rknn_report.json")
    )

    require_file(eval_metrics_path, "请先运行 dvc repro evaluate")
    require_file(onnx_model, "请先运行 dvc repro export_onnx")
    require_file(rknn_model, "请先运行 dvc repro convert_rknn")

    metrics_data = read_metrics(eval_metrics_path)
    train_data = read_train_metrics(train_metrics_path)
    convert_report = read_convert_report(convert_report_path)

    metrics = {
        "precision": metrics_data.get("precision"),
        "recall": metrics_data.get("recall"),
        "map50": metrics_data.get("map50"),
        "map50_95": metrics_data.get("map50_95"),
        "fitness": metrics_data.get("fitness"),
    }

    # 如果后续 convert_report 或边缘验证写了 latency_ms，可以在这里自动带入。
    if convert_report:
        latency_ms = convert_report.get("latency_ms")
        if latency_ms is not None:
            metrics["latency_ms"] = safe_float(latency_ms)

    promote, promote_reasons = should_promote(metrics, promotion_threshold)

    rknn_md5 = file_md5(rknn_model)
    onnx_md5 = file_md5(onnx_model)

    model_version_name = (
        f"{task}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{rknn_md5[:8]}"
    )

    result: dict[str, Any] = {
        "task": task,
        "stage": "register_model",
        "status": "success",
        "model_name": model_name,
        "model_version_name": model_version_name,
        "model_format": "rknn",
        "bbox_type": "oriented_bbox",
        "paths": {
            "eval_metrics": eval_metrics_path.as_posix(),
            "train_metrics": train_metrics_path.as_posix() if train_metrics_path.exists() else None,
            "onnx_model": onnx_model.as_posix(),
            "rknn_model": rknn_model.as_posix(),
            "convert_report": convert_report_path.as_posix() if convert_report_path.exists() else None,
            "registry_result": registry_result_path.as_posix(),
        },
        "artifacts": {
            "onnx_size_bytes": onnx_model.stat().st_size,
            "rknn_size_bytes": rknn_model.stat().st_size,
            "onnx_md5": onnx_md5,
            "rknn_md5": rknn_md5,
        },
        "metrics": metrics,
        "promotion": {
            "recommended": promote,
            "threshold": promotion_threshold,
            "reasons": promote_reasons,
        },
        "tags": {
            **tags,
            "task": task,
            "model_format": "rknn",
            "bbox_type": "oriented_bbox",
            "auto_registered": "true",
        },
        "train_summary": {
            "available": train_data is not None,
            "trained_at": train_data.get("trained_at") if isinstance(train_data, dict) else None,
            "best_pt": train_data.get("best_pt") if isinstance(train_data, dict) else None,
            "last_pt": train_data.get("last_pt") if isinstance(train_data, dict) else None,
        },
        "convert_summary": {
            "available": convert_report is not None,
            "target_platform": convert_report.get("target_platform") if isinstance(convert_report, dict) else None,
            "do_quantization": convert_report.get("do_quantization") if isinstance(convert_report, dict) else None,
            "onnx_shapes": convert_report.get("onnx_shapes") if isinstance(convert_report, dict) else None,
            "expected_runtime_output": convert_report.get("expected_runtime_output")
            if isinstance(convert_report, dict)
            else None,
        },
        "registered_at": datetime.now().isoformat(),
    }

    # 可选 MLflow。默认尝试，但失败不阻断。
    mlflow_result = maybe_log_to_mlflow(
        registry_cfg=registry_cfg,
        result=result,
        rknn_model=rknn_model,
        onnx_model=onnx_model,
        eval_metrics_path=eval_metrics_path,
        train_metrics_path=train_metrics_path,
        convert_report_path=convert_report_path,
    )
    result["mlflow"] = mlflow_result

    save_json(result, registry_result_path)

    print("=" * 60)
    print("OBB 模型注册完成")
    print("=" * 60)
    print(f"模型名称: {model_name}")
    print(f"模型版本: {model_version_name}")
    print(f"任务类型: {task}")
    print(f"ONNX: {onnx_model}")
    print(f"RKNN: {rknn_model}")
    print(f"RKNN MD5: {rknn_md5}")
    print(f"RKNN 大小: {rknn_model.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"precision: {metrics.get('precision')}")
    print(f"recall:    {metrics.get('recall')}")
    print(f"mAP50:     {metrics.get('map50')}")
    print(f"mAP50-95:  {metrics.get('map50_95')}")
    print(f"是否建议推广: {promote}")
    for reason in promote_reasons:
        print(f"  - {reason}")
    print(f"MLflow 状态: {mlflow_result.get('status')}")
    if mlflow_result.get("run_id"):
        print(f"MLflow run_id: {mlflow_result.get('run_id')}")
    if mlflow_result.get("error"):
        print(f"MLflow 提示: {mlflow_result.get('error')}")
    print(f"注册结果: {registry_result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
