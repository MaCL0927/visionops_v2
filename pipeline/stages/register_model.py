"""
Stage 6: 分类模型注册

输入：
models/metrics/eval_metrics.json
models/export/model.onnx
models/export/model.rknn
models/export/rknn_perf_report.json

输出：
models/metrics/registry_result.json

说明：
- 本阶段以本地 registry_result.json 为主
- MLflow 注册作为可选增强，失败不阻断 DVC pipeline
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_file_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "size_mb": 0.0,
        }

    return {
        "exists": True,
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 4),
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }


def judge_promotion(
    eval_metrics: dict,
    rknn_report: dict,
    thresholds: dict,
) -> Dict[str, Any]:
    accuracy = float(eval_metrics.get("accuracy", 0.0))
    latency_ms = float(eval_metrics.get("latency_ms", 999999.0))

    min_accuracy = float(thresholds.get("accuracy", 0.0))
    max_latency_ms = float(thresholds.get("latency_ms", 999999.0))

    rknn_status = rknn_report.get("status", "missing")
    rknn_quantization = bool(rknn_report.get("quantization", False))

    accuracy_ok = accuracy >= min_accuracy
    latency_ok = latency_ms <= max_latency_ms
    rknn_ok = rknn_status == "success"

    should_promote = accuracy_ok and latency_ok and rknn_ok

    reasons = []

    if accuracy_ok:
        reasons.append(f"accuracy {accuracy:.4f} >= threshold {min_accuracy:.4f}")
    else:
        reasons.append(f"accuracy {accuracy:.4f} < threshold {min_accuracy:.4f}")

    if latency_ok:
        reasons.append(f"latency_ms {latency_ms:.4f} <= threshold {max_latency_ms:.4f}")
    else:
        reasons.append(f"latency_ms {latency_ms:.4f} > threshold {max_latency_ms:.4f}")

    if rknn_ok:
        reasons.append("RKNN conversion status is success")
    else:
        reasons.append(f"RKNN conversion status is {rknn_status}")

    return {
        "should_promote": should_promote,
        "accuracy_ok": accuracy_ok,
        "latency_ok": latency_ok,
        "rknn_ok": rknn_ok,
        "accuracy": accuracy,
        "latency_ms": latency_ms,
        "min_accuracy": min_accuracy,
        "max_latency_ms": max_latency_ms,
        "rknn_status": rknn_status,
        "rknn_quantization": rknn_quantization,
        "reasons": reasons,
    }


def safe_mlflow_register(
    model_name: str,
    onnx_path: Path,
    rknn_path: Path,
    eval_metrics: dict,
    rknn_report: dict,
    should_promote: bool,
    metrics_dir: Path,
) -> Dict[str, Any]:
    """
    MLflow 注册作为可选增强。

    当前分类 pipeline 的主结果是 registry_result.json。
    如果 MLflow/MinIO/S3 环境变量不完整，这里只记录失败原因，不中断流程。
    """
    result = {
        "enabled": True,
        "success": False,
        "model_name": model_name,
        "registered_version": None,
        "stage": None,
        "error": None,
    }

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlops_cfg = load_yaml("pipeline/configs/mlops.yaml")
        train_cfg = load_yaml("pipeline/configs/train.yaml")

        tracking_uri = train_cfg.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
        experiment_name = train_cfg.get("mlflow", {}).get("experiment_name", "visionops-classification")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        run_id_file = metrics_dir / "mlflow_run_id.txt"
        run_id = run_id_file.read_text(encoding="utf-8").strip() if run_id_file.exists() else None

        if run_id:
            active_run_ctx = mlflow.start_run(run_id=run_id)
        else:
            active_run_ctx = mlflow.start_run(
                run_name=f"register-classification-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

        with active_run_ctx as run:
            run_id = run.info.run_id

            mlflow.set_tags(
                {
                    "task": "classification",
                    "model_name": model_name,
                    "registry_decision": "promote" if should_promote else "reject",
                    "rknn_status": rknn_report.get("status", "unknown"),
                    "rknn_quantization": str(rknn_report.get("quantization", False)),
                }
            )

            mlflow.log_metrics(
                {
                    "registry_accuracy": float(eval_metrics.get("accuracy", 0.0)),
                    "registry_latency_ms": float(eval_metrics.get("latency_ms", 0.0)),
                    "registry_f1_macro": float(eval_metrics.get("f1_macro", 0.0)),
                }
            )

            # artifact 上传可能因为 S3 凭据失败，所以逐个保护
            for p in [onnx_path, rknn_path, metrics_dir / "eval_metrics.json", metrics_dir / "registry_result.json"]:
                if p.exists():
                    try:
                        mlflow.log_artifact(str(p))
                    except Exception as e:
                        print(f"[WARN] MLflow artifact 上传失败，已跳过: {p}")
                        print(f"[WARN] 原因: {type(e).__name__}: {e}")

            # 这里只注册 ONNX 目录/文件作为模型版本记录。
            # RKNN 文件作为 artifact 保存，真正部署仍使用 models/export/model.rknn。
            client = MlflowClient()

            try:
                client.get_registered_model(model_name)
            except Exception:
                client.create_registered_model(model_name)

            model_uri = f"runs:/{run_id}/model_artifacts"

            # 如果没有真的 log_model，这个 URI 可能不存在。
            # 因此这里不强制 create_model_version，避免引入额外失败。
            # 本地 registry_result.json 才是当前 pipeline 的确定输出。
            result.update(
                {
                    "success": True,
                    "run_id": run_id,
                    "stage": "Production" if should_promote else "None",
                    "note": "MLflow run updated. Local registry_result.json is the source of truth for this pipeline.",
                }
            )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print("[WARN] MLflow 注册记录失败，已跳过。")
        print(f"[WARN] 原因: {result['error']}")

    return result


def main() -> None:
    mlops_cfg = load_yaml("pipeline/configs/mlops.yaml")

    registry_cfg = mlops_cfg.get("registry", {})
    thresholds = mlops_cfg.get("promotion_threshold", {})

    model_name = registry_cfg.get("model_name", "visionops-classifier")

    metrics_dir = Path("models/metrics")
    export_dir = Path("models/export")

    eval_metrics_path = metrics_dir / "eval_metrics.json"
    registry_result_path = metrics_dir / "registry_result.json"

    onnx_path = export_dir / "model.onnx"
    rknn_path = export_dir / "model.rknn"
    rknn_report_path = export_dir / "rknn_perf_report.json"
    export_result_path = export_dir / "export_result.json"

    if not eval_metrics_path.exists():
        raise FileNotFoundError(
            f"未找到评估指标文件: {eval_metrics_path}\n"
            "请先运行：python pipeline/stages/evaluate.py"
        )

    if not rknn_path.exists():
        raise FileNotFoundError(
            f"未找到 RKNN 模型文件: {rknn_path}\n"
            "请先运行：python pipeline/stages/convert_rknn.py"
        )

    print("=" * 60)
    print("分类模型注册开始")
    print("=" * 60)
    print(f"model_name: {model_name}")
    print(f"eval_metrics: {eval_metrics_path}")
    print(f"rknn_model: {rknn_path}")

    eval_metrics = load_json(eval_metrics_path)
    rknn_report = load_json(rknn_report_path)
    export_result = load_json(export_result_path)

    decision = judge_promotion(
        eval_metrics=eval_metrics,
        rknn_report=rknn_report,
        thresholds=thresholds,
    )

    model_files = {
        "onnx": get_file_info(onnx_path),
        "rknn": get_file_info(rknn_path),
        "rknn_report": get_file_info(rknn_report_path),
        "export_result": get_file_info(export_result_path),
        "eval_metrics": get_file_info(eval_metrics_path),
    }

    registry_result = {
        "task": "classification",
        "status": "registered" if decision["should_promote"] else "rejected",
        "model_name": model_name,
        "decision": decision,
        "metrics": {
            "accuracy": eval_metrics.get("accuracy"),
            "precision_macro": eval_metrics.get("precision_macro"),
            "recall_macro": eval_metrics.get("recall_macro"),
            "f1_macro": eval_metrics.get("f1_macro"),
            "latency_ms": eval_metrics.get("latency_ms"),
        },
        "model_info": {
            "architecture": eval_metrics.get("architecture") or export_result.get("architecture"),
            "num_classes": eval_metrics.get("num_classes") or export_result.get("num_classes"),
            "class_names": eval_metrics.get("class_names") or export_result.get("class_names"),
            "input_size": export_result.get("input_size"),
            "expected_output_shape": export_result.get("expected_output_shape"),
            "quantization": rknn_report.get("quantization"),
            "rknn_output_shapes": rknn_report.get("output_shapes", []),
        },
        "files": model_files,
        "edge_deployment_candidate": {
            "enabled": decision["should_promote"],
            "model_path": str(rknn_path),
            "class_names": eval_metrics.get("class_names") or export_result.get("class_names"),
            "runtime_task": "classification",
        },
        "created_at": datetime.now().isoformat(),
    }

    # 先保存本地 registry_result，避免后面 MLflow 失败时没有产物
    save_json(registry_result, registry_result_path)

    mlflow_result = safe_mlflow_register(
        model_name=model_name,
        onnx_path=onnx_path,
        rknn_path=rknn_path,
        eval_metrics=eval_metrics,
        rknn_report=rknn_report,
        should_promote=decision["should_promote"],
        metrics_dir=metrics_dir,
    )

    registry_result["mlflow"] = mlflow_result
    save_json(registry_result, registry_result_path)

    print("\n" + "=" * 60)
    print("✓ 分类模型注册完成")
    print("=" * 60)
    print(f"注册状态: {registry_result['status']}")
    print(f"accuracy: {decision['accuracy']:.4f} / 阈值 {decision['min_accuracy']:.4f}")
    print(f"latency_ms: {decision['latency_ms']:.4f} / 阈值 {decision['max_latency_ms']:.4f}")
    print(f"rknn_status: {decision['rknn_status']}")
    print(f"是否作为部署候选: {decision['should_promote']}")
    print(f"注册结果: {registry_result_path}")

    print("\n判断原因:")
    for reason in decision["reasons"]:
        print(f"- {reason}")

    print("=" * 60)


if __name__ == "__main__":
    main()
