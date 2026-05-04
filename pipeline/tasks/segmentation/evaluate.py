from __future__ import annotations

import json
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from pipeline.core.config import load_stage_config, project_path, require_file

try:
    import mlflow
except Exception:  # pragma: no cover - 允许无 MLflow 环境先跑通评估
    mlflow = None  # type: ignore


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_builtin(value: Any) -> Any:
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


def extract_results_dict(results: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}

    if hasattr(results, "results_dict"):
        try:
            data.update(to_builtin(results.results_dict))
        except Exception:
            pass

    if hasattr(results, "save_dir"):
        data["save_dir"] = Path(results.save_dir).as_posix()

    return data


def pick_metric(results_dict: dict[str, Any], candidates: list[str]) -> float | None:
    """
    Ultralytics 不同版本、不同任务的 metric key 可能略有差异。
    这里按候选 key 做兼容提取。
    """
    for key in candidates:
        if key in results_dict:
            try:
                return float(results_dict[key])
            except Exception:
                return None

    lowered = {str(k).lower(): k for k in results_dict.keys()}

    for candidate in candidates:
        c = candidate.lower()
        if c in lowered:
            key = lowered[c]
            try:
                return float(results_dict[key])
            except Exception:
                return None

    return None


def sanitize_mlflow_key(key: str) -> str:
    key = str(key).strip().replace("/", "_")
    key = re.sub(r"[^A-Za-z0-9_. -]+", "_", key)
    key = re.sub(r"\s+", "_", key)
    key = re.sub(r"_+", "_", key).strip("._- ")
    return key or "metric"


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


def safe_log_artifact(path: Path, artifact_path: str | None = None) -> None:
    if mlflow is None or not path.exists():
        return

    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as e:
        print(f"[WARN] MLflow log_artifact 失败: {path}, err={e}")


def safe_log_artifacts_from_dir(save_dir: Path, artifact_path: str = "eval") -> None:
    if mlflow is None or not save_dir.exists() or not save_dir.is_dir():
        return

    important_files = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
        "val_batch1_labels.jpg",
        "val_batch1_pred.jpg",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxPR_curve.png",
        "BoxR_curve.png",
        "MaskF1_curve.png",
        "MaskP_curve.png",
        "MaskPR_curve.png",
        "MaskR_curve.png",
    ]
    for name in important_files:
        p = save_dir / name
        if p.exists():
            safe_log_artifact(p, artifact_path=artifact_path)


def resolve_mlflow_context(metrics_dir: Path) -> tuple[bool, str | None, str, str, dict[str, Any]]:
    """
    优先读取 train_metrics.json 中的 mlflow_run_id，评估指标写回同一个 run。
    如果历史 train.py 还没写 run_id，则创建 eval 专用 run，保证实验会出现。
    """
    train_metrics_path = metrics_dir / "train_metrics.json"
    train_metrics: dict[str, Any] = {}
    if train_metrics_path.exists():
        try:
            train_metrics = load_json(train_metrics_path)
        except Exception as e:
            print(f"[WARN] 读取 train_metrics.json 失败: {e}")

    mlflow_info = train_metrics.get("mlflow", {}) if isinstance(train_metrics.get("mlflow"), dict) else {}
    tracking_uri = str(mlflow_info.get("tracking_uri") or "http://localhost:5000")
    experiment_name = str(mlflow_info.get("experiment_name") or "visionops-segmentation")
    run_id = str(train_metrics.get("mlflow_run_id") or mlflow_info.get("run_id") or "").strip()

    txt_path = metrics_dir / "mlflow_run_id.txt"
    if not run_id and txt_path.exists():
        run_id = txt_path.read_text(encoding="utf-8").strip()

    enabled = bool(mlflow_info.get("enabled", True))
    return enabled, run_id or None, tracking_uri, experiment_name, train_metrics


def main() -> None:
    cfg = load_stage_config("evaluate")

    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg.get("output", {})

    checkpoint_path = project_path(
        model_cfg.get("checkpoint_path", "models/checkpoints_segmentation/best.pt")
    )
    data_yaml = project_path(dataset_cfg.get("yaml_path", "data/processed_segmentation/data.yaml"))

    metrics_dir = project_path(output_cfg.get("metrics_dir", "models/metrics_segmentation"))
    eval_metrics_path = project_path(
        output_cfg.get("eval_metrics", metrics_dir / "eval_metrics.json")
    )

    img_size = int(eval_cfg.get("img_size", 640))
    batch_size = int(eval_cfg.get("batch_size", 16))
    device = eval_cfg.get("device", "cpu")

    require_file(checkpoint_path, "请先运行 dvc repro train 生成 models/checkpoints_segmentation/best.pt")
    require_file(data_yaml, "请先运行 dvc repro preprocess 生成 data/processed_segmentation/data.yaml")

    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Segmentation 模型评估开始")
    print("=" * 60)
    print(f"模型权重: {checkpoint_path}")
    print(f"数据配置: {data_yaml}")
    print(f"输入尺寸: {img_size}")
    print(f"batch_size: {batch_size}")
    print(f"device: {device}")
    print(f"指标输出: {eval_metrics_path}")
    print("=" * 60)

    model = YOLO(str(checkpoint_path))

    results = model.val(
        task="segment",
        data=str(data_yaml),
        imgsz=img_size,
        batch=batch_size,
        device=device,
    )

    results_dict = extract_results_dict(results)

    # 常见 Ultralytics seg 指标 key 兼容。
    box_map50 = pick_metric(results_dict, [
        "metrics/mAP50(B)",
        "metrics/box/mAP50",
        "metrics/mAP50",
        "map50",
    ])

    box_map50_95 = pick_metric(results_dict, [
        "metrics/mAP50-95(B)",
        "metrics/box/mAP50-95",
        "metrics/mAP50-95",
        "map50_95",
    ])

    mask_map50 = pick_metric(results_dict, [
        "metrics/mAP50(M)",
        "metrics/seg/mAP50",
        "metrics/mAP50(mask)",
        "metrics/mAP50(Mask)",
        "mask_map50",
    ])

    mask_map50_95 = pick_metric(results_dict, [
        "metrics/mAP50-95(M)",
        "metrics/seg/mAP50-95",
        "metrics/mAP50-95(mask)",
        "metrics/mAP50-95(Mask)",
        "mask_map50_95",
    ])

    eval_metrics = {
        "task": "segmentation",
        "stage": "evaluate",
        "status": "success",
        "checkpoint_path": checkpoint_path.as_posix(),
        "data_yaml": data_yaml.as_posix(),
        # 顶层 map50/map50_95 用 mask 指标，供 register_model 和 UI 直接读取。
        "map50": mask_map50,
        "map50_95": mask_map50_95,
        "box_metrics": {
            "map50": box_map50,
            "map50_95": box_map50_95,
        },
        "mask_metrics": {
            "map50": mask_map50,
            "map50_95": mask_map50_95,
        },
        "params": {
            "img_size": img_size,
            "batch_size": batch_size,
            "device": device,
        },
        "raw_results": results_dict,
        "evaluated_at": datetime.now().isoformat(),
    }

    mlflow_enabled, run_id, tracking_uri, experiment_name, train_metrics = resolve_mlflow_context(metrics_dir)
    eval_run_id: str | None = run_id

    if mlflow_enabled and mlflow is not None:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            if run_id:
                run_ctx = mlflow.start_run(run_id=run_id)
                print(f"[INFO] 评估指标写入已有 MLflow run: {run_id}")
            else:
                run_ctx = mlflow.start_run(run_name=f"segmentation-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
                print("[WARN] 未找到 train 阶段 run_id，创建 eval 专用 MLflow run")

            with run_ctx as active_run:
                eval_run_id = active_run.info.run_id

                safe_log_param("eval_task", "segmentation")
                safe_log_param("eval_checkpoint_path", checkpoint_path.as_posix())
                safe_log_param("eval_data_yaml", data_yaml.as_posix())
                safe_log_param("eval_img_size", img_size)
                safe_log_param("eval_batch_size", batch_size)
                safe_log_param("eval_device", device)

                safe_log_metric("map50", mask_map50)
                safe_log_metric("map50_95", mask_map50_95)
                safe_log_metric("mask_map50", mask_map50)
                safe_log_metric("mask_map50_95", mask_map50_95)
                safe_log_metric("box_map50", box_map50)
                safe_log_metric("box_map50_95", box_map50_95)

                # 尽量记录原始结果中的所有数值字段。
                for k, v in results_dict.items():
                    safe_log_metric(f"eval_{k}", v)

        except Exception as e:
            print(f"[WARN] MLflow 评估日志写入失败，评估结果仍会保存到本地 JSON: {e}")
    elif mlflow is None:
        print("[WARN] 未安装 mlflow，跳过 MLflow 评估日志写入")
    else:
        print("[INFO] train_metrics.json 中标记 MLflow disabled，跳过评估日志写入")

    eval_metrics["mlflow"] = {
        "enabled": bool(mlflow_enabled and mlflow is not None),
        "experiment_name": experiment_name,
        "tracking_uri": tracking_uri,
        "run_id": eval_run_id,
        "source_train_run_id": run_id,
    }

    save_json(eval_metrics, eval_metrics_path)

    if eval_run_id:
        (metrics_dir / "mlflow_run_id.txt").write_text(eval_run_id, encoding="utf-8")

    # eval_metrics_path 写完之后再作为 artifact 记录。
    if mlflow_enabled and mlflow is not None and eval_run_id:
        try:
            with mlflow.start_run(run_id=eval_run_id):
                safe_log_artifact(eval_metrics_path, artifact_path="metrics")
                save_dir_raw = results_dict.get("save_dir")
                if save_dir_raw:
                    safe_log_artifacts_from_dir(Path(save_dir_raw), artifact_path="eval")
        except Exception as e:
            print(f"[WARN] MLflow 评估 artifact 写入失败: {e}")

    print("✓ Segmentation 模型评估完成")
    print(f"mask mAP50:     {mask_map50}")
    print(f"mask mAP50-95:  {mask_map50_95}")
    print(f"box mAP50:      {box_map50}")
    print(f"box mAP50-95:   {box_map50_95}")
    print(f"评估指标: {eval_metrics_path}")
    if eval_run_id:
        print(f"MLflow run_id: {eval_run_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
