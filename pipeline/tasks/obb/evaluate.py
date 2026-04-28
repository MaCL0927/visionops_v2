from __future__ import annotations

import json
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

    # 再做一次模糊兜底
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


def main() -> None:
    cfg = load_stage_config("evaluate")

    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg.get("output", {})

    checkpoint_path = project_path(
        model_cfg.get("checkpoint_path", "models/checkpoints_obb/best.pt")
    )
    data_yaml = project_path(dataset_cfg.get("yaml_path", "data/processed_obb/data.yaml"))

    metrics_dir = project_path(output_cfg.get("metrics_dir", "models/metrics_obb"))
    eval_metrics_path = project_path(
        output_cfg.get("eval_metrics", metrics_dir / "eval_metrics.json")
    )

    img_size = int(eval_cfg.get("img_size", 640))
    batch_size = int(eval_cfg.get("batch_size", 16))
    device = eval_cfg.get("device", "cpu")

    require_file(checkpoint_path, "请先运行 dvc repro train 生成 models/checkpoints_obb/best.pt")
    require_file(data_yaml, "请先运行 dvc repro preprocess 生成 data/processed_obb/data.yaml")

    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OBB 模型评估开始")
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
        task="obb",
        data=str(data_yaml),
        imgsz=img_size,
        batch=batch_size,
        device=device,
    )

    results_dict = extract_results_dict(results)

    # 常见 Ultralytics OBB / detection 指标 key 兼容。
    precision = pick_metric(
        results_dict,
        [
            "metrics/precision(B)",
            "metrics/precision(OBB)",
            "metrics/precision",
            "precision",
        ],
    )
    recall = pick_metric(
        results_dict,
        [
            "metrics/recall(B)",
            "metrics/recall(OBB)",
            "metrics/recall",
            "recall",
        ],
    )
    map50 = pick_metric(
        results_dict,
        [
            "metrics/mAP50(B)",
            "metrics/mAP50(OBB)",
            "metrics/mAP50",
            "map50",
        ],
    )
    map50_95 = pick_metric(
        results_dict,
        [
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(OBB)",
            "metrics/mAP50_95",
            "map50_95",
        ],
    )
    fitness = pick_metric(
        results_dict,
        [
            "fitness",
            "metrics/fitness",
        ],
    )

    eval_metrics = {
        "task": "obb_detection",
        "stage": "evaluate",
        "status": "success",
        "checkpoint_path": checkpoint_path.as_posix(),
        "data_yaml": data_yaml.as_posix(),
        "metrics": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "fitness": fitness,
        },
        "params": {
            "img_size": img_size,
            "batch_size": batch_size,
            "device": device,
        },
        "raw_results": results_dict,
        "evaluated_at": datetime.now().isoformat(),
    }

    save_json(eval_metrics, eval_metrics_path)

    print("✓ OBB 模型评估完成")
    print(f"precision: {precision}")
    print(f"recall:    {recall}")
    print(f"mAP50:     {map50}")
    print(f"mAP50-95:  {map50_95}")
    print(f"fitness:   {fitness}")
    print(f"评估指标: {eval_metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
