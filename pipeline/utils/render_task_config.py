from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.core.config import PROJECT_ROOT, get_task_type, load_task_config
from pipeline.core.io import save_yaml

ROOT = PROJECT_ROOT
CONFIG_DIR = ROOT / "pipeline" / "configs"
RUNTIME_DIR = ROOT / "edge" / "runtime"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _as_hw(value: Any, default: list[int]) -> list[int]:
    if isinstance(value, int):
        return [value, value]
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        if len(parts) == 1 and parts[0].isdigit():
            n = int(parts[0])
            return [n, n]
        if len(parts) >= 2:
            return [int(parts[0]), int(parts[1])]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [int(value[0]), int(value[1])]
    return default


def _class_names(cfg: dict[str, Any]) -> list[str]:
    names = cfg.get("classes", {}).get("names", [])
    if isinstance(names, dict):
        def key_sort(k):
            try:
                return int(k)
            except Exception:
                return str(k)
        names = [names[k] for k in sorted(names, key=key_sort)]
    names = [str(x) for x in names]
    num = int(cfg.get("classes", {}).get("num_classes", len(names)))
    if len(names) != num:
        raise ValueError(f"classes.num_classes={num} 与 names 数量 {len(names)} 不一致")
    return names


def build_runtime_class_names(cfg: dict[str, Any], task_type: str, input_size: list[int]) -> dict[str, Any]:
    names = _class_names(cfg)
    edge = cfg.get("edge", {})
    return {
        "task": task_type,
        "task_type": task_type,
        "input_size": input_size,
        "num_classes": len(names),
        "class_names": names,
        "names": names,
        "topk": int(edge.get("topk", min(5, len(names)))) if task_type == "classification" else None,
        "conf_threshold": float(edge.get("conf_threshold", 0.25)),
        "nms_threshold": float(edge.get("nms_threshold", 0.45)),
    }


def build_edge_env(cfg: dict[str, Any], task_type: str, input_size: list[int]) -> str:
    names = _class_names(cfg)
    edge = cfg.get("edge", {})
    lines = [
        f"DEVICE_ID={edge.get('device_id', 'rk3588-001')}",
        "MODEL_PATH=/opt/visionops/models/current.rknn",
        f"INFERENCE_URL=http://localhost:{edge.get('port', 8080)}",
        f"REPORT_INTERVAL={edge.get('report_interval', 60)}",
        f"TASK={task_type}",
        f"NPU_CORE={edge.get('npu_core', 'auto')}",
        f"NUM_CLASSES={len(names)}",
        f"INPUT_SIZE={input_size[0]},{input_size[1]}",
        "CLASS_NAMES_FILE=/opt/visionops/edge/runtime/class_names.yaml",
        f"PORT={edge.get('port', 8080)}",
        f"METRICS_PORT={edge.get('metrics_port', 9091)}",
        f"CONF_THRESHOLD={edge.get('conf_threshold', 0.25)}",
        f"NMS_THRESHOLD={edge.get('nms_threshold', 0.45)}",
        f"TOPK={edge.get('topk', min(5, len(names)))}",
        f"WARMUP_RUNS={edge.get('warmup_runs', 3)}",
    ]
    return "\n".join(lines) + "\n"


def build_detection(cfg: dict[str, Any]) -> dict[Path, dict[str, Any]]:
    names = _class_names(cfg)
    model = cfg.get("model", {})
    train = cfg.get("train", {})
    ds = cfg.get("dataset", {})
    out = cfg.get("output", {})
    export = cfg.get("export", {})
    rknn = cfg.get("rknn", {})
    mlflow = cfg.get("mlflow", {})
    img = int(_as_hw(model.get("input_size", 640), [640, 640])[0])

    return {
        CONFIG_DIR / "detection_data.generated.yaml": {
            "dataset": {
                "root": ds.get("raw_root", "data/raw_detection"),
                "yaml_path": ds.get("raw_yaml", "data/raw_detection/data.yaml"),
                "task": "detect",
                "class_names": names,
                "num_classes": len(names),
            },
            "preprocess": {
                "copy_to_processed": True,
                "processed_root": ds.get("processed_root", "data/processed_detection"),
                "check_images": True,
                "check_labels": True,
            },
        },
        CONFIG_DIR / "detection_train.generated.yaml": {
            "model": {
                "architecture": model.get("architecture", "yolov8n"),
                "weights": model.get("pretrained_weights", "yolov8n.pt"),
                "num_classes": len(names),
                "pretrained": True,
            },
            "dataset": {"yaml_path": ds.get("processed_yaml", "data/processed_detection/data.yaml")},
            "train": {
                "epochs": train.get("epochs", 50),
                "batch_size": train.get("batch_size", 16),
                "img_size": img,
                "lr0": train.get("lr0", 0.001),
                "device": train.get("device", "cpu"),
                "workers": train.get("workers", 4),
                "patience": train.get("patience", 20),
                "cache": train.get("cache", False),
                "project": train.get("project", "models/runs/detect"),
                "name": train.get("name", "visionops_detection"),
                "exist_ok": train.get("exist_ok", True),
            },
            "mlflow": {
                "experiment_name": mlflow.get("experiment_name", "visionops-detection"),
                "tracking_uri": mlflow.get("tracking_uri", "http://localhost:5000"),
                "log_artifacts": True,
                "log_model": False,
            },
            "output": {
                "checkpoint_dir": out.get("checkpoint_dir", "models/checkpoints_detection"),
                "metrics_dir": out.get("metrics_dir", "models/metrics_detection"),
            },
        },
        CONFIG_DIR / "detection_export.generated.yaml": {
            "export": {
                "checkpoint_path": export.get("checkpoint_path", "models/checkpoints_detection/best.pt"),
                "output_path": export.get("output_path", "models/export_detection/model.onnx"),
                "imgsz": export.get("imgsz", img),
                "opset": export.get("opset", 12),
                "simplify": export.get("simplify", True),
                "dynamic": export.get("dynamic", False),
                "mode": export.get("mode", "rockchip"),
            },
            "external_export": cfg.get("external_export", {}),
        },
        CONFIG_DIR / "detection_rknn.generated.yaml": {
            "python_exec": rknn.get("python_exec", "python"),
            "target_platform": rknn.get("target_platform", "rk3588"),
            "output": {
                "onnx_model": rknn.get("onnx_model", "models/export_detection/model.onnx"),
                "rknn_model": rknn.get("rknn_model", "models/export_detection/model.rknn"),
                "perf_report": rknn.get("perf_report", "models/export_detection/rknn_report.json"),
            },
            "quantization": {
                "dataset": rknn.get("quantization", {}).get("dataset", "data/processed_detection/images/val"),
                "dataset_size": rknn.get("quantization", {}).get("dataset_size", 100),
                "quantized_dtype": rknn.get("quantization", {}).get("quantized_dtype", "asymmetric_quantized-8"),
            },
            "io_config": {
                "mean_values": rknn.get("input", {}).get("mean_values", [[0, 0, 0]]),
                "std_values": rknn.get("input", {}).get("std_values", [[255, 255, 255]]),
                "input_size_list": [[1, 3, img, img]],
            },
            "build": {
                "do_quantization": rknn.get("build", {}).get("do_quantization", True),
                "optimization_level": rknn.get("build", {}).get("optimization_level", 3),
            },
            "runtime": {
                "perf_debug": rknn.get("runtime", {}).get("perf_debug", False),
                "eval_mem": rknn.get("runtime", {}).get("eval_mem", False),
            },
            "check_output_shapes": rknn.get("check_output_shapes", True),
            "perf_debug": rknn.get("runtime", {}).get("perf_debug", False),
        },
        CONFIG_DIR / "detection_mlops.generated.yaml": {
            "registry": {
                "model_name": mlflow.get("registered_model_name", "visionops-detection-rk3588"),
                "task": "detection",
                "promotion_threshold": mlflow.get(
                    "promotion_threshold",
                    {"map50": 0.5, "map50_95": 0.3, "latency_ms": 120.0},
                ),
                "tags": {
                    "platform": rknn.get("target_platform", "rk3588"),
                    "quantized": str(rknn.get("build", {}).get("do_quantization", True)).lower(),
                    "auto_registered": "true",
                    "task": "detection",
                    "model_family": model.get("architecture", "yolov8"),
                },
            },
            "paths": {
                "eval_metrics": "models/metrics_detection/eval_metrics.json",
                "train_metrics": "models/metrics_detection/train_metrics.json",
                "mlflow_run_id": "models/metrics_detection/mlflow_run_id.txt",
                "onnx_model": "models/export_detection/model.onnx",
                "rknn_model": "models/export_detection/model.rknn",
                "registry_result": "models/metrics_detection/registry_result.json",
            },
        },
    }


def build_classification(cfg: dict[str, Any]) -> dict[Path, dict[str, Any]]:
    names = _class_names(cfg)
    ds = cfg.get("dataset", {})
    model = cfg.get("model", {})
    train = cfg.get("train", {})
    export = cfg.get("export", {})
    rknn = cfg.get("rknn", {})
    mlflow = cfg.get("mlflow", {})
    input_hw = _as_hw(model.get("input_size", [224, 224]), [224, 224])
    processed = ds.get("processed_root", "data/processed_classification")
    normalize = train.get("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})

    return {
        CONFIG_DIR / "classification_data.generated.yaml": {
            "preprocess": {
                "img_size": input_hw,
                "augment": train.get("augment", False),
                "val_split": ds.get("val_split", 0.2),
                "resize": True,
                "normalize": normalize,
                "num_workers": train.get("num_workers", train.get("workers", 4)),
            },
            "paths": {
                "raw_data": ds.get("raw_root", "data/raw_classification"),
                "processed": processed,
            },
            "classes": {"names": names, "num_classes": len(names)},
        },
        CONFIG_DIR / "classification_train.generated.yaml": {
            "model": {
                "architecture": model.get("architecture", "mobilenetv3"),
                "num_classes": len(names),
                "pretrained": model.get("pretrained", True),
            },
            "train": {
                "epochs": train.get("epochs", 30),
                "batch_size": train.get("batch_size", 16),
                "img_size": input_hw,
                "lr": train.get("lr", 0.001),
                "optimizer": train.get("optimizer", "adamw"),
                "lr_scheduler": train.get("lr_scheduler", "cosine"),
                "weight_decay": train.get("weight_decay", 1e-4),
            },
            "early_stopping": train.get("early_stopping", {"enabled": True, "patience": 10, "min_delta": 0.0}),
            "amp": train.get("amp", True),
            "device": train.get("device", "cuda"),
            "mlflow": {
                "experiment_name": mlflow.get("experiment_name", "visionops-classification"),
                "tracking_uri": mlflow.get("tracking_uri", "http://localhost:5000"),
            },
            "output": {
                "checkpoint_dir": "models/checkpoints_classification",
                "metrics_dir": "models/metrics_classification",
            },
        },
        CONFIG_DIR / "classification_export.generated.yaml": {
            "onnx": {
                "opset_version": export.get("opset", export.get("opset_version", 12)),
                "dynamic_axes": export.get("dynamic_axes"),
                "input_size": [1, 3, input_hw[0], input_hw[1]],
                "simplify": export.get("simplify", False),
                "output_path": export.get("output_path", "models/export_classification/model.onnx"),
            }
        },
        CONFIG_DIR / "classification_rknn.generated.yaml": {
            "python_exec": rknn.get("python_exec", "python"),
            "target_platform": rknn.get("target_platform", "rk3588"),
            "quantization": {
                "do_quantization": rknn.get("build", {}).get(
                    "do_quantization",
                    rknn.get("quantization", {}).get("enable", False),
                ),
                "dataset": rknn.get("quantization", {}).get("dataset", f"{processed}/val"),
                "dataset_size": rknn.get("quantization", {}).get("dataset_size", 100),
                "quantized_dtype": rknn.get("quantization", {}).get("quantized_dtype", "asymmetric_quantized-8"),
            },
            "io_config": {
                "input_size_list": [[1, 3, input_hw[0], input_hw[1]]],
                "mean_values": rknn.get("input", {}).get("mean_values", [[123.675, 116.28, 103.53]]),
                "std_values": rknn.get("input", {}).get("std_values", [[58.395, 57.12, 57.375]]),
            },
            "output": {
                "onnx_model": rknn.get("onnx_model", "models/export_classification/model.onnx"),
                "rknn_model": rknn.get("rknn_model", "models/export_classification/model.rknn"),
                "perf_report": rknn.get("perf_report", "models/export_classification/rknn_report.json"),
            },
            "npu_core_mask": rknn.get("npu_core_mask"),
            "perf_debug": False,
            "eval_mem": False,
            "check_output_shapes": True,
        },
        CONFIG_DIR / "classification_mlops.generated.yaml": {
            "registry": {
                "model_name": mlflow.get("registered_model_name", "visionops-classification-rk3588")
            },
            "promotion_threshold": mlflow.get(
                "promotion_threshold",
                {"accuracy": 0.8, "latency_ms": 999999},
            ),
            "paths": {
                "metrics_dir": "models/metrics_classification",
                "export_dir": "models/export_classification",
                "eval_metrics": "models/metrics_classification/eval_metrics.json",
                "rknn_model": "models/export_classification/model.rknn",
                "registry_result": "models/metrics_classification/registry_result.json",
            },
        },
    }


def main() -> None:
    cfg = load_task_config()
    task_type = get_task_type(cfg)
    input_hw = _as_hw(
        cfg.get("model", {}).get("input_size", [640, 640] if task_type == "detection" else [224, 224]),
        [640, 640] if task_type == "detection" else [224, 224],
    )
    generated = build_detection(cfg) if task_type == "detection" else build_classification(cfg)
    for path, data in generated.items():
        save_yaml(data, path)
    save_yaml(build_runtime_class_names(cfg, task_type, input_hw), RUNTIME_DIR / "class_names.yaml")
    write_text(RUNTIME_DIR / "edge.env", build_edge_env(cfg, task_type, input_hw))

    print("✓ generated:")
    for p in list(generated.keys()) + [RUNTIME_DIR / "class_names.yaml", RUNTIME_DIR / "edge.env"]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
