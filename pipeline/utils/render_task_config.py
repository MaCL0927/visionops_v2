from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
TASK_FILE = ROOT / "pipeline/configs/task.yaml"

OUT_DET_DATA = ROOT / "pipeline/configs/detection_data.generated.yaml"
OUT_DET_TRAIN = ROOT / "pipeline/configs/detection_train.generated.yaml"
OUT_DET_EXPORT = ROOT / "pipeline/configs/detection_export.generated.yaml"
OUT_DET_RKNN = ROOT / "pipeline/configs/detection_rknn.generated.yaml"
OUT_CLASS_NAMES = ROOT / "edge/runtime/class_names.yaml"
OUT_EDGE_ENV = ROOT / "edge/runtime/edge.env"


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def validate_task_cfg(cfg: dict):
    class_names = cfg["classes"]["names"]
    num_classes = cfg["classes"]["num_classes"]
    if len(class_names) != num_classes:
        raise ValueError(
            f"classes.num_classes={num_classes} 与 names 数量 {len(class_names)} 不一致"
        )

    raw_yaml = ROOT / cfg["dataset"]["raw_yaml"]
    if not raw_yaml.exists():
        raise FileNotFoundError(f"原始数据 YAML 不存在: {raw_yaml}")

    raw_cfg = load_yaml(raw_yaml)
    raw_names = raw_cfg.get("names", [])
    if raw_names != class_names:
        raise ValueError(
            f"task.yaml 的 classes.names 与 {raw_yaml} 的 names 不一致\n"
            f"task.yaml: {class_names}\nraw_yaml: {raw_names}"
        )


def build_detection_data(cfg: dict) -> dict:
    return {
        "dataset": {
            "root": cfg["dataset"]["raw_root"],
            "yaml_path": cfg["dataset"]["raw_yaml"],
            "task": "detect",
            "class_names": cfg["classes"]["names"],
            "num_classes": cfg["classes"]["num_classes"],
        },
        "preprocess": {
            "copy_to_processed": True,
            "processed_root": cfg["dataset"]["processed_root"],
            "check_images": True,
            "check_labels": True,
        },
    }


def build_detection_train(cfg: dict) -> dict:
    return {
        "model": {
            "architecture": cfg["model"]["architecture"],
            "weights": cfg["model"]["pretrained_weights"],
            "num_classes": cfg["classes"]["num_classes"],
            "pretrained": True,
        },
        "dataset": {
            "yaml_path": cfg["dataset"]["processed_yaml"],
        },
        "train": {
            "epochs": cfg["train"]["epochs"],
            "batch_size": cfg["train"]["batch_size"],
            "img_size": cfg["model"]["input_size"],
            "lr0": cfg["train"]["lr0"],
            "device": cfg["train"]["device"],
            "workers": cfg["train"]["workers"],
            "patience": cfg["train"]["patience"],
            "cache": cfg["train"]["cache"],
            "project": cfg["train"]["project"],
            "name": cfg["train"]["name"],
            "exist_ok": cfg["train"]["exist_ok"],
        },
        "mlflow": {
            "experiment_name": "visionops-detection",
            "tracking_uri": "http://localhost:5000",
            "log_artifacts": True,
            "log_model": False,
        },
        "output": {
            "checkpoint_dir": "models/checkpoints_detection",
            "metrics_dir": "models/metrics_detection",
        },
    }


def build_detection_export(cfg: dict) -> dict:
    export_cfg = cfg["export"]

    result = {
        "export": {
            "checkpoint_path": export_cfg["checkpoint_path"],
            "output_path": export_cfg["output_path"],
            "imgsz": export_cfg["imgsz"],
            "opset": export_cfg["opset"],
            "simplify": export_cfg["simplify"],
            "dynamic": export_cfg["dynamic"],
            "mode": export_cfg["mode"],
        }
    }

    if "external_export" in cfg:
        result["external_export"] = {
            "python_exec": cfg["external_export"].get("python_exec", "python"),
            "script_path": cfg["external_export"].get(
                "script_path", "tools/export_yolov8_rknn_onnx.py"
            ),
        }

    return result
    

def build_detection_rknn(cfg: dict) -> dict:
    imgsz = int(cfg["model"]["input_size"])
    rknn_cfg = cfg["rknn"]

    return {
        "target_platform": rknn_cfg["target_platform"],
        "output": {
            "onnx_model": rknn_cfg["onnx_model"],
            "rknn_model": rknn_cfg["rknn_model"],
            "perf_report": rknn_cfg["perf_report"],
        },
        "quantization": {
            "do_quantization": rknn_cfg["build"]["do_quantization"],
            "dataset": rknn_cfg["quantization"]["dataset"],
            "dataset_size": rknn_cfg["quantization"]["dataset_size"],
            "quantized_dtype": rknn_cfg["quantization"]["quantized_dtype"],
        },
        "io_config": {
            "mean_values": rknn_cfg["input"]["mean_values"],
            "std_values": rknn_cfg["input"]["std_values"],
            "input_size_list": [[1, 3, imgsz, imgsz]],
        },
        "optimization_level": rknn_cfg["build"]["optimization_level"],
        "perf_debug": rknn_cfg["runtime"]["perf_debug"],
        "eval_mem": rknn_cfg["runtime"]["eval_mem"],
    }

def build_class_names(cfg: dict) -> dict:
    return {
        "task": cfg["task"]["name"],
        "class_names": cfg["classes"]["names"],
        "num_classes": cfg["classes"]["num_classes"],
    }


def build_edge_env(cfg: dict) -> str:
    lines = [
        "DEVICE_ID=rk3588-001",
        "MODEL_PATH=/opt/visionops/models/current.rknn",
        f"NPU_CORE={cfg['edge']['npu_core']}",
        f"NUM_CLASSES={cfg['classes']['num_classes']}",
        "CLASS_NAMES_FILE=/opt/visionops/edge/runtime/class_names.yaml",
        f"PORT={cfg['edge']['port']}",
        f"METRICS_PORT={cfg['edge']['metrics_port']}",
        f"CONF_THRESHOLD={cfg['edge']['conf_threshold']}",
        f"NMS_THRESHOLD={cfg['edge']['nms_threshold']}",
        f"WARMUP_RUNS={cfg['edge']['warmup_runs']}",
    ]
    return "\n".join(lines) + "\n"


def main():
    cfg = load_yaml(TASK_FILE)
    validate_task_cfg(cfg)

    dump_yaml(OUT_DET_DATA, build_detection_data(cfg))
    dump_yaml(OUT_DET_TRAIN, build_detection_train(cfg))
    dump_yaml(OUT_DET_EXPORT, build_detection_export(cfg))
    dump_yaml(OUT_DET_RKNN, build_detection_rknn(cfg))
    dump_yaml(OUT_CLASS_NAMES, build_class_names(cfg))
    write_text(OUT_EDGE_ENV, build_edge_env(cfg))

    print("✓ generated:")
    print(f"  - {OUT_DET_DATA}")
    print(f"  - {OUT_DET_TRAIN}")
    print(f"  - {OUT_DET_EXPORT}")
    print(f"  - {OUT_DET_RKNN}")
    print(f"  - {OUT_CLASS_NAMES}")
    print(f"  - {OUT_EDGE_ENV}")


if __name__ == "__main__":
    main()
