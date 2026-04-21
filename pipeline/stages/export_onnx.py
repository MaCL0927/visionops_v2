"""
ONNX模型导出 Stage
将PyTorch checkpoint导出为ONNX格式，为RKNN转换做准备
"""
import os
import sys
import yaml
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def export_onnx(cfg: dict) -> dict:
    onnx_cfg = cfg.get("onnx", {})
    opset = onnx_cfg.get("opset_version", 12)
    input_size = onnx_cfg.get("input_size", [1, 3, 640, 640])
    output_path = "models/export/model.onnx"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = "models/checkpoints/best.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")

    train_cfg = load_config("pipeline/configs/train.yaml")
    arch = train_cfg["model"]["architecture"]

    if arch.lower().startswith("yolov8"):
        # YOLOv8有自己的export
        try:
            from ultralytics import YOLO
            model = YOLO(checkpoint_path)
            export_result = model.export(format="onnx", opset=opset,
                                          imgsz=input_size[-1], simplify=True)
            # ultralytics导出后移动到期望路径
            import shutil
            exported = str(export_result)
            if exported != output_path:
                shutil.move(exported, output_path)
        except ImportError:
            raise ImportError("请安装 ultralytics")
    else:
        # torchvision模型
        from server.training.model_utils import build_model
        model = build_model(arch, train_cfg["model"]["num_classes"], pretrained=False)

        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()

        dummy_input = torch.randn(*input_size)
        dynamic_axes = onnx_cfg.get("dynamic_axes", None)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        # 验证ONNX
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX模型验证通过")
        except ImportError:
            print("onnx未安装，跳过验证")

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    result = {
        "status": "success",
        "onnx_path": output_path,
        "size_mb": round(size_mb, 2),
        "opset_version": opset,
        "input_size": input_size,
        "exported_at": datetime.now().isoformat(),
    }
    return result


def main():
    cfg_export = load_config("pipeline/configs/export.yaml") if Path("pipeline/configs/export.yaml").exists() else {}
    cfg_train = load_config("pipeline/configs/train.yaml")
    cfg = {**cfg_train, **cfg_export}

    print("=" * 50)
    print("导出ONNX模型")
    print("=" * 50)

    result = export_onnx(cfg)

    Path("models/export").mkdir(parents=True, exist_ok=True)
    print(f"\n✓ 导出成功: {result['onnx_path']} ({result['size_mb']} MB)")


if __name__ == "__main__":
    main()
