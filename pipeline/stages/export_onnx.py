"""
Stage 4: 分类模型导出 ONNX

输入：
models/checkpoints/best.pt

输出：
models/export/
├── model.onnx
└── export_result.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import yaml
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.training.model_utils import build_model


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_torch_load(path: Path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_static_input_size(input_size) -> List[int]:
    """
    RKNN 更适合静态输入 shape。
    这里强制检查 ONNX input_size 为 [1, 3, H, W]。
    """
    if not isinstance(input_size, list) or len(input_size) != 4:
        raise ValueError(
            f"onnx.input_size 必须是 [1, 3, H, W] 格式，当前为: {input_size}"
        )

    if input_size[0] != 1:
        raise ValueError(
            f"分类 RKNN 建议 batch 固定为 1，当前 input_size={input_size}"
        )

    if input_size[1] != 3:
        raise ValueError(
            f"分类图像模型输入通道数应为 3，当前 input_size={input_size}"
        )

    h, w = int(input_size[2]), int(input_size[3])
    if h <= 0 or w <= 0:
        raise ValueError(f"非法输入尺寸: {input_size}")

    return [1, 3, h, w]


def main() -> None:
    train_cfg = load_yaml("pipeline/configs/train.yaml")
    export_cfg = load_yaml("pipeline/configs/export.yaml")

    output_cfg = train_cfg.get("output", {})
    onnx_cfg = export_cfg.get("onnx", {})

    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "models/checkpoints/"))
    checkpoint_path = checkpoint_dir / "best.pt"

    export_dir = Path("models/export")
    export_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_dir / "model.onnx"
    export_result_path = export_dir / "export_result.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"未找到 checkpoint: {checkpoint_path}\n"
            "请先运行：python pipeline/stages/train.py"
        )

    ckpt = safe_torch_load(checkpoint_path, map_location="cpu")

    architecture = ckpt.get(
        "architecture",
        train_cfg.get("model", {}).get("architecture", "mobilenetv3"),
    )
    num_classes = int(
        ckpt.get(
            "num_classes",
            train_cfg.get("model", {}).get("num_classes", 0),
        )
    )
    class_names = ckpt.get("class_names", [])
    img_size = ckpt.get("img_size", train_cfg.get("train", {}).get("img_size", [224, 224]))
    normalize = ckpt.get("normalize", {})

    input_size = onnx_cfg.get("input_size", [1, 3, img_size[0], img_size[1]])
    input_size = ensure_static_input_size(input_size)

    opset_version = int(onnx_cfg.get("opset_version", 12))
    simplify = bool(onnx_cfg.get("simplify", True))
    dynamic_axes = onnx_cfg.get("dynamic_axes", None)

    if dynamic_axes not in [None, "null"]:
        raise ValueError(
            "分类 RKNN 导出建议不要使用 dynamic_axes。\n"
            "请将 pipeline/configs/export.yaml 中 onnx.dynamic_axes 设置为 null。"
        )

    if num_classes <= 0:
        raise ValueError(f"num_classes 非法: {num_classes}")

    if class_names and len(class_names) != num_classes:
        raise ValueError(
            f"class_names 数量与 num_classes 不一致: "
            f"len(class_names)={len(class_names)}, num_classes={num_classes}"
        )

    print("=" * 60)
    print("分类 ONNX 导出开始")
    print("=" * 60)
    print(f"checkpoint: {checkpoint_path}")
    print(f"模型结构: {architecture}")
    print(f"类别数: {num_classes}")
    print(f"类别名: {class_names}")
    print(f"输入尺寸: {input_size}")
    print(f"opset_version: {opset_version}")
    print(f"dynamic_axes: {dynamic_axes}")
    print(f"ONNX 输出路径: {onnx_path}")

    model = build_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(*input_size, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes=None,
        )

    onnx_check_ok = False
    onnx_simplified = False

    try:
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        onnx_check_ok = True
        print("✓ ONNX checker 检查通过")
    except Exception as e:
        print(f"[WARN] ONNX checker 检查失败: {type(e).__name__}: {e}")

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            onnx_model = onnx.load(str(onnx_path))
            simplified_model, check = onnx_simplify(onnx_model)

            if check:
                onnx.save(simplified_model, str(onnx_path))
                onnx_simplified = True
                print("✓ ONNX simplify 成功")
            else:
                print("[WARN] ONNX simplify check 未通过，保留原模型")
        except ImportError:
            print("[WARN] 未安装 onnxsim，跳过 simplify")
        except Exception as e:
            print(f"[WARN] ONNX simplify 失败，保留原模型: {type(e).__name__}: {e}")

    # 简单验证 ONNXRuntime 输出形状
    ort_check_ok = False
    ort_output_shape = None
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {"images": dummy_input.numpy().astype(np.float32)}
        ort_outputs = sess.run(None, ort_inputs)

        if len(ort_outputs) != 1:
            raise RuntimeError(f"分类模型应只有 1 个输出，当前输出数量: {len(ort_outputs)}")

        ort_output_shape = list(ort_outputs[0].shape)

        expected_shape = [1, num_classes]
        if ort_output_shape != expected_shape:
            raise RuntimeError(
                f"ONNX 输出 shape 不符合分类任务预期。"
                f"期望 {expected_shape}，实际 {ort_output_shape}"
            )

        ort_check_ok = True
        print(f"✓ ONNXRuntime 验证通过，输出 shape={ort_output_shape}")

    except ImportError:
        print("[WARN] 未安装 onnxruntime，跳过 ONNXRuntime 验证")
    except Exception as e:
        print(f"[WARN] ONNXRuntime 验证失败: {type(e).__name__}: {e}")

    onnx_size_mb = onnx_path.stat().st_size / 1024 / 1024

    export_result = {
        "task": "classification",
        "status": "success",
        "checkpoint": str(checkpoint_path),
        "onnx_model": str(onnx_path),
        "onnx_size_mb": round(onnx_size_mb, 4),
        "architecture": architecture,
        "num_classes": num_classes,
        "class_names": class_names,
        "input_size": input_size,
        "output_names": ["logits"],
        "expected_output_shape": [1, num_classes],
        "opset_version": opset_version,
        "dynamic_axes": None,
        "simplify": simplify,
        "onnx_check_ok": onnx_check_ok,
        "onnx_simplified": onnx_simplified,
        "ort_check_ok": ort_check_ok,
        "ort_output_shape": ort_output_shape,
        "normalize": normalize,
        "exported_at": datetime.now().isoformat(),
        "rknn_notes": {
            "static_shape_required": True,
            "recommended_rknn_input_size_list": [input_size],
            "recommended_output_postprocess": "softmax(logits) -> argmax",
            "do_not_add_softmax_before_export": True,
            "mean_std_should_match_training": True,
        },
    }

    with open(export_result_path, "w", encoding="utf-8") as f:
        json.dump(export_result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("✓ 分类 ONNX 导出完成")
    print("=" * 60)
    print(f"ONNX 路径: {onnx_path}")
    print(f"ONNX 大小: {onnx_size_mb:.4f} MB")
    print(f"输出 shape: [1, {num_classes}]")
    print(f"导出报告: {export_result_path}")


if __name__ == "__main__":
    main()
