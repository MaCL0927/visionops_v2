from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx

from pipeline.core.config import load_stage_config, project_path, require_file


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _shape_from_value_info(value_info: Any) -> list[int | str]:
    dims: list[int | str] = []
    shape = value_info.type.tensor_type.shape

    for dim in shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append("?")

    return dims


def inspect_onnx_shapes(onnx_path: Path) -> dict[str, Any]:
    model = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(model)

    inputs = {
        x.name: _shape_from_value_info(x)
        for x in model.graph.input
    }

    outputs = {
        x.name: _shape_from_value_info(x)
        for x in model.graph.output
    }

    return {
        "inputs": inputs,
        "outputs": outputs,
    }


def need_reexec_in_rknn_env(python_exec: str) -> bool:
    """
    当前环境如果不能 import rknn.api.RKNN，就切换到 task.yaml 指定的 rknn python 环境。
    """
    try:
        from rknn.api import RKNN  # noqa: F401
        return False
    except Exception:
        current_python = Path(sys.executable).resolve()
        target_python = Path(python_exec).expanduser().resolve()

        if not target_python.exists():
            raise FileNotFoundError(
                f"当前环境无法 import rknn.api.RKNN，且配置的 python_exec 不存在: {target_python}"
            )

        if current_python == target_python:
            raise RuntimeError(
                "当前 Python 已经是 rknn python_exec，但仍无法 import rknn.api.RKNN。"
                "请检查 rknn-toolkit2 是否安装。"
            )

        return True


def reexec_with_rknn_python(python_exec: str) -> None:
    cmd = [python_exec, "-m", "pipeline.stages.convert_rknn"]
    print("[INFO] 当前环境没有 rknn-toolkit2，切换到 RKNN 环境执行:")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    sys.exit(0)


def collect_quant_images(dataset_dir: Path, max_count: int) -> list[Path]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not dataset_dir.exists():
        raise FileNotFoundError(f"量化数据目录不存在: {dataset_dir}")

    images = sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    )

    if not images:
        raise FileNotFoundError(f"量化数据目录中没有找到图片: {dataset_dir}")

    return images[:max_count]


def make_quant_dataset_file(dataset_dir: Path, dataset_size: int) -> Path:
    images = collect_quant_images(dataset_dir, dataset_size)

    tmp_dir = Path(tempfile.mkdtemp(prefix="visionops_obb_rknn_quant_"))
    dataset_txt = tmp_dir / "dataset.txt"

    with dataset_txt.open("w", encoding="utf-8") as f:
        for img in images:
            f.write(img.resolve().as_posix() + "\n")

    print(f"[INFO] 量化图片数量: {len(images)}")
    print(f"[INFO] 量化 dataset.txt: {dataset_txt}")

    return dataset_txt


def normalize_nested_list(value: Any, default: list[list[float]]) -> list[list[float]]:
    if value is None:
        return default

    if isinstance(value, list):
        return value

    return default


def main() -> None:
    cfg = load_stage_config("convert_rknn")

    python_exec = str(cfg.get("python_exec", "python"))

    if need_reexec_in_rknn_env(python_exec):
        reexec_with_rknn_python(python_exec)

    from rknn.api import RKNN

    target_platform = str(cfg.get("target_platform", "rk3588"))

    output_cfg = cfg.get("output", {})
    quant_cfg = cfg.get("quantization", {})
    io_cfg = cfg.get("io_config", {})
    build_cfg = cfg.get("build", {})
    runtime_cfg = cfg.get("runtime", {})

    onnx_model = project_path(output_cfg.get("onnx_model", "models/export_obb/model.onnx"))
    rknn_model = project_path(output_cfg.get("rknn_model", "models/export_obb/model.rknn"))
    perf_report = project_path(output_cfg.get("perf_report", "models/export_obb/rknn_report.json"))

    require_file(onnx_model, "请先运行 dvc repro export_onnx 生成 OBB ONNX")

    rknn_model.parent.mkdir(parents=True, exist_ok=True)
    perf_report.parent.mkdir(parents=True, exist_ok=True)

    do_quantization = bool(build_cfg.get("do_quantization", True))
    optimization_level = int(build_cfg.get("optimization_level", 3))

    dataset_dir = project_path(quant_cfg.get("dataset", "data/processed_obb/images/val"))
    dataset_size = int(quant_cfg.get("dataset_size", 100))
    quantized_dtype = str(quant_cfg.get("quantized_dtype", "asymmetric_quantized-8"))

    mean_values = normalize_nested_list(
        io_cfg.get("mean_values"),
        [[0, 0, 0]],
    )
    std_values = normalize_nested_list(
        io_cfg.get("std_values"),
        [[255, 255, 255]],
    )
    input_size_list = io_cfg.get("input_size_list", [[1, 3, 640, 640]])

    onnx_shapes = inspect_onnx_shapes(onnx_model)

    print("=" * 60)
    print("OBB RKNN 转换开始")
    print("=" * 60)
    print(f"ONNX 模型: {onnx_model}")
    print(f"RKNN 输出: {rknn_model}")
    print(f"目标平台: {target_platform}")
    print(f"量化: {do_quantization}")
    print(f"量化 dtype: {quantized_dtype}")
    print(f"optimization_level: {optimization_level}")
    print(f"mean_values: {mean_values}")
    print(f"std_values: {std_values}")
    print(f"input_size_list: {input_size_list}")
    print("ONNX 输入 shape:")
    for name, shape in onnx_shapes["inputs"].items():
        print(f"  {name}: {shape}")
    print("ONNX 输出 shape:")
    for name, shape in onnx_shapes["outputs"].items():
        print(f"  {name}: {shape}")
    print("=" * 60)

    dataset_txt: Path | None = None
    if do_quantization:
        dataset_txt = make_quant_dataset_file(dataset_dir, dataset_size)

    rknn = RKNN(verbose=True)

    ret = rknn.config(
        target_platform=target_platform,
        mean_values=mean_values,
        std_values=std_values,
        quantized_dtype=quantized_dtype,
        optimization_level=optimization_level,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config 失败，ret={ret}")

    ret = rknn.load_onnx(
        model=onnx_model.as_posix(),
        input_size_list=input_size_list,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx 失败，ret={ret}")

    ret = rknn.build(
        do_quantization=do_quantization,
        dataset=dataset_txt.as_posix() if dataset_txt is not None else None,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.build 失败，ret={ret}")

    ret = rknn.export_rknn(rknn_model.as_posix())
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn 失败，ret={ret}")

    rknn.release()

    if not rknn_model.exists():
        raise FileNotFoundError(f"RKNN 转换结束但未找到输出文件: {rknn_model}")

    report = {
        "task": "obb_detection",
        "stage": "convert_rknn",
        "status": "success",
        "target_platform": target_platform,
        "onnx_model": onnx_model.as_posix(),
        "rknn_model": rknn_model.as_posix(),
        "rknn_size_bytes": rknn_model.stat().st_size,
        "do_quantization": do_quantization,
        "quantized_dtype": quantized_dtype,
        "optimization_level": optimization_level,
        "quant_dataset": dataset_dir.as_posix() if do_quantization else None,
        "quant_dataset_size": dataset_size if do_quantization else 0,
        "onnx_shapes": onnx_shapes,
        "expected_runtime_output": {
            "note": "Rockchip RKNN-optimized YOLOv8 OBB output: 3 raw detect heads + 1 angle branch. Decode DFL and angle on CPU.",
            "type": "rockchip_yolov8_obb",
            "shapes": onnx_shapes["outputs"],
        },
        "io_config": {
            "mean_values": mean_values,
            "std_values": std_values,
            "input_size_list": input_size_list,
        },
        "runtime": {
            "perf_debug": bool(runtime_cfg.get("perf_debug", False)),
            "eval_mem": bool(runtime_cfg.get("eval_mem", False)),
        },
        "converted_at": datetime.now().isoformat(),
    }

    save_json(report, perf_report)

    print("✓ OBB RKNN 转换完成")
    print(f"RKNN 模型: {rknn_model}")
    print(f"模型大小: {rknn_model.stat().st_size / 1024 / 1024:.2f} MB")
    print("输出 shape:")
    for name, shape in onnx_shapes["outputs"].items():
        print(f"  {name}: {shape}")
    print(f"转换报告: {perf_report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
