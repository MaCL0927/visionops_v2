"""
Stage 5: 分类 RKNN 模型转换

输入：
models/export/model.onnx

输出：
models/export/
├── model.rknn
└── rknn_perf_report.json

说明：
- 支持自动切换到 rknn311 环境执行
- 支持 INT8 量化
- 校准图像来自 data/processed/val/
- 分类模型输入为 RGB + mean/std，由 rknn.config 负责归一化
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def maybe_reexec_in_rknn_env(cfg: dict) -> None:
    """
    如果当前进程不是 RKNN 内部执行，则自动切换到配置指定的 python_exec 重新执行本脚本。

    通过环境变量 VISIONOPS_RKNN_INTERNAL=1 防止无限递归。
    """
    if os.environ.get("VISIONOPS_RKNN_INTERNAL") == "1":
        return

    python_exec = cfg.get("python_exec")
    if not python_exec:
        raise RuntimeError(
            "pipeline/configs/rknn.yaml 缺少 python_exec，无法自动切换到 RKNN 环境。\n"
            "请添加例如：python_exec: /home/pc/anaconda3/envs/rknn311/bin/python"
        )

    python_path = Path(python_exec)
    if not python_path.exists():
        raise FileNotFoundError(f"RKNN Python 不存在: {python_exec}")

    env = os.environ.copy()
    env["VISIONOPS_RKNN_INTERNAL"] = "1"

    cmd = [str(python_path), __file__]

    print("=" * 60)
    print(">>> 当前不在 RKNN 执行环境，自动切换解释器")
    print(f">>> python_exec: {python_exec}")
    print(f">>> cmd: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, env=env)
    raise SystemExit(result.returncode)


def get_input_hw(input_size_list: List[List[int]]) -> Tuple[int, int]:
    """
    从 input_size_list 中获取 H, W。
    期望格式：[[1, 3, H, W]]
    """
    if not input_size_list or not isinstance(input_size_list[0], list):
        raise ValueError(f"io_config.input_size_list 格式错误: {input_size_list}")

    shape = input_size_list[0]
    if len(shape) != 4:
        raise ValueError(f"分类模型 input_size_list 应为 [[1, 3, H, W]]，当前: {input_size_list}")

    if int(shape[0]) != 1 or int(shape[1]) != 3:
        raise ValueError(f"分类模型建议输入为 [1, 3, H, W]，当前: {shape}")

    h, w = int(shape[2]), int(shape[3])
    return h, w


def collect_image_files(dataset_path: str) -> List[str]:
    image_files: List[str] = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_files.extend(glob(os.path.join(dataset_path, "**", ext), recursive=True))

    image_files = sorted(set(image_files))
    return image_files


def load_calibration_dataset(
    dataset_path: str,
    size: int = 100,
    img_size: List[int] | None = None,
) -> List[Tuple[str, np.ndarray]]:
    """
    加载分类量化校准图像。

    重要：
    - 这里输出 RGB uint8 图像
    - 不在这里做 mean/std normalize
    - mean/std 交给 rknn.config(mean_values, std_values) 处理
    - 这和训练里的 torchvision Normalize 等价
    """
    import cv2

    img_size = img_size or [224, 224]
    h, w = int(img_size[0]), int(img_size[1])

    image_files = collect_image_files(dataset_path)

    if not image_files:
        raise FileNotFoundError(
            f"校准数据集路径中没有找到图像: {dataset_path}\n"
            "请确认 pipeline/configs/rknn.yaml 中 quantization.dataset 指向 data/processed/val/"
        )

    np.random.shuffle(image_files)
    selected = image_files[: min(size, len(image_files))]

    images: List[Tuple[str, np.ndarray]] = []

    for img_path in selected:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        images.append((img_path, img_rgb))

    if not images:
        raise RuntimeError("校准图像读取失败，无法继续 RKNN 量化")

    print(f"加载了 {len(images)} 张分类校准图像")
    return images


def write_calibration_list(
    calibration_images: List[Tuple[str, np.ndarray]],
) -> str:
    """
    将校准图像写入临时目录，并生成 RKNN dataset txt。

    RKNN 的 dataset txt 需要是一行一个图片路径。
    """
    import cv2

    tmp_dir = Path(tempfile.mkdtemp(prefix="visionops_cls_rknn_calib_"))
    dataset_list_path = tmp_dir / "dataset.txt"

    with dataset_list_path.open("w", encoding="utf-8") as f:
        for i, (_, img_rgb) in enumerate(calibration_images):
            tmp_img_path = tmp_dir / f"calib_{i:04d}.jpg"

            # cv2.imwrite 需要 BGR，所以从 RGB 转回 BGR 保存
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(tmp_img_path), img_bgr)

            if ok:
                f.write(str(tmp_img_path) + "\n")

    return str(dataset_list_path)


def simulate_conversion(cfg: dict) -> dict:
    """
    未安装 rknn-toolkit2 时，生成占位结果，便于先打通 DVC 流程。

    正式转换时不应该走到这里。
    如果你看到 status=simulated，说明没有进入 rknn311 环境或环境缺 rknn-toolkit2。
    """
    print("警告: rknn-toolkit2 未安装，执行模拟 RKNN 转换（classification）")

    output_cfg = cfg["output"]
    rknn_path = Path(output_cfg["rknn_model"])
    report_path = Path(output_cfg["perf_report"])

    rknn_path.parent.mkdir(parents=True, exist_ok=True)

    with rknn_path.open("wb") as f:
        f.write(b"RKNN_CLASSIFICATION_SIMULATED_MODEL")

    report = {
        "task": "classification",
        "status": "simulated",
        "platform": cfg["target_platform"],
        "quantization": cfg.get("quantization", {}).get("do_quantization", True),
        "model_size_mb": round(rknn_path.stat().st_size / 1024 / 1024, 6),
        "onnx_path": output_cfg.get("onnx_model", "models/export/model.onnx"),
        "rknn_path": str(rknn_path),
        "converted_at": datetime.now().isoformat(),
        "note": "rknn-toolkit2 未安装，仅生成占位符文件",
    }

    save_json(report, report_path)
    return report


def convert_to_rknn(cfg: dict) -> dict:
    try:
        from rknn.api import RKNN
    except ImportError as e:
        print(f"警告: 当前解释器中未安装 rknn-toolkit2: {e}")
        return simulate_conversion(cfg)

    output_cfg = cfg["output"]
    quant_cfg = cfg.get("quantization", {})
    io_cfg = cfg.get("io_config", {})

    onnx_path = output_cfg.get("onnx_model", "models/export/model.onnx")
    rknn_path = output_cfg["rknn_model"]
    report_path = output_cfg["perf_report"]

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")

    Path(rknn_path).parent.mkdir(parents=True, exist_ok=True)

    input_size_list = io_cfg.get("input_size_list", [[1, 3, 224, 224]])
    h, w = get_input_hw(input_size_list)

    mean_values = io_cfg.get("mean_values", [[123.675, 116.28, 103.53]])
    std_values = io_cfg.get("std_values", [[58.395, 57.12, 57.375]])

    do_quantization = bool(quant_cfg.get("do_quantization", True))

    print("=" * 60)
    print(">>> 初始化 RKNN")
    print("=" * 60)
    print(f"ONNX: {onnx_path}")
    print(f"RKNN: {rknn_path}")
    print(f"input_size_list: {input_size_list}")
    print(f"mean_values: {mean_values}")
    print(f"std_values: {std_values}")
    print(f"do_quantization: {do_quantization}")

    rknn = RKNN(verbose=True)

    print("\n>>> 配置 RKNN 参数...")
    ret = rknn.config(
        mean_values=mean_values,
        std_values=std_values,
        target_platform=cfg["target_platform"],
        quantized_dtype=quant_cfg.get("quantized_dtype", "asymmetric_quantized-8"),
        optimization_level=int(cfg.get("optimization_level", 3)),
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config() 失败: {ret}")

    print(f"\n>>> 加载 ONNX: {onnx_path}")
    ret = rknn.load_onnx(
        model=onnx_path,
        input_size_list=input_size_list,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx() 失败: {ret}")

    dataset_list_path = None

    if do_quantization:
        print("\n>>> 准备分类量化校准数据...")
        calibration_images = load_calibration_dataset(
            dataset_path=quant_cfg.get("dataset", "data/processed/val/"),
            size=int(quant_cfg.get("dataset_size", 100)),
            img_size=[h, w],
        )

        dataset_list_path = write_calibration_list(calibration_images)

        print(f">>> 校准列表: {dataset_list_path}")
        print("\n>>> 构建 RKNN（INT8 量化）...")
        ret = rknn.build(
            do_quantization=True,
            dataset=dataset_list_path,
            rknn_batch_size=1,
        )
    else:
        print("\n>>> 构建 RKNN（不量化）...")
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        raise RuntimeError(f"rknn.build() 失败: {ret}")

    print(f"\n>>> 导出 RKNN: {rknn_path}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn() 失败: {ret}")

    output_shapes = []
    if cfg.get("check_output_shapes", True):
        print("\n>>> 检查 RKNN 模拟器输出 shape...")
        try:
            ret = rknn.init_runtime(target=None)
            if ret == 0:
                dummy_input = np.zeros((1, h, w, 3), dtype=np.uint8)
                outputs = rknn.inference(inputs=[dummy_input])

                for i, out in enumerate(outputs):
                    shape = tuple(out.shape)
                    dtype = str(out.dtype)
                    output_shapes.append(
                        {
                            "index": i,
                            "shape": list(shape),
                            "dtype": dtype,
                        }
                    )
                    print(f"[RKNN_OUTPUT] output[{i}] shape={shape}, dtype={dtype}")
            else:
                print(f"警告: init_runtime(target=None) 失败，跳过输出 shape 检查: {ret}")
        except Exception as e:
            print(f"警告: RKNN 输出 shape 检查失败，跳过: {type(e).__name__}: {e}")

    perf_report = {
        "status": "converted",
        "platform": cfg["target_platform"],
        "quantization": do_quantization,
    }

    if cfg.get("perf_debug", False):
        print("\n>>> 执行 RKNN 模拟器性能分析...")
        try:
            ret = rknn.init_runtime(target=None)
            if ret == 0:
                dummy_input = np.random.randint(
                    0,
                    255,
                    size=(1, h, w, 3),
                    dtype=np.uint8,
                )
                perf_results = rknn.eval_perf(inputs=[dummy_input], is_print=True)
                perf_report["perf_details"] = str(perf_results)
            else:
                perf_report["perf_error"] = f"init_runtime failed: {ret}"
        except Exception as e:
            perf_report["perf_error"] = f"{type(e).__name__}: {e}"

    rknn.release()

    model_size = os.path.getsize(rknn_path) / 1024 / 1024

    report = {
        "task": "classification",
        "status": "success",
        "platform": cfg["target_platform"],
        "quantization": do_quantization,
        "model_size_mb": round(model_size, 4),
        "onnx_path": onnx_path,
        "rknn_path": rknn_path,
        "report_path": report_path,
        "input_size_list": input_size_list,
        "mean_values": mean_values,
        "std_values": std_values,
        "dataset": quant_cfg.get("dataset", "data/processed/val/"),
        "dataset_size": quant_cfg.get("dataset_size", 100),
        "output_shapes": output_shapes,
        "perf": perf_report,
        "converted_at": datetime.now().isoformat(),
        "notes": {
            "expected_output": "classification logits, shape should usually be [1, num_classes]",
            "postprocess": "softmax(logits) -> argmax",
            "input_layout_for_runtime_check": "NHWC uint8 image is used for RKNN simulator inference",
        },
    }

    return report


def main() -> None:
    cfg_path = "pipeline/configs/rknn.yaml"
    cfg = load_config(cfg_path)

    maybe_reexec_in_rknn_env(cfg)

    output_cfg = cfg["output"]
    onnx_path = output_cfg.get("onnx_model", "models/export/model.onnx")
    report_path = Path(output_cfg["perf_report"])

    print("=" * 60)
    print(f"分类 RKNN 模型转换 | 目标平台: {cfg['target_platform']}")
    print("=" * 60)
    print(f"RKNN 配置文件: {cfg_path}")

    if not os.path.exists(onnx_path):
        print(f"错误: ONNX 模型不存在: {onnx_path}")
        print("请先运行：python pipeline/stages/export_onnx.py")
        sys.exit(1)

    report = convert_to_rknn(cfg)
    save_json(report, report_path)

    print("\n" + "=" * 60)
    print("✓ 分类 RKNN 转换完成")
    print("=" * 60)
    print(f"模型路径: {report['rknn_path']}")
    print(f"模型大小: {report.get('model_size_mb', 'N/A')} MB")
    print(f"量化: {report.get('quantization', False)}")
    print(f"状态: {report['status']}")
    print(f"报告: {report_path}")

    if report.get("output_shapes"):
        print(f"RKNN 输出 shape: {report['output_shapes']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
