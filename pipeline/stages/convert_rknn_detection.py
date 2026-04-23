from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_calibration_dataset(dataset_path: str, size: int = 100, img_size: List[int] | None = None):
    """
    加载量化校准图像。
    - dataset_path 建议传 images/val 目录
    - 返回 RGB uint8 图像列表
    """
    import cv2
    from glob import glob

    img_size = img_size or [640, 640]

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_files.extend(glob(os.path.join(dataset_path, "**", ext), recursive=True))

    if not image_files:
        raise FileNotFoundError(f"校准数据集路径中没有找到图像: {dataset_path}")

    np.random.shuffle(image_files)
    selected = image_files[: min(size, len(image_files))]

    images = []
    for img_path in selected:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size[1], img_size[0]))
        images.append((img_path, img))

    if not images:
        raise RuntimeError("校准图像读取失败，无法继续 RKNN 量化")

    print(f"加载了 {len(images)} 张校准图像")
    return images


def simulate_conversion(cfg: dict) -> dict:
    """
    未安装 rknn-toolkit2 时，生成占位结果，便于先打通 detection 分支。
    """
    print("警告: rknn-toolkit2 未安装，执行模拟 RKNN 转换（detection）")

    output_cfg = cfg["output"]
    rknn_path = Path(output_cfg["rknn_model"])
    report_path = Path(output_cfg["perf_report"])

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    with rknn_path.open("wb") as f:
        f.write(b"RKNN_DETECTION_SIMULATED_MODEL")

    report = {
        "task": "detection",
        "status": "simulated",
        "platform": cfg["target_platform"],
        "quantization": cfg.get("quantization", {}).get("do_quantization", True),
        "model_size_mb": round(rknn_path.stat().st_size / 1024 / 1024, 6),
        "onnx_path": cfg["output"]["onnx_model"],
        "rknn_path": str(rknn_path),
        "converted_at": datetime.now().isoformat(),
        "note": "rknn-toolkit2 未安装，仅生成占位符文件",
    }
    save_json(report, report_path)
    return report


def convert_to_rknn(cfg: dict) -> dict:
    try:
        from rknn.api import RKNN
    except ImportError:
        return simulate_conversion(cfg)

    output_cfg = cfg["output"]
    quant_cfg = cfg.get("quantization", {})
    io_cfg = cfg.get("io_config", {})

    onnx_path = output_cfg["onnx_model"]
    rknn_path = output_cfg["rknn_model"]

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")

    Path(rknn_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(">>> 初始化 RKNN")
    print("=" * 60)
    rknn = RKNN(verbose=True)

    mean_values = io_cfg.get("mean_values", [[0, 0, 0]])
    std_values = io_cfg.get("std_values", [[255, 255, 255]])
    input_size_list = io_cfg.get("input_size_list", [[1, 3, 640, 640]])

    print("\n>>> 配置 RKNN 参数...")
    
    optimization_level = cfg.get("optimization_level", 3)

    ret = rknn.config(
        mean_values=mean_values,
        std_values=std_values,
        target_platform=cfg["target_platform"],
        quantized_dtype=quant_cfg.get("quantized_dtype", "asymmetric_quantized-8"),
        optimization_level=optimization_level,
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

    do_quantization = quant_cfg.get("do_quantization", True)

    if do_quantization:
        print("\n>>> 准备量化校准数据...")
        calibration_images = load_calibration_dataset(
            dataset_path=quant_cfg.get("dataset", "data/processed_detection/images/val"),
            size=quant_cfg.get("dataset_size", 100),
            img_size=input_size_list[0][2:] if len(input_size_list[0]) == 4 else [640, 640],
        )

        import tempfile
        import cv2

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            dataset_list_path = f.name
            for i, (src_path, img) in enumerate(calibration_images):
                tmp_path = f"/tmp/rknn_det_calib_{i:04d}.jpg"
                cv2.imwrite(tmp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                f.write(tmp_path + "\n")

        print("\n>>> 构建 RKNN（含 INT8 量化）...")
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

    perf_report = {
        "status": "converted",
        "platform": cfg["target_platform"],
        "quantization": do_quantization,
    }

    if cfg.get("perf_debug", False):
        print("\n>>> 执行 RKNN 模拟器性能分析...")
        ret = rknn.init_runtime(target=None)
        if ret == 0:
            dummy_input = np.random.randint(
                0,
                255,
                size=[1] + input_size_list[0][1:],
                dtype=np.uint8,
            )
            perf_results = rknn.eval_perf(inputs=[dummy_input], is_print=True)
            perf_report["perf_details"] = str(perf_results)

    rknn.release()

    model_size = os.path.getsize(rknn_path) / 1024 / 1024
    report = {
        "task": "detection",
        "status": "success",
        "platform": cfg["target_platform"],
        "quantization": do_quantization,
        "model_size_mb": round(model_size, 4),
        "onnx_path": onnx_path,
        "rknn_path": rknn_path,
        "converted_at": datetime.now().isoformat(),
        "perf": perf_report,
    }
    return report


def main() -> None:
    cfg_path = "pipeline/configs/detection_rknn.generated.yaml"
    if not Path(cfg_path).exists():
        cfg_path = "pipeline/configs/detection_rknn.yaml"
    cfg = load_config(cfg_path)

    print(f"RKNN配置文件: {cfg_path}")

    onnx_path = cfg["output"]["onnx_model"]
    report_path = Path(cfg["output"]["perf_report"])

    print("=" * 60)
    print(f"Detection RKNN 模型转换 | 目标平台: {cfg['target_platform']}")
    print("=" * 60)

    if not os.path.exists(onnx_path):
        print(f"错误: ONNX 模型不存在: {onnx_path}")
        print("请先运行 detection ONNX 导出阶段")
        sys.exit(1)

    report = convert_to_rknn(cfg)
    save_json(report, report_path)

    print("\n" + "=" * 60)
    print("✓ Detection RKNN 转换完成!")
    print(f"模型路径: {report['rknn_path']}")
    print(f"模型大小: {report.get('model_size_mb', 'N/A')} MB")
    print(f"量化: {report.get('quantization', False)}")
    print(f"状态: {report['status']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
