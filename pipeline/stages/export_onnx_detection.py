from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def export_with_ultralytics(
    checkpoint_path: Path,
    output_path: Path,
    opset: int,
    imgsz: int,
    simplify: bool,
    dynamic: bool,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint_path))

    # Ultralytics 会返回导出后的文件路径
    exported = model.export(
        format="onnx",
        opset=opset,
        imgsz=imgsz,
        simplify=simplify,
        dynamic=dynamic,
    )

    exported_path = Path(str(exported))
    if not exported_path.exists():
        raise FileNotFoundError(f"Ultralytics 导出结果不存在: {exported_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if exported_path.resolve() != output_path.resolve():
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(exported_path), str(output_path))

    return {
        "export_backend": "ultralytics",
        "raw_exported_path": str(exported_path),
        "final_output_path": str(output_path),
    }


def export_with_external_script(
    checkpoint_path: Path,
    output_path: Path,
    opset: int,
    imgsz: int,
    python_exec: str,
    script_path: Path,
) -> dict:
    if not script_path.exists():
        raise FileNotFoundError(f"外部导出脚本不存在: {script_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exec,
        str(script_path),
        "--weights",
        str(checkpoint_path),
        "--output",
        str(output_path),
        "--imgsz",
        str(imgsz),
        "--opset",
        str(opset),
    ]

    print("执行外部导出脚本:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if not output_path.exists():
        raise FileNotFoundError(f"外部导出完成，但未找到输出文件: {output_path}")

    return {
        "export_backend": "external",
        "script_path": str(script_path),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
        "final_output_path": str(output_path),
    }
    
def export_with_rockchip_yolo(
    checkpoint_path: Path,
    output_path: Path,
    imgsz: int,
    yolo_exec: str,
    project_dir: Path | None = None,
) -> dict:
    """
    使用 airockchip/ultralytics_yolov8 环境中的 yolo CLI 导出 RKNN 友好的 ONNX。

    等价于手动执行：
        yolo export model=best.pt format=rknn

    注意：
    - airockchip 的 format=rknn 实际会先导出 RKNN 友好的 ONNX
    - 这里最终只接收 .onnx，并统一移动到 output_path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not Path(yolo_exec).exists():
        raise FileNotFoundError(f"Rockchip yolo 命令不存在: {yolo_exec}")

    before_onnx_files = set(Path(".").rglob("*.onnx"))

    cmd = [
        yolo_exec,
        "export",
        f"model={checkpoint_path}",
        "format=rknn",
        f"imgsz={imgsz}",
    ]

    print("执行 Rockchip YOLO 导出:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(project_dir) if project_dir else None,
    )

    # 优先按 checkpoint 同目录推断
    candidate_paths = [
        checkpoint_path.with_suffix(".onnx"),
        checkpoint_path.parent / f"{checkpoint_path.stem}.onnx",
    ]

    # 再搜索新生成的 onnx
    after_onnx_files = set(Path(".").rglob("*.onnx"))
    new_onnx_files = list(after_onnx_files - before_onnx_files)

    candidate_paths.extend(new_onnx_files)

    exported_path = None
    for p in candidate_paths:
        if p.exists():
            exported_path = p
            break

    if exported_path is None:
        stdout_tail = result.stdout[-3000:]
        stderr_tail = result.stderr[-3000:]
        raise FileNotFoundError(
            "Rockchip YOLO 导出完成，但未找到 ONNX 文件。\n"
            f"stdout_tail:\n{stdout_tail}\n"
            f"stderr_tail:\n{stderr_tail}"
        )

    if output_path.exists():
        output_path.unlink()

    shutil.move(str(exported_path), str(output_path))

    return {
        "export_backend": "rockchip_yolo",
        "yolo_exec": str(yolo_exec),
        "raw_exported_path": str(exported_path),
        "final_output_path": str(output_path),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def validate_onnx(output_path: Path) -> dict:
    result = {
        "onnx_installed": False,
        "onnx_valid": False,
        "check_error": None,
    }

    try:
        import onnx  # type: ignore

        result["onnx_installed"] = True
        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)
        result["onnx_valid"] = True
    except ImportError:
        result["check_error"] = "onnx package not installed"
    except Exception as e:
        result["check_error"] = str(e)

    return result


def main() -> None:
    cfg_path = Path("pipeline/configs/detection_export.generated.yaml")
    if not cfg_path.exists():
        cfg_path = Path("pipeline/configs/detection_export.yaml")
    cfg = load_yaml(cfg_path)

    print(f"导出配置文件: {cfg_path}")

    export_cfg = cfg["export"]
    ext_cfg = cfg.get("external_export", {})

    checkpoint_path = Path(export_cfg.get("checkpoint_path", "models/checkpoints_detection/best.pt"))
    output_path = Path(export_cfg.get("output_path", "models/export_detection/model.onnx"))
    mode = str(export_cfg.get("mode", "ultralytics")).strip().lower()
    opset = int(export_cfg.get("opset", 12))
    imgsz = int(export_cfg.get("imgsz", 640))
    simplify = bool(export_cfg.get("simplify", True))
    dynamic = bool(export_cfg.get("dynamic", False))

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检测模型 checkpoint 不存在: {checkpoint_path}")

    print("=" * 60)
    print("YOLOv8 检测 ONNX 导出开始")
    print("=" * 60)
    print(f"导出模式: {mode}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output: {output_path}")
    print(f"imgsz: {imgsz}")
    print(f"opset: {opset}")
    print(f"simplify: {simplify}")
    print(f"dynamic: {dynamic}")
    print("=" * 60)

    safe_unlink(output_path)

    if mode == "ultralytics":
        export_info = export_with_ultralytics(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            opset=opset,
            imgsz=imgsz,
            simplify=simplify,
            dynamic=dynamic,
        )
    elif mode == "external":
        export_info = export_with_external_script(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            opset=opset,
            imgsz=imgsz,
            python_exec=str(ext_cfg.get("python_exec", "python")),
            script_path=Path(ext_cfg.get("script_path", "tools/export_yolov8_rknn_onnx.py")),
        )
    elif mode in ["rockchip", "rknn", "airockchip"]:
        export_info = export_with_rockchip_yolo(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            imgsz=imgsz,
            yolo_exec=str(ext_cfg.get(
                "yolo_exec",
                "/home/pc/anaconda3/envs/pt2onnx/bin/yolo"
            )),
            project_dir=Path(ext_cfg["project_dir"]) if ext_cfg.get("project_dir") else None,
        )
    else:
        raise ValueError(f"不支持的导出模式: {mode}")

    if not output_path.exists():
        raise FileNotFoundError(f"导出完成，但未找到 ONNX 文件: {output_path}")

    size_mb = output_path.stat().st_size / 1024 / 1024
    onnx_check = validate_onnx(output_path)

    result = {
        "task": "detection",
        "mode": mode,
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "size_mb": round(size_mb, 4),
        "imgsz": imgsz,
        "opset": opset,
        "simplify": simplify,
        "dynamic": dynamic,
        "onnx_check": onnx_check,
        "export_info": export_info,
        "exported_at": datetime.now().isoformat(),
    }

    save_json(result, Path("models/export_detection/export_result.json"))

    print("✓ YOLOv8 检测 ONNX 导出完成")
    print(f"ONNX 文件: {output_path}")
    print(f"文件大小: {size_mb:.2f} MB")
    print(f"ONNX 校验: {onnx_check}")
    print("=" * 60)


if __name__ == "__main__":
    main()
