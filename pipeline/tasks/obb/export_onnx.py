from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from pipeline.core.config import load_stage_config, project_path, require_file


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_cmd(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_exported_onnx(checkpoint_path: Path) -> Path:
    """
    Ultralytics 通常会把 ONNX 导出到 best.onnx，
    也就是和 best.pt 同目录、同 basename。
    """
    candidate = checkpoint_path.with_suffix(".onnx")
    if candidate.exists():
        return candidate

    weights_dir = checkpoint_path.parent
    candidates = sorted(weights_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"未找到导出的 ONNX。已检查: {candidate} 和 {weights_dir}/*.onnx"
    )


def export_with_rockchip_cli(
    yolo_exec: Path,
    checkpoint_path: Path,
    imgsz: int,
    opset: int,
    simplify: bool,
    dynamic: bool,
) -> Path:
    """
    使用瑞芯微修改过的 Ultralytics CLI 导出 ONNX。

    说明：
    - 训练仍可用标准 Ultralytics。
    - 用于 RKNN 的 ONNX 建议用 airockchip/ultralytics_yolov8 导出。
    - OBB 任务这里显式传 task=obb，避免 CLI 自动判断失败。
    """
    require_file(yolo_exec, "请确认 export.yolo_exec 指向 pt2onnx 环境中的 yolo 可执行文件")
    require_file(checkpoint_path, "请先运行 dvc repro train 生成 OBB best.pt")

    cmd = [
        str(yolo_exec),
        "export",
        f"model={checkpoint_path.as_posix()}",
        "format=rknn",
        "task=obb",
        f"imgsz={imgsz}",
        f"opset={opset}",
        f"simplify={str(simplify).lower()}",
        f"dynamic={str(dynamic).lower()}",
    ]

    run_cmd(cmd)
    return find_exported_onnx(checkpoint_path)


def export_with_standard_ultralytics(
    checkpoint_path: Path,
    imgsz: int,
    opset: int,
    simplify: bool,
    dynamic: bool,
) -> Path:
    """
    使用当前 Python 环境中的标准 Ultralytics 导出 ONNX。

    仅建议用于快速调试。如果后续要 RKNN 转换，优先用 rockchip mode。
    """
    require_file(checkpoint_path, "请先运行 dvc repro train 生成 OBB best.pt")

    model = YOLO(str(checkpoint_path))
    exported = model.export(
        format="onnx",
        task="obb",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
    )

    exported_path = Path(exported) if exported else find_exported_onnx(checkpoint_path)

    if not exported_path.exists():
        exported_path = find_exported_onnx(checkpoint_path)

    return exported_path


def main() -> None:
    cfg = load_stage_config("export")
    export_cfg = cfg.get("export", {})

    checkpoint_path = project_path(
        export_cfg.get("checkpoint_path", "models/checkpoints_obb/best.pt")
    )
    output_path = project_path(
        export_cfg.get("output_path", "models/export_obb/model.onnx")
    )

    imgsz = int(export_cfg.get("imgsz", 640))
    opset = int(export_cfg.get("opset", 12))
    simplify = bool(export_cfg.get("simplify", True))
    dynamic = bool(export_cfg.get("dynamic", False))
    mode = str(export_cfg.get("mode", "rockchip")).lower()
    yolo_exec = project_path(
        export_cfg.get("yolo_exec", "/home/pc/anaconda3/envs/pt2onnx/bin/yolo")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OBB ONNX 导出开始")
    print("=" * 60)
    print(f"导出模式: {mode}")
    print(f"模型权重: {checkpoint_path}")
    print(f"输出 ONNX: {output_path}")
    print(f"imgsz: {imgsz}")
    print(f"opset: {opset}")
    print(f"simplify: {simplify}")
    print(f"dynamic: {dynamic}")
    if mode == "rockchip":
        print(f"Rockchip yolo_exec: {yolo_exec}")
    print("=" * 60)

    if mode in {"rockchip", "rknn", "rk", "airockchip"}:
        exported_path = export_with_rockchip_cli(
            yolo_exec=yolo_exec,
            checkpoint_path=checkpoint_path,
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
        )
        actual_mode = "rockchip"
    elif mode in {"standard", "ultralytics", "default"}:
        exported_path = export_with_standard_ultralytics(
            checkpoint_path=checkpoint_path,
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
        )
        actual_mode = "standard"
    else:
        raise ValueError(
            f"不支持的 OBB ONNX 导出模式: {mode}，"
            f"应为 rockchip 或 standard"
        )

    if exported_path.resolve() != output_path.resolve():
        shutil.copy2(exported_path, output_path)

    if not output_path.exists():
        raise FileNotFoundError(f"ONNX 导出失败，未找到输出文件: {output_path}")

    report = {
        "task": "obb_detection",
        "stage": "export_onnx",
        "status": "success",
        "mode": actual_mode,
        "checkpoint_path": checkpoint_path.as_posix(),
        "exported_path": exported_path.as_posix(),
        "output_path": output_path.as_posix(),
        "output_size_bytes": output_path.stat().st_size,
        "params": {
            "imgsz": imgsz,
            "opset": opset,
            "simplify": simplify,
            "dynamic": dynamic,
            "yolo_exec": yolo_exec.as_posix() if actual_mode == "rockchip" else None,
        },
        "exported_at": datetime.now().isoformat(),
    }

    report_path = output_path.parent / "export_report.json"
    save_json(report, report_path)

    print("✓ OBB ONNX 导出完成")
    print(f"临时导出文件: {exported_path}")
    print(f"标准输出文件: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"导出报告: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
