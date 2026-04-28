from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.core.io import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "pipeline" / "configs"
TASK_CONFIG = CONFIG_DIR / "task.yaml"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: str | Path, hint: str | None = None) -> Path:
    path = project_path(path)
    if not path.exists():
        msg = f"必要文件不存在: {path}"
        if hint:
            msg += f"\n提示: {hint}"
        raise FileNotFoundError(msg)
    return path


def load_task_config() -> dict[str, Any]:
    return load_yaml(require_file(TASK_CONFIG, "请先创建 pipeline/configs/task.yaml"))


def normalize_task_type(task_type: str) -> str:
    t = str(task_type or "").strip().lower()
    if t in {"detect", "detection", "yolo_detection", "object_detection"}:
        return "detection"
    if t in {"cls", "classify", "classification", "image_classification"}:
        return "classification"
    raise ValueError(f"不支持的 task.type: {task_type!r}，应为 detection 或 classification")


def get_task_type(task_cfg: dict[str, Any] | None = None) -> str:
    cfg = task_cfg or load_task_config()
    task = cfg.get("task", {})
    return normalize_task_type(task.get("type") or task.get("name") or cfg.get("task_type"))
