#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
import uuid
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from .label_io import list_images, parse_yolo_label, save_yolo_label

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_CONTEXT_PATH = DATA_DIR / "model_context" / "manifest.json"

QUICK_TRAIN_MIN_IMAGES = int(os.environ.get("VISIONOPS_QUICK_TRAIN_MIN_IMAGES", "5"))
QUICK_TRAIN_EPOCHS = int(os.environ.get("VISIONOPS_QUICK_TRAIN_EPOCHS", "50"))
QUICK_TRAIN_IMGSZ = int(os.environ.get("VISIONOPS_QUICK_TRAIN_IMGSZ", "640"))
QUICK_TRAIN_BATCH = int(os.environ.get("VISIONOPS_QUICK_TRAIN_BATCH", "2"))
QUICK_DET_MODEL = os.environ.get("VISIONOPS_QUICK_DET_MODEL", "models/pretrained/yolov8n.pt")
QUICK_OBB_MODEL = os.environ.get("VISIONOPS_QUICK_OBB_MODEL", "models/pretrained/yolov8n-obb.pt")
QUICK_SEG_MODEL = os.environ.get("VISIONOPS_QUICK_SEG_MODEL", "models/pretrained/yolov8n-seg.pt")
QUICK_YOLO_CMD = os.environ.get("VISIONOPS_QUICK_YOLO_CMD", "yolo")
AUTO_LABEL_CONF = float(os.environ.get("VISIONOPS_QUICK_AUTO_CONF", "0.25"))
QUICK_TRAIN_MIN_PER_CLASS = int(os.environ.get("VISIONOPS_QUICK_TRAIN_MIN_PER_CLASS", "3"))

# ROI classification 数据制作：检测模型裁剪目标整体框，再由人工给 ROI 分配分类标签。
ROI_CLS_RAW_DIR = Path(os.environ.get("VISIONOPS_ROI_CLS_RAW_DIR", str(DATA_DIR / "raw_classification")))
ROI_CLS_SESSIONS_DIR = Path(os.environ.get("VISIONOPS_ROI_CLS_SESSIONS_DIR", str(DATA_DIR / "roi_classification_sessions")))
ROI_CLS_DEFAULT_DET_MODEL = os.environ.get("VISIONOPS_ROI_CLS_DEFAULT_DET_MODEL", "models/checkpoints_detection/best.pt")
ROI_CLS_DEFAULT_CONF = float(os.environ.get("VISIONOPS_ROI_CLS_DEFAULT_CONF", "0.35"))
ROI_CLS_DEFAULT_PADDING = float(os.environ.get("VISIONOPS_ROI_CLS_DEFAULT_PADDING", "0.05"))
ROI_CLS_MAX_CANDIDATES = int(os.environ.get("VISIONOPS_ROI_CLS_MAX_CANDIDATES", "0"))

router = APIRouter()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def get_current_batch_dir() -> Path:
    manifest = read_json(MODEL_CONTEXT_PATH, default={}) or {}
    source_manifest = manifest.get("source_manifest")
    if source_manifest:
        p = Path(source_manifest)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.parent

    package_name = manifest.get("package_name") or manifest.get("batch_id")
    if package_name:
        return PROJECT_ROOT / "data" / "raw_collected" / package_name

    raise RuntimeError("未找到当前 batch，请先处理上传数据集。")


def get_batch_paths() -> tuple[Path, Path, Path, Path, Path]:
    batch_dir = get_current_batch_dir()
    images_dir = batch_dir / "all_images"
    labels_dir = batch_dir / "labels"
    labels_auto_dir = batch_dir / "labels_auto"
    classes_path = batch_dir / "annotation_classes.json"

    if not images_dir.exists():
        raise FileNotFoundError(f"未找到图片目录: {images_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_auto_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir, images_dir, labels_dir, labels_auto_dir, classes_path


def get_image_size(path: Path) -> tuple[int, int]:
    if Image is None:
        raise RuntimeError("Pillow 未安装，无法读取图片尺寸。请安装 pillow。")
    with Image.open(path) as img:
        return img.size


def load_annotation_classes(classes_path: Path) -> list[str]:
    data = read_json(classes_path, default={}) or {}
    names = data.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    return []


def save_annotation_classes(classes_path: Path, names: list[str]) -> None:
    cleaned: list[str] = []
    for name in names:
        name = str(name).strip()
        if name and name not in cleaned:
            cleaned.append(name)

    write_json(classes_path, {
        "names": cleaned,
        "num_classes": len(cleaned),
        "note": "类别顺序即 YOLO 标注中的类别下标。第一个新增类别下标为 0。",
    })


def quick_state_path(batch_dir: Path) -> Path:
    return batch_dir / "quick_train" / "quick_state.json"


def load_quick_state(batch_dir: Path) -> dict[str, Any]:
    state = read_json(quick_state_path(batch_dir), default={}) or {}
    state.setdefault("quick_train", {})
    state.setdefault("last_auto_label", {})
    return state


def save_quick_state(batch_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_json(quick_state_path(batch_dir), state)


def normalize_annotation_task(task_type: str | None) -> str:
    task = str(task_type or "detection").strip().lower()
    if task in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg", "yolov8_seg"}:
        return "segmentation"
    if task in {"obb", "obb_detection", "oriented_detection", "rotated_detection"}:
        return "obb"
    return "detection"


def task_state_path(batch_dir: Path) -> Path:
    return batch_dir / "annotation_task.json"


def load_annotation_task(batch_dir: Path) -> str:
    data = read_json(task_state_path(batch_dir), default={}) or {}
    return normalize_annotation_task(data.get("task_type") or data.get("task") or "detection")


def save_annotation_task(batch_dir: Path, task_type: str) -> str:
    task = normalize_annotation_task(task_type)
    write_json(task_state_path(batch_dir), {
        "task_type": task,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "当前 batch 的标注任务类型。用于区分 OBB 9列标签与 segmentation 4点多边形标签。",
    })
    return task


def task_to_yolo_subcommand(task_type: str) -> str:
    task = normalize_annotation_task(task_type)
    if task == "obb":
        return "obb"
    if task == "segmentation":
        return "segment"
    return "detect"


def quick_model_for_task(task_type: str) -> str:
    task = normalize_annotation_task(task_type)
    if task == "obb":
        return QUICK_OBB_MODEL
    if task == "segmentation":
        return QUICK_SEG_MODEL
    return QUICK_DET_MODEL


def non_empty_label(path: Path) -> bool:
    return path.exists() and bool(path.read_text(encoding="utf-8", errors="ignore").strip())


def count_manual_labels(labels_dir: Path) -> int:
    return sum(1 for p in labels_dir.glob("*.txt") if non_empty_label(p))


def count_auto_labels(labels_auto_dir: Path) -> int:
    return sum(1 for p in labels_auto_dir.glob("*.txt") if non_empty_label(p))


def collect_class_image_counts(labels_dir: Path, num_classes: int) -> dict[int, int]:
    """Count how many manually labeled images contain each class id.

    The count is image-level rather than box-level, because quick learning needs
    at least a few different images for every class.
    """
    counts = {i: 0 for i in range(num_classes)}
    for label_path in labels_dir.glob("*.txt"):
        text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        class_ids: set[int] = set()
        for line in text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except Exception:
                continue
            if 0 <= class_id < num_classes:
                class_ids.add(class_id)
        for class_id in class_ids:
            counts[class_id] += 1
    return counts


def validate_quick_train_class_coverage(labels_dir: Path, classes: list[str]) -> dict[int, int]:
    counts = collect_class_image_counts(labels_dir, len(classes))
    insufficient = [
        f"{i}:{classes[i]}={counts.get(i, 0)}张"
        for i in range(len(classes))
        if counts.get(i, 0) < QUICK_TRAIN_MIN_PER_CLASS
    ]
    if insufficient:
        raise RuntimeError(
            "快速学习前类别覆盖不足。"
            f"请先为每个已创建类别至少人工确认 {QUICK_TRAIN_MIN_PER_CLASS} 张。"
            " 当前不足: " + "，".join(insufficient)
        )
    return counts


def build_quick_dataset(
    batch_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    classes: list[str],
    task_type: str,
) -> tuple[Path, int]:
    images = list_images(images_dir)
    usable: list[Path] = []

    for img in images:
        label = labels_dir / f"{img.stem}.txt"
        if non_empty_label(label):
            usable.append(img)

    if len(usable) < QUICK_TRAIN_MIN_IMAGES:
        raise RuntimeError(
            f"labels 下非空人工标注只有 {len(usable)} 张，至少需要 {QUICK_TRAIN_MIN_IMAGES} 张。"
        )
    if not classes:
        raise RuntimeError("还没有类别信息，请先标注至少一个框并选择/新建类别。")

    validate_quick_train_class_coverage(labels_dir, classes)

    dataset_dir = batch_dir / "quick_train" / "dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    # 少样本场景：保留少量验证集，但确保训练集优先有足够样本。
    rng = random.Random(42)
    rng.shuffle(usable)

    val_count = max(1, int(len(usable) * 0.2)) if len(usable) >= 6 else 1
    val_names = {p.name for p in usable[:val_count]}

    for img in usable:
        split = "val" if img.name in val_names else "train"
        shutil.copy2(img, dataset_dir / "images" / split / img.name)
        shutil.copy2(labels_dir / f"{img.stem}.txt", dataset_dir / "labels" / split / f"{img.stem}.txt")

    if not any((dataset_dir / "images" / "val").iterdir()):
        first = usable[0]
        shutil.copy2(first, dataset_dir / "images" / "val" / first.name)
        shutil.copy2(labels_dir / f"{first.stem}.txt", dataset_dir / "labels" / "val" / f"{first.stem}.txt")

    names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
    data_yaml = dataset_dir / "data.yaml"
    task_line = "task: segment\n" if normalize_annotation_task(task_type) == "segmentation" else ""
    data_yaml.write_text(
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"{task_line}\n"
        f"nc: {len(classes)}\n"
        "names:\n"
        f"{names_lines}\n",
        encoding="utf-8",
    )
    return data_yaml, len(usable)


def find_latest_best_pt(runs_dir: Path) -> Path | None:
    candidates = sorted(runs_dir.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


class QuickJobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def start(self, batch_dir: Path, name: str, target: Callable[..., dict[str, Any]], *args) -> str:
        job_id = uuid.uuid4().hex[:12]
        job_dir = batch_dir / "quick_train" / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "run.log"
        status_path = job_dir / "status.json"
        status = {
            "job_id": job_id,
            "name": name,
            "status": "running",
            "progress": 5,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": "",
            "log_path": rel(log_path),
            "message": "任务已启动",
        }
        write_json(status_path, status)
        with self.lock:
            self.jobs[job_id] = status

        def update(progress: int, message: str) -> None:
            status["progress"] = max(0, min(100, int(progress)))
            status["message"] = message
            write_json(status_path, status)

        def runner() -> None:
            try:
                with log_path.open("w", encoding="utf-8") as log_file:
                    log_file.write(f"[INFO] job_id={job_id}\n[INFO] name={name}\n")
                    log_file.flush()
                    result = target(log_file, update, *args)
                status["status"] = "success"
                status["progress"] = 100
                status["message"] = result.get("message", "完成") if isinstance(result, dict) else "完成"
                if isinstance(result, dict):
                    status.update(result)
                    status["progress"] = 100
            except Exception as exc:
                status["status"] = "failed"
                status["progress"] = 100
                status["message"] = str(exc)
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(f"\n[ERROR] {exc}\n")
            status["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            write_json(status_path, status)
            with self.lock:
                self.jobs[job_id] = status

        threading.Thread(target=runner, daemon=True).start()
        return job_id

    def get(self, batch_dir: Path, job_id: str) -> dict[str, Any]:
        status_path = batch_dir / "quick_train" / "jobs" / job_id / "status.json"
        status = read_json(status_path, default=None)
        if not status:
            raise KeyError(job_id)
        return status


quick_jobs = QuickJobManager()


def run_shell_command(cmd: str, log_file) -> None:
    log_file.write(f"[CMD] {cmd}\n")
    log_file.flush()
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log_file.write(line)
        log_file.flush()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"命令执行失败，returncode={code}")



# -----------------------------------------------------------------------------
# ROI classification 数据制作
# -----------------------------------------------------------------------------


def resolve_project_path(value: str | Path) -> Path:
    """把前端传入的相对路径统一解析到项目根目录下。"""
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def safe_label_name(name: str) -> str:
    """分类目录名安全化，防止空类别和路径穿越。"""
    cleaned = str(name).strip().replace("\\", "_").replace("/", "_")
    cleaned = cleaned.replace("..", "_")
    cleaned = "_".join(cleaned.split())
    if not cleaned:
        raise ValueError("类别名不能为空")
    return cleaned


def roi_cls_raw_dir() -> Path:
    p = ROI_CLS_RAW_DIR
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def reset_roi_cls_raw_dir() -> Path:
    """清空 data/raw_classification，避免新一轮 ROI 分类数据和旧数据混在一起。"""
    root = roi_cls_raw_dir()
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    return root


def roi_cls_sessions_dir() -> Path:
    p = ROI_CLS_SESSIONS_DIR
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def reset_roi_cls_sessions_dir() -> Path:
    """清空 data/roi_classification_sessions，避免每次生成都累积历史 session 文件夹。"""
    root = roi_cls_sessions_dir()
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    return root


def roi_cls_manifest_path(session_id: str) -> Path:
    sid = Path(str(session_id)).name
    return roi_cls_sessions_dir() / sid / "manifest.json"


def load_roi_cls_manifest(session_id: str) -> dict[str, Any]:
    path = roi_cls_manifest_path(session_id)
    data = read_json(path, default=None)
    if not data:
        raise FileNotFoundError(f"未找到 ROI 分类 session: {session_id}")
    return data


def save_roi_cls_manifest(data: dict[str, Any]) -> None:
    session_id = str(data.get("session_id") or "")
    if not session_id:
        raise ValueError("manifest 缺少 session_id")
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_json(roi_cls_manifest_path(session_id), data)


def list_roi_cls_classes() -> list[dict[str, Any]]:
    root = roi_cls_raw_dir()
    classes: list[dict[str, Any]] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        count = sum(1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        classes.append({"name": class_dir.name, "count": count})
    return classes


def create_roi_cls_class(name: str) -> dict[str, Any]:
    label = safe_label_name(name)
    class_dir = roi_cls_raw_dir() / label
    class_dir.mkdir(parents=True, exist_ok=True)
    return {"name": label, "path": rel(class_dir), "count": len(list(class_dir.glob("*")))}


def list_roi_cls_detectors() -> list[dict[str, Any]]:
    candidates: list[Path] = []
    default_model = resolve_project_path(ROI_CLS_DEFAULT_DET_MODEL)
    candidates.append(default_model)

    # ROI 分类数据制作只使用正式 detection pipeline 的检测模型。
    # 不再扫描 data/raw_collected/ 下的 quick_train 临时模型，避免误选临时/过期模型。
    checkpoints_dir = PROJECT_ROOT / "models" / "checkpoints_detection"
    if checkpoints_dir.exists():
        candidates.extend(checkpoints_dir.glob("*.pt"))
        candidates.extend(checkpoints_dir.rglob("*.pt"))

    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for path in candidates:
        try:
            p = path.resolve()
        except Exception:
            continue
        key = str(p)
        if key in seen or not p.exists() or not p.is_file():
            continue
        seen.add(key)
        result.append({
            "name": p.name,
            "path": rel(p),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    result.sort(key=lambda x: x.get("mtime", ""), reverse=True)
    return result


def list_roi_cls_sessions(limit: int = 20) -> list[dict[str, Any]]:
    root = roi_cls_sessions_dir()
    items: list[dict[str, Any]] = []
    for manifest_path in root.glob("*/manifest.json"):
        data = read_json(manifest_path, default={}) or {}
        session_id = data.get("session_id") or manifest_path.parent.name
        candidates = data.get("items", []) if isinstance(data.get("items"), list) else []
        labeled = sum(1 for x in candidates if x.get("status") == "labeled")
        items.append({
            "session_id": session_id,
            "batch_id": data.get("batch_id", ""),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "total": len(candidates),
            "labeled": labeled,
            "path": rel(manifest_path.parent),
        })
    items.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
    return items[:limit]


def normalize_model_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def pick_detection_box(
    yolo_result: Any,
    image_w: int,
    image_h: int,
    target_class_id: int | None,
    select_policy: str,
) -> tuple[list[float], float, int] | None:
    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        return None

    xyxy = yolo_result.boxes.xyxy.cpu().numpy()
    conf = yolo_result.boxes.conf.cpu().numpy()
    cls = yolo_result.boxes.cls.cpu().numpy().astype(int)
    frame_area = max(1.0, float(image_w * image_h))

    candidates: list[tuple[float, list[float], float, int]] = []
    for i in range(len(xyxy)):
        class_id = int(cls[i])
        if target_class_id is not None and class_id != target_class_id:
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh
        if area <= 1:
            continue
        if select_policy == "highest_conf":
            score = float(conf[i])
        elif select_policy == "largest_area":
            score = area
        else:
            score = float(conf[i]) * 0.7 + min(area / frame_area, 1.0) * 0.3
        candidates.append((score, [x1, y1, x2, y2], float(conf[i]), class_id))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, box, det_conf, det_cls = candidates[0]
    return box, det_conf, det_cls


def crop_roi_with_padding(
    image_path: Path,
    bbox: list[float],
    padding_ratio: float,
    crop_path: Path,
    preview_path: Path,
) -> dict[str, Any]:
    if Image is None:
        raise RuntimeError("Pillow 未安装，无法裁剪 ROI。请安装 pillow。")

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        px = bw * padding_ratio
        py = bh * padding_ratio
        rx1 = max(0, int(round(x1 - px)))
        ry1 = max(0, int(round(y1 - py)))
        rx2 = min(w, int(round(x2 + px)))
        ry2 = min(h, int(round(y2 + py)))
        if rx2 <= rx1 or ry2 <= ry1:
            raise RuntimeError(f"ROI 无效: {bbox}")

        crop = im.crop((rx1, ry1, rx2, ry2))
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path, quality=95)

        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview = im.copy()
        if ImageDraw is not None:
            draw = ImageDraw.Draw(preview)
            draw.rectangle((int(x1), int(y1), int(x2), int(y2)), outline=(0, 255, 0), width=4)
            draw.rectangle((rx1, ry1, rx2, ry2), outline=(255, 0, 0), width=3)
        preview.thumbnail((960, 720))
        preview.save(preview_path, quality=90)

    return {
        "image_size": [w, h],
        "roi_bbox": [rx1, ry1, rx2, ry2],
        "roi_size": [rx2 - rx1, ry2 - ry1],
    }




def roi_cls_detector_class_key(det_class_id: Any, det_class_name: Any) -> str:
    try:
        cid = int(det_class_id)
    except Exception:
        cid = -1
    cname = str(det_class_name or cid).strip() or str(cid)
    return f"{cid}:{cname}"


def default_relative_box() -> dict[str, float]:
    return {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}


def normalize_relative_box(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        value = default_relative_box()

    x1 = float(value.get("x1", 0.0))
    y1 = float(value.get("y1", 0.0))
    x2 = float(value.get("x2", 1.0))
    y2 = float(value.get("y2", 1.0))

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if (x2 - x1) < 0.02 or (y2 - y1) < 0.02:
        raise ValueError("精细 ROI 太小，请至少保留宽高 2% 以上的区域")

    return {
        "x1": round(x1, 6),
        "y1": round(y1, 6),
        "x2": round(x2, 6),
        "y2": round(y2, 6),
    }


def ensure_roi_policy(data: dict[str, Any]) -> dict[str, Any]:
    padding_ratio = float(data.get("padding_ratio", ROI_CLS_DEFAULT_PADDING))
    policy = data.get("roi_policy")
    if not isinstance(policy, dict):
        policy = {}

    policy.setdefault("schema_version", 1)
    policy.setdefault("mode", "class_relative_box")
    policy.setdefault("coordinate", "relative_to_padded_detection_box")
    policy.setdefault("default", {
        "enabled": False,
        "mode": "full_box",
        "base": "det_bbox_with_padding",
        "padding_ratio": padding_ratio,
        "relative_box": default_relative_box(),
    })
    if not isinstance(policy.get("by_detector_class"), dict):
        policy["by_detector_class"] = {}

    data["roi_policy"] = policy
    return policy


def roi_policy_for_item(data: dict[str, Any], item: dict[str, Any]) -> dict[str, Any] | None:
    policy = ensure_roi_policy(data)
    key = roi_cls_detector_class_key(item.get("det_class_id"), item.get("det_class_name"))
    entry = policy.get("by_detector_class", {}).get(key)
    if isinstance(entry, dict) and entry.get("enabled"):
        return entry
    return None


def compute_final_roi_bbox_from_item(data: dict[str, Any], item: dict[str, Any]) -> tuple[list[int], str, dict[str, float]]:
    """
    返回最终用于分类训练/推理的 ROI bbox。

    基础 bbox 是 item["roi_bbox"]，也就是检测框 + padding 后的 base ROI。
    如果当前检测类别配置了精细 ROI，则在 base ROI 内按 relative_box 再裁一次。
    """
    base = item.get("roi_bbox") or item.get("base_roi_bbox")
    if not isinstance(base, list) or len(base) != 4:
        raise ValueError(f"候选项缺少有效 roi_bbox: {item.get('id')}")

    bx1, by1, bx2, by2 = [float(v) for v in base]
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)

    entry = roi_policy_for_item(data, item)
    if entry:
        rel_box = normalize_relative_box(entry.get("relative_box"))
        mode = "relative_box"
    else:
        rel_box = default_relative_box()
        mode = "full_box"

    fx1 = bx1 + bw * rel_box["x1"]
    fy1 = by1 + bh * rel_box["y1"]
    fx2 = bx1 + bw * rel_box["x2"]
    fy2 = by1 + bh * rel_box["y2"]

    final = [int(round(fx1)), int(round(fy1)), int(round(fx2)), int(round(fy2))]
    if final[2] <= final[0] or final[3] <= final[1]:
        raise ValueError(f"最终 ROI 无效: {final}")

    return final, mode, rel_box


def crop_item_final_roi_to_path(data: dict[str, Any], item: dict[str, Any], dst_path: Path) -> dict[str, Any]:
    """
    从原图重新裁剪当前 item 的最终 ROI，并保存到分类训练目录。

    注意：不要直接复制 candidates/crop_xxx.jpg，因为该图只是 base ROI；
    一旦启用精细 ROI，训练图必须重新从 source_image 按 final_roi_bbox 裁剪。
    """
    if Image is None:
        raise RuntimeError("Pillow 未安装，无法裁剪 ROI。请安装 pillow。")

    source_rel = str(item.get("source_image") or "")
    if not source_rel:
        raise ValueError(f"候选项缺少 source_image: {item.get('id')}")

    source_path = PROJECT_ROOT / source_rel if not Path(source_rel).is_absolute() else Path(source_rel)
    if not source_path.exists():
        raise FileNotFoundError(f"原图不存在: {source_path}")

    final_bbox, mode, rel_box = compute_final_roi_bbox_from_item(data, item)

    with Image.open(source_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        x1, y1, x2, y2 = final_bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"最终 ROI 超出图像范围或无效: {[x1, y1, x2, y2]}")
        crop = im.crop((x1, y1, x2, y2))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst_path, quality=95)

    item["base_roi_bbox"] = item.get("roi_bbox")
    item["final_roi_bbox"] = [x1, y1, x2, y2]
    item["final_roi_size"] = [x2 - x1, y2 - y1]
    item["roi_mode"] = mode
    item["relative_box"] = rel_box

    return {
        "final_roi_bbox": item["final_roi_bbox"],
        "final_roi_size": item["final_roi_size"],
        "roi_mode": mode,
        "relative_box": rel_box,
    }


def rebuild_labeled_samples_for_detector_class(
    data: dict[str, Any],
    det_class_id: Any,
    det_class_name: Any,
) -> int:
    """
    当前检测类别的精细 ROI policy 更新后，把该检测类别下已经 labeled 的分类样本重新裁剪并覆盖。
    """
    target_key = roi_cls_detector_class_key(det_class_id, det_class_name)
    rebuilt = 0

    for item in data.get("items", []):
        if item.get("status") != "labeled":
            continue
        item_key = roi_cls_detector_class_key(item.get("det_class_id"), item.get("det_class_name"))
        if item_key != target_key:
            continue

        exported = str(item.get("exported_path") or "").strip()
        if not exported:
            continue

        dst_path = PROJECT_ROOT / exported if not Path(exported).is_absolute() else Path(exported)
        crop_item_final_roi_to_path(data, item, dst_path)
        rebuilt += 1

    return rebuilt


def _roi_cls_build_worker(log_file, update, batch_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"无法导入 ultralytics.YOLO，请确认服务端环境已安装 ultralytics: {exc}")

    _, images_dir, _, _, _ = get_batch_paths()
    images = list_images(images_dir)
    if not images:
        raise RuntimeError(f"未找到图片: {images_dir}")

    model_value = payload.get("detector_model") or ROI_CLS_DEFAULT_DET_MODEL
    model_path = resolve_project_path(model_value)
    if not model_path.exists():
        raise RuntimeError(f"检测模型不存在: {model_path}")

    update(6, "正在清空旧的 ROI 分类训练数据和历史候选 session")
    raw_dir = reset_roi_cls_raw_dir()
    sessions_dir = reset_roi_cls_sessions_dir()
    log_file.write(f"[INFO] reset raw_classification_dir={raw_dir}\n")
    log_file.write(f"[INFO] reset roi_classification_sessions_dir={sessions_dir}\n")
    log_file.flush()

    conf = float(payload.get("conf_threshold", ROI_CLS_DEFAULT_CONF))
    padding_ratio = float(payload.get("padding_ratio", ROI_CLS_DEFAULT_PADDING))
    select_policy = str(payload.get("select_policy", "conf_area"))
    target_class_id_raw = payload.get("target_class_id", None)
    target_class_name = str(payload.get("target_class_name") or "").strip()
    target_class_id: int | None = None
    if target_class_id_raw not in {None, "", "null"}:
        target_class_id = int(target_class_id_raw)

    # 每次只保留一个当前 ROI 候选 session，避免 data/roi_classification_sessions 下不断新增历史文件夹。
    session_id = "current"
    session_dir = roi_cls_sessions_dir() / session_id
    candidates_dir = session_dir / "candidates"
    previews_dir = session_dir / "previews"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    log_file.write(f"[INFO] session_id={session_id}\n")
    log_file.write(f"[INFO] images_dir={images_dir}\n")
    log_file.write(f"[INFO] detector_model={model_path}\n")
    log_file.flush()

    update(10, "正在加载检测模型")
    model = YOLO(str(model_path))
    model_names = normalize_model_names(getattr(model, "names", {}))
    # 目标类别名优先级高于目标类别 ID：两者同时填写时，以类别名为准。
    if target_class_name:
        inv = {v: k for k, v in model_names.items()}
        if target_class_name not in inv:
            raise RuntimeError(f"检测模型中找不到类别 {target_class_name}，当前类别: {model_names}")
        target_class_id = inv[target_class_name]

    items: list[dict[str, Any]] = []
    miss_count = 0
    max_candidates = ROI_CLS_MAX_CANDIDATES if ROI_CLS_MAX_CANDIDATES > 0 else len(images)

    update(15, f"开始检测并裁剪 ROI，共 {len(images)} 张")
    for idx, img_path in enumerate(images):
        if len(items) >= max_candidates:
            break
        progress = 15 + int((idx + 1) / max(1, len(images)) * 75)
        if idx % 5 == 0:
            update(progress, f"正在处理 {idx + 1}/{len(images)}: {img_path.name}")

        try:
            image_w, image_h = get_image_size(img_path)
            result = model.predict(str(img_path), conf=conf, verbose=False)[0]
            picked = pick_detection_box(
                yolo_result=result,
                image_w=image_w,
                image_h=image_h,
                target_class_id=target_class_id,
                select_policy=select_policy,
            )
            if picked is None:
                miss_count += 1
                continue

            bbox, det_conf, det_cls = picked
            item_id = f"crop_{len(items) + 1:06d}"
            crop_path = candidates_dir / f"{item_id}.jpg"
            preview_path = previews_dir / f"{item_id}.jpg"
            crop_info = crop_roi_with_padding(
                image_path=img_path,
                bbox=bbox,
                padding_ratio=padding_ratio,
                crop_path=crop_path,
                preview_path=preview_path,
            )
            items.append({
                "id": item_id,
                "status": "pending",
                "source_image": rel(img_path),
                "source_filename": img_path.name,
                "crop_path": rel(crop_path),
                "preview_path": rel(preview_path),
                "bbox": [round(float(v), 2) for v in bbox],
                "roi_bbox": crop_info["roi_bbox"],
                "roi_size": crop_info["roi_size"],
                "image_size": crop_info["image_size"],
                "det_conf": float(det_conf),
                "det_class_id": int(det_cls),
                "det_class_name": model_names.get(int(det_cls), str(det_cls)),
                "assigned_label": "",
                "exported_path": "",
            })
        except Exception as exc:
            log_file.write(f"[WARN] {img_path.name}: {exc}\n")
            log_file.flush()
            continue

    manifest = {
        "session_id": session_id,
        "task_type": "roi_classification_data",
        "batch_id": batch_dir.name,
        "images_dir": rel(images_dir),
        "detector_model": rel(model_path),
        "target_class_id": target_class_id,
        "target_class_name": target_class_name,
        "conf_threshold": conf,
        "padding_ratio": padding_ratio,
        "select_policy": select_policy,
        "raw_classification_dir": rel(roi_cls_raw_dir()),
        "roi_policy": {
            "schema_version": 1,
            "mode": "class_relative_box",
            "coordinate": "relative_to_padded_detection_box",
            "default": {
                "enabled": False,
                "mode": "full_box",
                "base": "det_bbox_with_padding",
                "padding_ratio": padding_ratio,
                "relative_box": default_relative_box(),
            },
            "by_detector_class": {},
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(images),
        "candidate_count": len(items),
        "miss_count": miss_count,
        "items": items,
    }
    write_json(session_dir / "manifest.json", manifest)
    update(95, "正在写入 ROI 分类候选 manifest")
    return {
        "message": f"ROI 候选生成完成：{len(items)} 个，未检测到目标 {miss_count} 张",
        "session_id": session_id,
        "candidate_count": len(items),
        "miss_count": miss_count,
        "manifest_path": rel(session_dir / "manifest.json"),
    }


@router.get("/api/annotator/roi-cls/session")
def roi_cls_session_info() -> dict[str, Any]:
    try:
        batch_dir, images_dir, _, _, _ = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "batch_id": batch_dir.name,
        "images_dir": rel(images_dir),
        "image_count": len(list_images(images_dir)),
        "raw_classification_dir": rel(roi_cls_raw_dir()),
        "sessions_dir": rel(roi_cls_sessions_dir()),
        "detectors": list_roi_cls_detectors(),
        "classes": list_roi_cls_classes(),
        "sessions": list_roi_cls_sessions(),
        "defaults": {
            "detector_model": ROI_CLS_DEFAULT_DET_MODEL,
            "conf_threshold": ROI_CLS_DEFAULT_CONF,
            "padding_ratio": ROI_CLS_DEFAULT_PADDING,
            "select_policy": "conf_area",
        },
    }


@router.post("/api/annotator/roi-cls/classes")
def roi_cls_add_class(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        created = create_roi_cls_class(str(payload.get("class_name") or payload.get("label") or ""))
        return {"message": "类别已创建", "class": created, "classes": list_roi_cls_classes()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/annotator/roi-cls/build-candidates")
def roi_cls_build_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        batch_dir = get_current_batch_dir()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    job_id = quick_jobs.start(batch_dir, "roi-cls-build-candidates", _roi_cls_build_worker, batch_dir, payload)
    return {"job_id": job_id, "message": "ROI 分类候选生成任务已开始"}


@router.get("/api/annotator/roi-cls/sessions/{session_id}")
def roi_cls_get_session(session_id: str) -> dict[str, Any]:
    try:
        data = load_roi_cls_manifest(session_id)
        return {"manifest": data, "classes": list_roi_cls_classes()}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/api/annotator/roi-cls/file/{session_id}/{kind}/{filename}")
def roi_cls_file(session_id: str, kind: str, filename: str) -> FileResponse:
    kind = Path(kind).name
    filename = Path(filename).name
    if kind not in {"candidates", "previews"}:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {kind}")
    path = roi_cls_sessions_dir() / Path(session_id).name / kind / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {rel(path)}")
    return FileResponse(path)


@router.post("/api/annotator/roi-cls/label")
def roi_cls_label_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        session_id = str(payload.get("session_id") or "")
        item_id = str(payload.get("item_id") or "")
        label = safe_label_name(str(payload.get("label") or ""))
        if not session_id or not item_id:
            raise ValueError("session_id 和 item_id 不能为空")

        create_roi_cls_class(label)
        data = load_roi_cls_manifest(session_id)
        items = data.get("items", [])
        item = next((x for x in items if x.get("id") == item_id), None)
        if item is None:
            raise FileNotFoundError(f"未找到候选项: {item_id}")
        crop_path = PROJECT_ROOT / item.get("crop_path", "")
        if not crop_path.exists():
            raise FileNotFoundError(f"ROI 图片不存在: {crop_path}")

        source_stem = Path(str(item.get("source_filename") or item_id)).stem
        dst_dir = roi_cls_raw_dir() / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_name = f"{session_id}_{item_id}_{source_stem}{crop_path.suffix.lower()}"
        dst_path = dst_dir / dst_name

        # 如果该候选之前已经被分到其他类别，重新选择类别时要把旧类别中的样本删除，
        # 避免同一张 ROI 同时出现在多个分类目录里。
        old_exported = str(item.get("exported_path") or "").strip()
        if old_exported:
            old_path = PROJECT_ROOT / old_exported if not Path(old_exported).is_absolute() else Path(old_exported)
            try:
                if old_path.exists() and old_path.resolve() != dst_path.resolve():
                    old_path.unlink()
            except Exception:
                pass
        for existing in roi_cls_raw_dir().glob(f"*/{dst_name}"):
            try:
                if existing.resolve() != dst_path.resolve():
                    existing.unlink()
            except Exception:
                pass

        # 根据当前 ROI policy 重新从原图裁剪最终训练图。
        # 未启用精细 ROI 时，这等价于复制 base ROI；启用后会保存精细 ROI。
        crop_meta = crop_item_final_roi_to_path(data, item, dst_path)

        item["status"] = "labeled"
        item["assigned_label"] = label
        item["exported_path"] = rel(dst_path)
        item["labeled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        item.update(crop_meta)
        save_roi_cls_manifest(data)
        return {
            "message": f"已保存为分类样本: {label}",
            "item": item,
            "classes": list_roi_cls_classes(),
            "manifest": data,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))




@router.post("/api/annotator/roi-cls/roi-policy")
def roi_cls_save_roi_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """
    保存某个检测类别的一套精细 ROI 规则。

    规则是相对于 base ROI 的比例框：
      base ROI = detection bbox + padding
      final ROI = base ROI 内的 relative_box

    同一个检测类别只保留一套规则；重复确认会覆盖该类别已有规则。
    保存后会把该检测类别下已经 labeled 的分类训练样本重新裁剪并覆盖。
    """
    try:
        session_id = str(payload.get("session_id") or "")
        item_id = str(payload.get("item_id") or "")
        enabled = bool(payload.get("enabled", True))
        if not session_id or not item_id:
            raise ValueError("session_id 和 item_id 不能为空")

        data = load_roi_cls_manifest(session_id)
        items = data.get("items", [])
        item = next((x for x in items if x.get("id") == item_id), None)
        if item is None:
            raise FileNotFoundError(f"未找到候选项: {item_id}")

        det_class_id = item.get("det_class_id")
        det_class_name = item.get("det_class_name")
        key = roi_cls_detector_class_key(det_class_id, det_class_name)
        rel_box = normalize_relative_box(payload.get("relative_box"))

        policy = ensure_roi_policy(data)
        by_class = policy.setdefault("by_detector_class", {})
        entry = {
            "enabled": enabled,
            "mode": "relative_box" if enabled else "full_box",
            "base": "det_bbox_with_padding",
            "padding_ratio": float(data.get("padding_ratio", ROI_CLS_DEFAULT_PADDING)),
            "relative_box": rel_box if enabled else default_relative_box(),
            "det_class_id": int(det_class_id),
            "det_class_name": str(det_class_name or det_class_id),
            "class_key": key,
            "coordinate": "relative_to_padded_detection_box",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        by_class[key] = entry
        policy["updated_at"] = entry["updated_at"]
        data["roi_policy"] = policy

        rebuilt_count = rebuild_labeled_samples_for_detector_class(data, det_class_id, det_class_name)
        save_roi_cls_manifest(data)

        return {
            "message": (
                f"已保存检测类别 {key} 的精细 ROI 规则，并重新裁剪已分类样本 {rebuilt_count} 张"
                if enabled else
                f"已关闭检测类别 {key} 的精细 ROI，并重新裁剪已分类样本 {rebuilt_count} 张"
            ),
            "class_key": key,
            "policy": entry,
            "rebuilt_count": rebuilt_count,
            "manifest": data,
            "classes": list_roi_cls_classes(),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/annotator/roi-cls/skip")
def roi_cls_skip_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        session_id = str(payload.get("session_id") or "")
        item_id = str(payload.get("item_id") or "")
        data = load_roi_cls_manifest(session_id)
        item = next((x for x in data.get("items", []) if x.get("id") == item_id), None)
        if item is None:
            raise FileNotFoundError(f"未找到候选项: {item_id}")
        old_exported = str(item.get("exported_path") or "").strip()
        if old_exported:
            old_path = PROJECT_ROOT / old_exported if not Path(old_exported).is_absolute() else Path(old_exported)
            try:
                if old_path.exists():
                    old_path.unlink()
            except Exception:
                pass
        item["status"] = "skipped"
        item["assigned_label"] = ""
        item["exported_path"] = ""
        item["skipped_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_roi_cls_manifest(data)
        return {"message": "已跳过", "item": item, "manifest": data, "classes": list_roi_cls_classes()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/annotator", response_class=HTMLResponse)
def annotator_page() -> str:
    return ANNOTATOR_HTML


@router.get("/api/annotator/session")
def annotation_session() -> dict[str, Any]:
    try:
        batch_dir, images_dir, labels_dir, labels_auto_dir, classes_path = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    images = list_images(images_dir)
    if not images:
        raise HTTPException(status_code=404, detail=f"未找到图片: {images_dir}")

    quick_state = load_quick_state(batch_dir)
    return {
        "batch_id": batch_dir.name,
        "images_dir": rel(images_dir),
        "labels_dir": rel(labels_dir),
        "labels_auto_dir": rel(labels_auto_dir),
        "classes_path": rel(classes_path),
        "classes": load_annotation_classes(classes_path),
        "images": [p.name for p in images],
        "total": len(images),
        "manual_label_count": count_manual_labels(labels_dir),
        "auto_label_count": count_auto_labels(labels_auto_dir),
        "quick_train": quick_state.get("quick_train", {}),
        "last_auto_label": quick_state.get("last_auto_label", {}),
        "default_task_type": load_annotation_task(batch_dir),
    }


@router.get("/api/annotator/image/{index}")
def annotation_image(index: int) -> dict[str, Any]:
    try:
        batch_dir, images_dir, labels_dir, labels_auto_dir, _ = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    images = list_images(images_dir)
    if index < 0 or index >= len(images):
        raise HTTPException(status_code=404, detail=f"图片索引越界: {index}")

    img = images[index]
    image_w, image_h = get_image_size(img)
    manual_label_path = labels_dir / f"{img.stem}.txt"
    auto_label_path = labels_auto_dir / f"{img.stem}.txt"

    if manual_label_path.exists():
        label_source = "manual"
        active_label_path = manual_label_path
    elif non_empty_label(auto_label_path):
        label_source = "auto"
        active_label_path = auto_label_path
    else:
        label_source = "none"
        active_label_path = manual_label_path

    task_type = load_annotation_task(batch_dir)
    annotations = parse_yolo_label(active_label_path, image_w=image_w, image_h=image_h, task_type=task_type)

    return {
        "index": index,
        "filename": img.name,
        "image_url": f"/api/annotator/file/{index}",
        "image_w": image_w,
        "image_h": image_h,
        "manual_label_path": rel(manual_label_path),
        "auto_label_path": rel(auto_label_path),
        "label_path": rel(active_label_path),
        "label_source": label_source,
        "needs_confirm": label_source == "auto",
        "task_type": task_type,
        "annotations": annotations,
    }


@router.get("/api/annotator/file/{index}")
def annotation_file(index: int) -> FileResponse:
    try:
        _, images_dir, _, _, _ = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    images = list_images(images_dir)
    if index < 0 or index >= len(images):
        raise HTTPException(status_code=404, detail=f"图片索引越界: {index}")

    return FileResponse(images[index])


@router.post("/api/annotator/classes")
def annotation_save_classes(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        batch_dir, _, _, _, classes_path = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    classes = payload.get("classes", [])
    if not isinstance(classes, list):
        raise HTTPException(status_code=400, detail="classes 必须是列表")

    task_type = normalize_annotation_task(payload.get("task_type") or load_annotation_task(batch_dir))
    save_annotation_task(batch_dir, task_type)
    save_annotation_classes(classes_path, [str(x) for x in classes])
    return {"message": "类别已保存", "classes_path": rel(classes_path), "num_classes": len(classes), "task_type": task_type}


@router.post("/api/annotator/task")
def annotation_save_task(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        batch_dir, _, _, _, _ = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    task_type = normalize_annotation_task(payload.get("task_type"))
    save_annotation_task(batch_dir, task_type)
    return {"message": "任务类型已保存", "task_type": task_type}


@router.post("/api/annotator/save")
def annotation_save(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        batch_dir, images_dir, labels_dir, _, classes_path = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    filename = str(payload.get("filename", ""))
    if not filename:
        raise HTTPException(status_code=400, detail="filename 不能为空")

    image_path = images_dir / Path(filename).name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"图片不存在: {filename}")

    image_w, image_h = get_image_size(image_path)
    task_type = normalize_annotation_task(payload.get("task_type", "detection"))
    if task_type not in {"detection", "obb", "segmentation"}:
        raise HTTPException(status_code=400, detail=f"不支持的任务类型: {task_type}")
    save_annotation_task(batch_dir, task_type)

    annotations = payload.get("annotations", [])
    if not isinstance(annotations, list):
        raise HTTPException(status_code=400, detail="annotations 必须是列表")

    classes = payload.get("classes", [])
    if isinstance(classes, list):
        save_annotation_classes(classes_path, [str(x) for x in classes])

    label_path = labels_dir / f"{image_path.stem}.txt"
    save_yolo_label(label_path=label_path, annotations=annotations, image_w=image_w, image_h=image_h, task_type=task_type)
    return {"message": "已保存人工标注", "label_path": rel(label_path), "count": len(annotations), "task_type": task_type}


@router.post("/api/annotator/confirm-auto")
def annotation_confirm_auto(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        _, images_dir, labels_dir, labels_auto_dir, _ = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    filename = str(payload.get("filename", ""))
    if not filename:
        raise HTTPException(status_code=400, detail="filename 不能为空")

    image_path = images_dir / Path(filename).name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"图片不存在: {filename}")

    src = labels_auto_dir / f"{image_path.stem}.txt"
    dst = labels_dir / f"{image_path.stem}.txt"
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"没有找到自动标注文件: {rel(src)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"message": "已确认自动标注并复制到 labels", "label_path": rel(dst)}


def _quick_train_worker(log_file, update, batch_dir: Path, task_type: str, classes: list[str]) -> dict[str, Any]:
    _, images_dir, labels_dir, _, _ = get_batch_paths()
    update(10, "正在扫描 labels 下的人工标注")
    data_yaml, count = build_quick_dataset(batch_dir, images_dir, labels_dir, classes, task_type)
    runs_dir = batch_dir / "quick_train" / "runs"
    yolo_task = task_to_yolo_subcommand(task_type)
    model = quick_model_for_task(task_type)
    run_name = f"{yolo_task}_quick"
    run_dir = runs_dir / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    update(25, f"开始快速学习，训练图片 {count} 张")
    cmd = (
        f"{shlex.quote(QUICK_YOLO_CMD)} {yolo_task} train "
        f"model={shlex.quote(model)} "
        f"data={shlex.quote(str(data_yaml))} "
        f"epochs={QUICK_TRAIN_EPOCHS} imgsz={QUICK_TRAIN_IMGSZ} batch={QUICK_TRAIN_BATCH} "
        f"mosaic=0.0 mixup=0.0 copy_paste=0.0 degrees=0.0 perspective=0.0 "
        f"translate=0.02 scale=0.5 fliplr=0.5 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 "
        f"patience=20 "
        f"project={shlex.quote(str(runs_dir))} name={shlex.quote(run_name)} exist_ok=True"
    )
    run_shell_command(cmd, log_file)
    update(90, "正在整理快速学习模型")
    best_pt = find_latest_best_pt(run_dir) or find_latest_best_pt(runs_dir)
    if best_pt is None:
        raise RuntimeError("快速学习完成，但没有找到 best.pt")
    state = load_quick_state(batch_dir)
    state["quick_train"] = {
        "task_type": task_type,
        "model_path": rel(best_pt),
        "train_images": count,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_quick_state(batch_dir, state)
    return {"message": "快速学习完成", "model_path": rel(best_pt), "train_images": count}


def _auto_label_worker(log_file, update, batch_dir: Path, task_type: str) -> dict[str, Any]:
    _, images_dir, labels_dir, labels_auto_dir, _ = get_batch_paths()
    state = load_quick_state(batch_dir)
    model_path = state.get("quick_train", {}).get("model_path")
    if not model_path:
        raise RuntimeError("还没有快速学习模型，请先点击“快速学习”。")
    model_abs = PROJECT_ROOT / model_path if not Path(model_path).is_absolute() else Path(model_path)
    if not model_abs.exists():
        raise RuntimeError(f"快速学习模型不存在: {model_abs}")

    images = list_images(images_dir)
    remaining = [p for p in images if not (labels_dir / f"{p.stem}.txt").exists()]
    if not remaining:
        return {"message": "没有剩余未确认图片", "auto_labeled_count": 0}

    update(10, f"清理历史自动标注，准备预标注 {len(remaining)} 张图片")
    if labels_auto_dir.exists():
        shutil.rmtree(labels_auto_dir)
    labels_auto_dir.mkdir(parents=True, exist_ok=True)

    source_dir = batch_dir / "quick_train" / "predict_source"
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    for img in remaining:
        shutil.copy2(img, source_dir / img.name)

    pred_root = batch_dir / "quick_train" / "predict"
    if pred_root.exists():
        shutil.rmtree(pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)
    yolo_task = task_to_yolo_subcommand(task_type)
    run_name = f"{yolo_task}_predict"
    update(30, "正在用快速学习模型预标注剩余图片")
    cmd = (
        f"{shlex.quote(QUICK_YOLO_CMD)} {yolo_task} predict "
        f"model={shlex.quote(str(model_abs))} source={shlex.quote(str(source_dir))} "
        f"imgsz={QUICK_TRAIN_IMGSZ} conf={AUTO_LABEL_CONF} save_txt=True save_conf=False "
        f"project={shlex.quote(str(pred_root))} name={shlex.quote(run_name)} exist_ok=True"
    )
    run_shell_command(cmd, log_file)

    update(85, "正在写入 labels_auto")
    pred_labels = pred_root / run_name / "labels"
    count = 0
    non_empty = 0
    for img in remaining:
        src_label = pred_labels / f"{img.stem}.txt"
        dst_label = labels_auto_dir / f"{img.stem}.txt"
        if src_label.exists() and non_empty_label(src_label):
            shutil.copy2(src_label, dst_label)
            non_empty += 1
        elif dst_label.exists():
            dst_label.unlink()
        count += 1

    state["last_auto_label"] = {
        "task_type": task_type,
        "count": count,
        "non_empty_count": non_empty,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_quick_state(batch_dir, state)
    return {"message": "预标注剩余图片完成", "auto_labeled_count": count, "auto_non_empty_count": non_empty}


@router.post("/api/annotator/quick-train")
def annotation_quick_train(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        batch_dir, _, _, _, classes_path = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    task_type = normalize_annotation_task(payload.get("task_type", "detection"))
    if task_type not in {"detection", "obb", "segmentation"}:
        raise HTTPException(status_code=400, detail=f"不支持的任务类型: {task_type}")
    save_annotation_task(batch_dir, task_type)
    classes = payload.get("classes")
    if isinstance(classes, list):
        save_annotation_classes(classes_path, [str(x) for x in classes])
    else:
        classes = load_annotation_classes(classes_path)
    job_id = quick_jobs.start(batch_dir, "quick-train", _quick_train_worker, batch_dir, task_type, [str(x) for x in classes])
    return {"job_id": job_id, "message": "快速学习已开始"}


@router.post("/api/annotator/auto-label-remaining")
def annotation_auto_label_remaining(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        batch_dir, _, _, _, _ = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    task_type = normalize_annotation_task(payload.get("task_type", "detection"))
    if task_type not in {"detection", "obb", "segmentation"}:
        raise HTTPException(status_code=400, detail=f"不支持的任务类型: {task_type}")
    save_annotation_task(batch_dir, task_type)
    job_id = quick_jobs.start(batch_dir, "auto-label-remaining", _auto_label_worker, batch_dir, task_type)
    return {"job_id": job_id, "message": "预标注剩余图片已开始"}


@router.get("/api/annotator/jobs/{job_id}")
def annotation_job_status(job_id: str) -> dict[str, Any]:
    try:
        batch_dir = get_current_batch_dir()
        return quick_jobs.get(batch_dir, job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"未知任务: {job_id}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


ANNOTATOR_HTML = r'''
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>VisionOps 标注器</title>
  <style>
    * { box-sizing: border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; background:#eef2f7; color:#111827; overflow:hidden; }
    .top { height:52px; display:flex; align-items:center; gap:12px; padding:0 14px; background:#fff; border-bottom:1px solid #dbe3ef; }
    .title { font-weight:800; font-size:17px; }
    .badge { background:#eef2ff; color:#3730a3; border-radius:999px; padding:5px 9px; font-size:12px; white-space:nowrap; }
    .badge.warn { background:#fff7ed; color:#c2410c; }
    .layout { display:grid; grid-template-columns:220px minmax(0,1fr) 300px; height:calc(100vh - 52px); }
    .side { background:#fff; border-right:1px solid #dbe3ef; overflow:auto; padding:10px; }
    .right { background:#fff; border-left:1px solid #dbe3ef; overflow:auto; padding:14px; }
    .center { overflow:hidden; position:relative; background:#e9eef6; }
    canvas { width:100%; height:100%; display:block; cursor:default; }
    canvas.draw-ready { cursor:crosshair; }
    button, select, input { border:0; border-radius:9px; padding:9px 11px; font-weight:700; }
    button { background:#2563eb; color:#fff; cursor:pointer; }
    button.dark { background:#111827; }
    button.green { background:#059669; }
    button.orange { background:#d97706; }
    button.gray { background:#e5e7eb; color:#111827; }
    button.big { width:100%; padding:14px 14px; font-size:16px; border-radius:12px; }
    select, input { background:#f3f4f6; color:#111827; width:100%; }
    .row { display:flex; gap:7px; flex-wrap:wrap; margin:9px 0; }
    .imgitem { padding:7px 9px; border-radius:8px; cursor:pointer; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .imgitem.active { background:#2563eb; color:#fff; }
    .imgitem.auto { border-left:4px solid #f97316; }
    .imgitem.manual { border-left:4px solid #059669; }
    .muted { color:#64748b; font-size:13px; line-height:1.55; }
    .section-title { margin:16px 0 7px; font-size:15px; font-weight:800; }
    .boxlist { font-size:13px; margin-top:10px; }
    .boxitem { padding:7px; border:1px solid #e5e7eb; border-radius:8px; margin-bottom:6px; cursor:pointer; }
    .boxitem.active { border-color:#2563eb; background:#eff6ff; }
    .hint { background:#f8fafc; border:1px solid #e5e7eb; padding:10px; border-radius:10px; font-size:12px; line-height:1.6; }
    .auto-notice { display:none; background:#fff7ed; border:1px solid #fdba74; color:#9a3412; border-radius:12px; padding:10px; font-size:13px; line-height:1.5; margin:10px 0; }
    .canvas-auto-warning {
      display:none; position:absolute; top:18px; right:22px; z-index:4;
      pointer-events:none; user-select:none;
      background:rgba(220,38,38,.18); color:rgba(185,28,28,.92);
      border:2px solid rgba(220,38,38,.45); border-radius:16px;
      padding:14px 20px; font-size:30px; font-weight:900; letter-spacing:1px;
      text-shadow:0 1px 2px rgba(255,255,255,.85);
    }
    .assist-panel { margin-top:18px; padding-top:14px; border-top:1px solid #e5e7eb; }
    .roi-launch-panel { margin-top:18px; padding-top:14px; border-top:1px solid #e5e7eb; }
    .roi-modal { width:min(1500px, 98vw); max-height:96vh; overflow:hidden; padding:0; display:flex; flex-direction:column; }
    .roi-modal-header { display:flex; align-items:center; justify-content:space-between; gap:14px; padding:20px 24px; border-bottom:1px solid #e5e7eb; }
    .roi-modal-header h2 { margin:0; font-size:26px; }
    .roi-modal-header .muted { font-size:15px; }
    .roi-modal-body { display:grid; grid-template-columns:390px minmax(0,1fr); gap:0; min-height:0; overflow:hidden; }
    .roi-modal-left { padding:22px; border-right:1px solid #e5e7eb; overflow:auto; max-height:calc(96vh - 84px); }
    .roi-modal-main { padding:22px 24px; overflow:auto; max-height:calc(96vh - 84px); }
    .roi-modal h3, .roi-modal .section-title { font-size:19px; }
    .roi-modal .muted, .roi-modal .hint, .roi-modal .quick-msg, .roi-modal .roi-small { font-size:15px; }
    .roi-modal button, .roi-modal select, .roi-modal input { font-size:16px; padding:11px 13px; }
    .roi-preview { width:100%; max-height:360px; object-fit:contain; background:#f8fafc; border:1px solid #e5e7eb; border-radius:12px; margin:8px 0; }
    .roi-crop { max-width:100%; max-height:360px; width:auto; height:auto; object-fit:contain; background:#111827; border-radius:12px; margin:0; display:block; }
    .roi-crop-wrap { display:inline-block; position:relative; max-width:100%; background:#111827; border-radius:12px; margin:8px 0; overflow:hidden; vertical-align:top; }
    .fine-roi-panel { margin:0; padding:12px 14px; border:1px solid #e5e7eb; border-radius:12px; background:#f8fafc; min-height:100%; }
    .fine-roi-toolbar { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:0 0 6px; }
    .fine-roi-toolbar label { display:inline-flex; align-items:center; gap:6px; font-size:15px; font-weight:800; color:#111827; }
    .fine-roi-toolbar input[type="checkbox"] { width:auto; }
    .fine-roi-msg { color:#64748b; font-size:14px; line-height:1.45; }
    .roi-review-top { display:grid; grid-template-columns:minmax(0,1.08fr) minmax(360px,.92fr); gap:18px; align-items:start; margin:6px 0 12px; }
    .roi-info-card { border:1px solid #e5e7eb; border-radius:12px; background:#fff; padding:12px 14px; }
    .roi-info-head { display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; margin-bottom:8px; font-size:15px; color:#111827; }
    .roi-info-head strong { font-size:17px; }
    .roi-info-chip { display:inline-flex; align-items:center; padding:4px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:700; font-size:13px; }
    .roi-info-lines { display:grid; gap:5px; }
    .roi-info-line { font-size:14px; color:#475569; line-height:1.45; }
    .roi-info-line b { color:#111827; }
    .roi-info-file { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .fine-roi-box { display:none; position:absolute; border:3px solid #f59e0b; background:rgba(245,158,11,.16); box-shadow:0 0 0 9999px rgba(15,23,42,.16); cursor:move; z-index:5; }
    .fine-roi-handle { position:absolute; width:14px; height:14px; background:#f59e0b; border:2px solid #fff; border-radius:999px; z-index:6; }
    .fine-roi-handle.nw { left:-8px; top:-8px; cursor:nwse-resize; }
    .fine-roi-handle.ne { right:-8px; top:-8px; cursor:nesw-resize; }
    .fine-roi-handle.sw { left:-8px; bottom:-8px; cursor:nesw-resize; }
    .fine-roi-handle.se { right:-8px; bottom:-8px; cursor:nwse-resize; }
    .roi-class-buttons { display:flex; flex-wrap:wrap; gap:10px; margin:10px 0; }
    .roi-class-buttons button { background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; padding:10px 12px; }
    .roi-small { font-size:15px; color:#475569; line-height:1.6; white-space:pre-wrap; }
    .roi-nav { display:flex; gap:10px; margin:10px 0; }
    .roi-modal-grid { display:grid; grid-template-columns:minmax(420px, 1.05fr) minmax(320px, .95fr); gap:18px; align-items:start; }
    .roi-image-col { display:flex; flex-direction:column; align-items:flex-start; }
    .roi-image-col.crop-col { align-items:center; }
    .roi-image-col .muted { width:100%; }
    .roi-crop-stage { width:100%; display:flex; justify-content:center; align-items:flex-start; }
    .roi-field-label { margin:12px 0 6px; font-size:15px; font-weight:800; color:#111827; }
    .roi-param-note { margin:6px 0 0; color:#64748b; font-size:13px; line-height:1.45; }
    .roi-class-card { background:#f8fafc; border:1px solid #e5e7eb; border-radius:14px; padding:14px; margin-bottom:14px; }
    .progress { height:14px; background:#e5e7eb; border-radius:999px; overflow:hidden; margin-top:10px; }
    .progress-inner { height:100%; width:0%; background:#2563eb; transition:width .3s ease; }
    .quick-msg { margin-top:8px; font-size:12px; color:#475569; line-height:1.45; }
    .modal-mask { display:none; position:fixed; inset:0; background:rgba(15,23,42,.45); z-index:10; align-items:center; justify-content:center; }
    .modal { width:420px; background:#fff; border-radius:16px; padding:18px; box-shadow:0 20px 60px rgba(15,23,42,.3); }

    /* ROI 分类数据制作弹窗必须覆盖通用 .modal 的 420px 宽度。
       上一版 .modal 写在 .roi-modal 后面，导致 .roi-modal 被压回 420px，
       右侧审核区只剩很窄一条。这里用更高优先级的 .modal.roi-modal 修正。 */
    .modal.roi-modal {
      width:min(1500px, 98vw); height:min(1050px, 96vh); max-height:96vh;
      padding:0; display:flex; flex-direction:column; overflow:hidden;
    }
    .modal.roi-modal .roi-modal-body {
      flex:1; min-height:0; display:grid;
      grid-template-columns:minmax(380px, 420px) minmax(760px, 1fr);
      overflow:hidden;
    }
    .modal.roi-modal .roi-modal-left,
    .modal.roi-modal .roi-modal-main {
      max-height:none; min-height:0; overflow:auto;
    }
    .modal.roi-modal .roi-preview { max-height:360px; }
    .modal.roi-modal .roi-crop { max-height:360px; }
    @media (max-width: 1180px) {
      .modal.roi-modal { width:98vw; height:96vh; }
      .modal.roi-modal .roi-modal-body { grid-template-columns:360px minmax(0,1fr); }
      .roi-review-top, .roi-modal-grid { grid-template-columns:1fr; }
    }
    @media (max-width: 880px) {
      .modal.roi-modal .roi-modal-body { grid-template-columns:1fr; overflow:auto; }
      .modal.roi-modal .roi-modal-left,
      .modal.roi-modal .roi-modal-main { overflow:visible; max-height:none; }
    }

    .class-grid { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
    .class-chip { background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; }

    .review-done-float { position:fixed; right:22px; bottom:22px; z-index:25; width:260px; }
    .review-done-float button { width:100%; padding:15px 16px; border-radius:14px; font-size:16px; box-shadow:0 14px 32px rgba(217,119,6,.32); }
    .review-job-msg { margin-top:8px; background:rgba(255,255,255,.96); border:1px solid #fed7aa; color:#9a3412; border-radius:12px; padding:9px 10px; font-size:12px; line-height:1.45; box-shadow:0 10px 26px rgba(15,23,42,.12); display:none; white-space:pre-wrap; }
  </style>
</head>
<body>
  <div class="top">
    <div class="title">VisionOps 标注器</div>
    <div id="batch" class="badge">batch: -</div>
    <div id="progress" class="badge">0 / 0</div>
    <div id="labelSource" class="badge">标注来源: -</div>
    <div id="saveStatus" class="badge">未保存</div>
    <div id="zoomStatus" class="badge">zoom: 100%</div>
  </div>

  <div class="layout">
    <div class="side">
      <div class="row">
        <button onclick="prevImage()">A 上一张</button>
        <button onclick="nextImage()">D 下一张</button>
      </div>
      <div id="imageList"></div>
    </div>

    <div id="canvasWrap" class="center">
      <canvas id="canvas"></canvas>
      <div id="canvasAutoWarning" class="canvas-auto-warning">自动标注待确认</div>
    </div>

    <div class="right">
      <h3>标注设置</h3>
      <div class="muted">任务类型</div>
      <select id="taskType" onchange="onTaskChange()">
        <option value="detection">Detection 矩形框</option>
        <option value="obb">OBB 四点框</option>
        <option value="segmentation">Segmentation 多边形</option>
      </select>

      <div class="section-title">类别</div>
      <div id="classSummary" class="muted">还没有类别。画完第一个框后新增类别。</div>

      <div class="row">
        <button class="green" onclick="startDraw()">W 开始标注</button>
        <button class="orange" onclick="deleteSelected()">Q 删除</button>
        <button class="dark" onclick="saveCurrent(true)">S 保存</button>
        <button class="gray" onclick="finishSegmentationPolygon()">Enter 完成多边形</button>
      </div>

      <div id="autoNotice" class="auto-notice">
        自动标注待确认：无误按 <b>E</b>，或点击下方按钮复制到 labels。
        <div class="row"><button class="green" onclick="confirmAutoLabel()">E 确认自动标注</button></div>
      </div>

      <div class="hint">
        A/D：上一张/下一张。W：开始标注。Q：删除。S：保存。E：确认自动标注。Esc：取消。<br>
        Detection：W 后拖拽矩形框。OBB：W 后依次点击 4 个角点。Segmentation：W 后依次点击多边形点，双击或 Enter 完成。鼠标滚轮缩放。<br>
        人工标注保存到 labels；模型预标注保存到 labels_auto。
      </div>

      <div class="assist-panel">
        <h3>模型辅助</h3>
        <button class="green big" onclick="quickTrain()">快速学习</button>
        <div style="height:10px"></div>
        <button class="orange big" onclick="autoLabelRemaining()">预标注剩余图片</button>
        <div class="progress"><div id="quickProgress" class="progress-inner"></div></div>
        <div id="quickMsg" class="quick-msg">状态：未开始</div>
      </div>

      <div class="roi-launch-panel">
        <h3>ROI 分类数据制作</h3>
        <div class="hint">
          用 detection 模型把当前 batch 的目标整体框裁剪成分类 ROI，再人工分配类别，输出到 data/raw_classification/&lt;类别&gt;/。
        </div>
        <button class="green big" onclick="openRoiModal()">打开 ROI 分类数据制作</button>
      </div>

      <h3>当前标注</h3>
      <div id="boxList" class="boxlist"></div>
    </div>
  </div>


  <div class="review-done-float">
    <button class="orange" onclick="acceptReviewedFromAnnotator()">确认审核完成</button>
    <div id="reviewJobMsg" class="review-job-msg"></div>
  </div>

  <div id="classModal" class="modal-mask">
    <div class="modal">
      <h3 style="margin:0 0 8px;">选择这个框的类别</h3>
      <div class="muted">类别顺序就是 YOLO 标签里的下标。第一次新增类别下标为 0，后续依次递增。</div>
      <div id="existingClasses" class="class-grid"></div>
      <div class="section-title">新建类别</div>
      <input id="newClassInput" placeholder="例如 earphone / phone / defect" />
      <div class="row">
        <button class="green" onclick="confirmNewClass()">新建并使用</button>
        <button class="gray" onclick="cancelClassModal()">取消本次框</button>
      </div>
    </div>
  </div>

  <div id="roiClsModal" class="modal-mask">
    <div class="modal roi-modal">
      <div class="roi-modal-header">
        <div>
          <h2>ROI 分类数据制作</h2>
          <div class="muted">第一版 roi_classification：检测框整体裁剪 → 人工选择分类类别 → 保存到 data/raw_classification。</div>
        </div>
        <button class="gray" onclick="closeRoiModal()">关闭</button>
      </div>

      <div class="roi-modal-body">
        <div class="roi-modal-left">
          <div class="hint">
            生成候选时会先清空 <b>data/raw_classification/</b> 下已有数据，并删除旧的 ROI 候选 session，避免新旧分类样本混在一起。请确认旧数据不再需要后再生成。
          </div>

          <div class="section-title">1. 检测模型</div>
          <select id="roiDetectorSelect"></select>

          <div class="section-title">2. 检测与裁剪参数</div>
          <div class="row">
            <div style="width:48%">
              <div class="roi-field-label">目标类别 ID</div>
              <input id="roiTargetClassId" placeholder="可空，例如 0" />
              <div class="roi-param-note">只裁剪该检测类别；留空表示所有类别都可作为候选。</div>
            </div>
            <div style="width:48%">
              <div class="roi-field-label">目标类别名</div>
              <input id="roiTargetClassName" placeholder="可空，例如 tube" />
              <div class="roi-param-note">和类别 ID 二选一即可；同时填写时优先使用类别名。</div>
            </div>
          </div>
          <div class="row">
            <div style="width:48%">
              <div class="roi-field-label">检测阈值 conf</div>
              <input id="roiConf" placeholder="默认 0.35" />
              <div class="roi-param-note">低于该置信度的检测框不会参与裁剪；数值越高，候选越少但更可靠。</div>
            </div>
            <div style="width:48%">
              <div class="roi-field-label">裁剪扩边 padding</div>
              <input id="roiPadding" placeholder="默认 0.05，可设 0" />
              <div class="roi-param-note">0 表示严格按检测框裁剪；0.05 表示四周各扩展框宽/高的 5%。</div>
            </div>
          </div>
          <div class="roi-field-label">多框选择策略</div>
          <select id="roiSelectPolicy">
            <option value="conf_area">置信度+面积</option>
            <option value="highest_conf">最高置信度</option>
            <option value="largest_area">最大面积</option>
          </select>
          <div class="roi-param-note">每张图只取一个目标框作为 ROI 候选；如果画面中只有一个目标，默认“置信度+面积”即可。</div>
          <div style="height:14px"></div>
          <button class="green big" onclick="buildRoiCandidates()">生成 ROI 候选</button>
          <div class="progress"><div id="roiProgress" class="progress-inner"></div></div>
          <div id="roiMsg" class="quick-msg">状态：未开始</div>
        </div>

        <div class="roi-modal-main">
          <div class="roi-class-card">
            <div class="section-title" style="margin-top:0">3. 分类类别</div>
            <div class="muted">点击类别按钮即可把当前 ROI 保存到 data/raw_classification/&lt;类别&gt;/。类别不限于 ok/ng，可按缺陷类型新增。</div>
            <div id="roiClassButtons" class="roi-class-buttons"></div>
            <div class="row">
              <input id="roiNewClassInput" placeholder="新增分类类别，例如 ok / gap_large / scratch" style="width:70%" />
              <button class="gray" onclick="addRoiClass()">新增</button>
            </div>
          </div>

          <div class="section-title">4. 候选 ROI 审核</div>
          <div class="roi-review-top">
            <div id="roiCandidateInfo" class="roi-info-card">暂无候选。</div>
            <div class="fine-roi-panel">
              <div class="fine-roi-toolbar">
                <label><input id="fineRoiEnable" type="checkbox" onchange="onFineRoiToggle()" /> 启用精细 ROI</label>
                <button id="fineRoiConfirmBtn" class="gray" onclick="confirmFineRoiPolicy()" style="display:none">确认精细 ROI</button>
              </div>
              <div id="fineRoiMsg" class="fine-roi-msg">默认不启用精细 ROI，直接使用检测框 + padding 的完整 ROI。</div>
            </div>
          </div>
          <div class="roi-modal-grid">
            <div class="roi-image-col">
              <div class="muted">原图预览：绿色为检测框，红色为实际裁剪 ROI</div>
              <img id="roiPreviewImg" class="roi-preview" style="display:none" />
            </div>
            <div class="roi-image-col crop-col">
              <div class="muted">裁剪后的分类 ROI（已限制显示高度，实际训练图仍为原始裁剪尺寸）</div>
              <div class="roi-crop-stage">
              <div id="roiCropWrap" class="roi-crop-wrap" style="display:none">
                <img id="roiCropImg" class="roi-crop" />
                <div id="fineRoiBox" class="fine-roi-box">
                  <div class="fine-roi-handle nw" data-handle="nw"></div>
                  <div class="fine-roi-handle ne" data-handle="ne"></div>
                  <div class="fine-roi-handle sw" data-handle="sw"></div>
                  <div class="fine-roi-handle se" data-handle="se"></div>
                </div>
              </div>
              </div>
            </div>
          </div>
          <div class="roi-nav">
            <button class="gray" onclick="prevRoiCandidate()">上一张</button>
            <button class="gray" onclick="nextRoiCandidate()">下一张</button>
            <button class="orange" onclick="skipRoiCandidate()">跳过</button>
          </div>
          <div class="hint">
            先确认裁剪 ROI 是否完整覆盖目标，再点击上方类别按钮完成归类。分类类别不限制为 ok/ng，可以按实际缺陷类型新增。
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
let session = null;
let currentIndex = 0;
let currentMeta = null;
let img = new Image();
let annotations = [];
let classes = [];
let selectedIndex = -1;
let drawingMode = false;
let dragging = false;
let startPt = null;
let tempPt = null;
let obbPoints = [];
let segPoints = [];
let pendingAnnotation = null;
let baseScale = 1, zoom = 1, scale = 1, offsetX = 0, offsetY = 0;
let mouseCanvas = null;
let quickJobTimer = null;
let roiJobTimer = null;
let roiInfo = null;
let roiSession = null;
let roiCandidates = [];
let roiCurrentIndex = 0;
let fineRoiState = {
  enabled: false,
  box: {x1:0, y1:0, x2:1, y2:1},
  dragging: false,
  dragMode: '',
  startMouse: null,
  startBox: null
};
let labelSource = 'none';
let dirty = false;

const canvas = document.getElementById('canvas');
const wrap = document.getElementById('canvasWrap');
const ctx = canvas.getContext('2d');

function currentTask() { return document.getElementById('taskType').value; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

async function init() {
  const res = await fetch('/api/annotator/session');
  const data = await res.json();
  if (!res.ok) { alert(data.detail || '初始化失败'); return; }
  session = data;
  classes = Array.isArray(session.classes) ? [...session.classes] : [];
  document.getElementById('taskType').value = session.default_task_type || 'detection';
  document.getElementById('batch').textContent = 'batch: ' + session.batch_id;
  renderClassSummary();
  resizeCanvas();
  await loadImage(0);
  await loadRoiClsInfo();
}

async function refreshSession() {
  const res = await fetch('/api/annotator/session');
  const data = await res.json();
  if (res.ok) session = data;
}

function renderClassSummary() {
  const el = document.getElementById('classSummary');
  el.textContent = classes.length ? classes.map((n, i) => `${i}: ${n}`).join('，') : '还没有类别。画完第一个框后新增类别。';
}

function imageLabelKind(name) {
  if (!session) return '';
  const stem = name.replace(/\.[^.]+$/, '');
  // 后端没有返回逐图状态，这里只按当前图刷新；列表状态简化显示。
  return '';
}

function renderImageList() {
  document.getElementById('imageList').innerHTML = session.images.map((name, i) =>
    `<div class="imgitem ${i===currentIndex?'active':''}" onclick="gotoImage(${i})">${i+1}. ${name}</div>`
  ).join('');
}

function updateLabelSourceUI() {
  const el = document.getElementById('labelSource');
  const notice = document.getElementById('autoNotice');
  const canvasWarn = document.getElementById('canvasAutoWarning');
  const isAuto = labelSource === 'auto';
  el.className = 'badge';
  if (labelSource === 'manual') el.textContent = '标注来源: 人工 labels';
  else if (isAuto) { el.textContent = '标注来源: 自动 labels_auto，待确认'; el.className = 'badge warn'; }
  else el.textContent = '标注来源: 无';
  notice.style.display = isAuto ? 'block' : 'none';
  canvasWarn.style.display = isAuto ? 'block' : 'none';
}

async function loadImage(index) {
  currentIndex = Math.max(0, Math.min(index, session.total - 1));
  const res = await fetch('/api/annotator/image/' + currentIndex);
  const data = await res.json();
  if (!res.ok) { alert(data.detail || '加载图片失败'); return; }
  currentMeta = data;
  if (data.task_type) document.getElementById('taskType').value = data.task_type;
  annotations = data.annotations || [];
  labelSource = data.label_source || 'none';
  dirty = false;
  selectedIndex = -1;
  drawingMode = false;
  dragging = false;
  obbPoints = [];
  segPoints = [];
  pendingAnnotation = null;
  mouseCanvas = null;
  zoom = 1;
  img = new Image();
  img.onload = () => { fitImageToCanvas(); draw(); };
  img.src = data.image_url + '?t=' + Date.now();
  document.getElementById('progress').textContent = `${currentIndex + 1} / ${session.total}`;
  document.getElementById('saveStatus').textContent = '已加载';
  updateLabelSourceUI();
  renderImageList();
  renderBoxList();
}

function resizeCanvas() {
  const rect = wrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(300, Math.floor(rect.width * dpr));
  canvas.height = Math.max(300, Math.floor(rect.height * dpr));
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (img && img.naturalWidth) { fitImageToCanvas(true); draw(); }
}

function canvasCssSize() { const r = canvas.getBoundingClientRect(); return {w:r.width, h:r.height}; }
function fitImageToCanvas(keepZoom=false) {
  const {w, h} = canvasCssSize();
  baseScale = Math.min(w / img.naturalWidth, h / img.naturalHeight);
  if (!Number.isFinite(baseScale) || baseScale <= 0) baseScale = 1;
  if (!keepZoom) zoom = 1;
  scale = baseScale * zoom;
  offsetX = (w - img.naturalWidth * scale) / 2;
  offsetY = (h - img.naturalHeight * scale) / 2;
  updateZoomStatus();
}
function updateZoomStatus() { document.getElementById('zoomStatus').textContent = 'zoom: ' + Math.round(zoom * 100) + '%'; }
function toCanvasPt(evt) { const r = canvas.getBoundingClientRect(); return {x:evt.clientX-r.left, y:evt.clientY-r.top}; }
function toImgPtFromCanvas(p) { return {x:clamp((p.x-offsetX)/scale,0,img.naturalWidth), y:clamp((p.y-offsetY)/scale,0,img.naturalHeight)}; }
function toImgPt(evt) { return toImgPtFromCanvas(toCanvasPt(evt)); }
function sx(x){return x*scale+offsetX;} function sy(y){return y*scale+offsetY;}

function draw() {
  const {w, h} = canvasCssSize();
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#e9eef6'; ctx.fillRect(0,0,w,h);
  if (img && img.naturalWidth) ctx.drawImage(img, offsetX, offsetY, img.naturalWidth*scale, img.naturalHeight*scale);
  annotations.forEach((ann,i)=>drawAnn(ann, i===selectedIndex));
  if (dragging && startPt && tempPt) drawAnn({type:'bbox', class_id:-1, x1:startPt.x, y1:startPt.y, x2:tempPt.x, y2:tempPt.y}, true, true);
  if (obbPoints.length) drawAnn({type:'obb', class_id:-1, points:obbPoints}, true, true);
  if (segPoints.length) drawAnn({type:'segmentation', class_id:-1, points:segPoints}, true, true);
  if (drawingMode && mouseCanvas) drawCrosshair(mouseCanvas.x, mouseCanvas.y);
}

function drawCrosshair(x, y) {
  const {w, h} = canvasCssSize();
  ctx.save(); ctx.strokeStyle='rgba(239,68,68,.85)'; ctx.lineWidth=1; ctx.setLineDash([6,4]);
  ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.moveTo(x,0); ctx.lineTo(x,h); ctx.stroke();
  ctx.setLineDash([]); ctx.fillStyle='rgba(239,68,68,.95)'; ctx.font='16px Arial'; ctx.fillText('标注中', x+8, y-8); ctx.restore();
}
function clsName(id) { if (id < 0) return '待选择类别'; return classes[id] || ('class_' + id); }
function drawAnn(ann, active=false, temp=false) {
  ctx.save(); ctx.lineWidth=active?3:2; ctx.strokeStyle=temp?'#f59e0b':(active?'#22c55e':'#38bdf8'); ctx.fillStyle=ctx.strokeStyle;
  const label = `${ann.class_id>=0?ann.class_id:'?'}: ${clsName(ann.class_id)}`;
  if (ann.type === 'obb' || ann.type === 'segmentation') {
    const pts = ann.points || [];
    if (pts.length) {
      ctx.beginPath(); ctx.moveTo(sx(pts[0][0]), sy(pts[0][1]));
      for (let i=1;i<pts.length;i++) ctx.lineTo(sx(pts[i][0]), sy(pts[i][1]));
      if (ann.type === 'segmentation' && pts.length >= 3) { ctx.closePath(); ctx.fillStyle='rgba(34,197,94,.20)'; ctx.fill(); ctx.fillStyle=ctx.strokeStyle; }
      if (ann.type === 'obb' && pts.length === 4) ctx.closePath();
      ctx.stroke();
      pts.forEach(p=>{ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 4, 0, Math.PI*2); ctx.fill();});
      ctx.font = "35px Arial";
      ctx.fillText(label, sx(pts[0][0])+4, sy(pts[0][1])-6);
    }
  } else {
    const x1=Math.min(ann.x1,ann.x2), y1=Math.min(ann.y1,ann.y2), x2=Math.max(ann.x1,ann.x2), y2=Math.max(ann.y1,ann.y2);
    ctx.strokeRect(sx(x1), sy(y1), (x2-x1)*scale, (y2-y1)*scale);
    ctx.font = "35px Arial";
    ctx.fillText(label, sx(x1)+4, sy(y1)-6);
  }
  ctx.restore();
}
function renderBoxList() {
  const list = document.getElementById('boxList');
  if (!annotations.length) { list.innerHTML='<div class="muted">暂无标注</div>'; return; }
  list.innerHTML = annotations.map((a,i)=>`<div class="boxitem ${i===selectedIndex?'active':''}" onclick="selectAnn(${i})">${i+1}. ${a.type === 'segmentation' ? 'seg' : a.type} | ${a.class_id}: ${clsName(a.class_id)}</div>`).join('');
}
function selectAnn(i){selectedIndex=i; renderBoxList(); draw();}
function markDirty(){ dirty = true; if (labelSource === 'auto') document.getElementById('saveStatus').textContent = '已修改自动标注，需保存为人工标注'; }
function startDraw(){drawingMode=true; dragging=false; startPt=null; tempPt=null; obbPoints=[]; segPoints=[]; pendingAnnotation=null; canvas.classList.add('draw-ready'); document.getElementById('saveStatus').textContent='标注中'; draw();}
function stopDraw(){drawingMode=false; dragging=false; startPt=null; tempPt=null; obbPoints=[]; segPoints=[]; canvas.classList.remove('draw-ready'); draw();}
async function onTaskChange(){stopDraw(); await fetch('/api/annotator/task',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_type:currentTask()})});}
function finishSegmentationPolygon(){
  if (currentTask() !== 'segmentation' || segPoints.length < 3) { return; }
  pendingAnnotation={type:'segmentation', class_id:-1, points:segPoints.map(p=>[p[0],p[1]])};
  segPoints=[]; stopDraw(); openClassModal(pendingAnnotation);
}

canvas.addEventListener('mousedown', e => {
  if (!drawingMode) return;
  const p = toImgPt(e);
  if (currentTask() === 'detection') { dragging=true; startPt=p; tempPt=p; }
  else if (currentTask() === 'segmentation') {
    segPoints.push([p.x,p.y]);
    draw();
  } else {
    obbPoints.push([p.x,p.y]);
    if (obbPoints.length === 4) { pendingAnnotation={type:'obb', class_id:-1, points:obbPoints.map(p=>[p[0],p[1]])}; obbPoints=[]; stopDraw(); openClassModal(pendingAnnotation); }
    draw();
  }
});
canvas.addEventListener('dblclick', e => {
  if (drawingMode && currentTask() === 'segmentation') finishSegmentationPolygon();
});
canvas.addEventListener('mousemove', e => { mouseCanvas=toCanvasPt(e); if (dragging) tempPt=toImgPt(e); if (drawingMode || dragging) draw(); });
canvas.addEventListener('mouseleave', () => { mouseCanvas=null; if (drawingMode || dragging) draw(); });
canvas.addEventListener('mouseup', e => {
  if (!dragging || currentTask() !== 'detection') return;
  tempPt = toImgPt(e);
  if (Math.abs(tempPt.x-startPt.x)>3 && Math.abs(tempPt.y-startPt.y)>3) {
    pendingAnnotation={type:'bbox', class_id:-1, x1:startPt.x, y1:startPt.y, x2:tempPt.x, y2:tempPt.y};
    dragging=false; stopDraw(); openClassModal(pendingAnnotation);
  } else { dragging=false; draw(); }
});
canvas.addEventListener('wheel', e => {
  e.preventDefault(); if (!img || !img.naturalWidth) return;
  const p = toCanvasPt(e); const before = toImgPtFromCanvas(p); const factor = e.deltaY < 0 ? 1.12 : 0.89;
  zoom = clamp(zoom * factor, 0.25, 8); scale = baseScale * zoom;
  offsetX = p.x - before.x * scale; offsetY = p.y - before.y * scale;
  updateZoomStatus(); draw();
}, {passive:false});

function openClassModal(ann) {
  pendingAnnotation = ann;
  const existing = document.getElementById('existingClasses');
  existing.innerHTML = classes.length ? classes.map((name,i)=>`<button class="class-chip" onclick="confirmExistingClass(${i})">${i}: ${name}</button>`).join('') : '<div class="muted">暂无类别，请新建第一个类别。</div>';
  document.getElementById('newClassInput').value = '';
  document.getElementById('classModal').style.display = 'flex';
  setTimeout(()=>document.getElementById('newClassInput').focus(),50);
}
function closeClassModal(){document.getElementById('classModal').style.display='none';}
function confirmExistingClass(classId){ if(!pendingAnnotation)return; pendingAnnotation.class_id=classId; annotations.push(pendingAnnotation); selectedIndex=annotations.length-1; pendingAnnotation=null; closeClassModal(); markDirty(); renderBoxList(); draw(); }
async function confirmNewClass(){ const input=document.getElementById('newClassInput'); const name=input.value.trim(); if(!name){alert('请输入类别名'); return;} let classId=classes.indexOf(name); if(classId<0){classes.push(name); classId=classes.length-1; renderClassSummary(); await saveClassesOnly();} confirmExistingClass(classId); }
function cancelClassModal(){ pendingAnnotation=null; closeClassModal(); draw(); }
async function saveClassesOnly(){ await fetch('/api/annotator/classes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({classes, task_type: currentTask()})}); }

async function saveCurrent(force=false){
  if(!currentMeta) return;
  // 自动标注未修改时，切图不会自动确认；必须按 E。
  if (labelSource === 'auto' && !dirty && !force) return;
  if (labelSource === 'none' && annotations.length === 0 && !dirty && !force) return;
  const res = await fetch('/api/annotator/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:currentMeta.filename, task_type:currentTask(), annotations, classes})});
  const data = await res.json();
  if(!res.ok){alert(data.detail || '保存失败'); return;}
  labelSource = 'manual';
  dirty = false;
  updateLabelSourceUI();
  document.getElementById('saveStatus').textContent = '已保存人工标注 ' + annotations.length + ' 个';
}

async function confirmAutoLabel(){
  if(!currentMeta) return;
  const res = await fetch('/api/annotator/confirm-auto',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:currentMeta.filename})});
  const data = await res.json();
  if(!res.ok){alert(data.detail || '确认失败'); return;}
  labelSource = 'manual';
  dirty = false;
  updateLabelSourceUI();
  document.getElementById('saveStatus').textContent = '已确认自动标注';
  await refreshSession();
}

async function nextImage(){await saveCurrent(false); await loadImage(currentIndex+1);}
async function prevImage(){await saveCurrent(false); await loadImage(currentIndex-1);}
async function gotoImage(i){await saveCurrent(false); await loadImage(i);}
function deleteSelected(){ if(!annotations.length)return; const i=selectedIndex>=0?selectedIndex:annotations.length-1; annotations.splice(i,1); selectedIndex=-1; markDirty(); renderBoxList(); draw(); }

function setQuickProgress(percent, text) {
  document.getElementById('quickProgress').style.width = Math.max(0, Math.min(100, percent || 0)) + '%';
  document.getElementById('quickMsg').textContent = text;
}

async function pollQuickJob(jobId) {
  const res = await fetch('/api/annotator/jobs/' + jobId);
  const data = await res.json();
  if (!res.ok) { setQuickProgress(100, data.detail || '任务状态读取失败'); return; }
  const text = `任务：${data.name}\n状态：${data.status}\n进度：${data.progress || 0}%\n${data.message || ''}`;
  setQuickProgress(data.progress || 0, text);
  if (data.status === 'success' || data.status === 'failed') {
    if (quickJobTimer) clearInterval(quickJobTimer);
    quickJobTimer = null;
    await refreshSession();
    if (data.status === 'success' && data.name === 'auto-label-remaining') await loadImage(currentIndex);
  }
}

async function startQuickJob(url, label) {
  await saveCurrent(true);
  const res = await fetch(url, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_type:currentTask(), classes})});
  const data = await res.json();
  if (!res.ok) { alert(data.detail || (label + '启动失败')); return; }
  setQuickProgress(5, data.message + '\njob_id=' + data.job_id);
  if (quickJobTimer) clearInterval(quickJobTimer);
  quickJobTimer = setInterval(() => pollQuickJob(data.job_id), 1500);
  pollQuickJob(data.job_id);
}

async function quickTrain() {
  const ok = confirm('将扫描 labels 下非空人工标注文件，并检查每个类别的人工样本数后快速训练临时模型。继续？');
  if (!ok) return;
  await startQuickJob('/api/annotator/quick-train', '快速学习');
}

async function autoLabelRemaining() {
  const ok = confirm('将清空 labels_auto，并用快速学习模型预标注所有 labels 中还没有确认文件的图片。继续？');
  if (!ok) return;
  await startQuickJob('/api/annotator/auto-label-remaining', '预标注剩余图片');
}

function setRoiProgress(percent, text) {
  const bar = document.getElementById('roiProgress');
  const msg = document.getElementById('roiMsg');
  if (bar) bar.style.width = Math.max(0, Math.min(100, percent || 0)) + '%';
  if (msg) msg.textContent = text || '';
}

async function openRoiModal() {
  const modal = document.getElementById('roiClsModal');
  if (modal) modal.style.display = 'flex';
  await loadRoiClsInfo();
}

function closeRoiModal() {
  const modal = document.getElementById('roiClsModal');
  if (modal) modal.style.display = 'none';
}

async function loadRoiClsInfo() {
  const res = await fetch('/api/annotator/roi-cls/session');
  const data = await res.json();
  if (!res.ok) {
    setRoiProgress(0, data.detail || 'ROI 分类数据制作信息加载失败');
    return;
  }
  roiInfo = data;
  renderRoiDetectorOptions();
  renderRoiClassButtons();
  if (Array.isArray(data.sessions) && data.sessions.length) {
    await loadRoiSession(data.sessions[0].session_id, false);
  } else {
    renderRoiCandidate();
  }
}

function renderRoiDetectorOptions() {
  const sel = document.getElementById('roiDetectorSelect');
  if (!sel || !roiInfo) return;
  const detectors = Array.isArray(roiInfo.detectors) ? roiInfo.detectors : [];
  if (!detectors.length) {
    sel.innerHTML = '<option value="">未发现检测模型，请先在 models/checkpoints_detection/ 下放置 .pt 模型</option>';
  } else {
    sel.innerHTML = detectors.map(m => `<option value="${escapeHtml(m.path)}">${escapeHtml(m.path)}</option>`).join('');
  }
  const d = roiInfo.defaults || {};
  document.getElementById('roiConf').value = d.conf_threshold || '0.35';
  document.getElementById('roiPadding').value = d.padding_ratio || '0.05';
  document.getElementById('roiSelectPolicy').value = d.select_policy || 'conf_area';
}

function escapeHtml(s) {
  return String(s || '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function renderRoiClassButtons() {
  const el = document.getElementById('roiClassButtons');
  if (!el) return;
  const classes = roiInfo && Array.isArray(roiInfo.classes) ? roiInfo.classes : [];
  if (!classes.length) {
    el.innerHTML = '<div class="muted">暂无分类类别。先新增 ok / ng 或其他具体类别。</div>';
    return;
  }
  el.innerHTML = classes.map(c =>
    `<button onclick="labelRoiCandidate('${escapeJs(c.name)}')">${escapeHtml(c.name)} (${c.count || 0})</button>`
  ).join('');
}

function escapeJs(s) {
  return String(s || '').replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

async function addRoiClass() {
  const input = document.getElementById('roiNewClassInput');
  const name = input.value.trim();
  if (!name) { alert('请输入类别名'); return; }
  const res = await fetch('/api/annotator/roi-cls/classes', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({class_name:name})
  });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || '新增类别失败'); return; }
  input.value = '';
  await loadRoiClsInfo();
  setRoiProgress(0, '类别已新增：' + data.class.name);
}

async function buildRoiCandidates() {
  const detectorModel = document.getElementById('roiDetectorSelect').value;
  if (!detectorModel) { alert('请先选择检测模型。模型需要位于 models/checkpoints_detection/ 下。'); return; }

  const ok = confirm('将使用 detection 模型对当前 batch 的 all_images 批量检测，并把检测框整体裁剪为 ROI 候选。\n\n注意：开始生成前会清空 data/raw_classification/ 下已有分类数据，并删除旧的 ROI 候选 session，避免新旧样本混乱。继续？');
  if (!ok) return;

  const payload = {
    detector_model: detectorModel,
    target_class_id: document.getElementById('roiTargetClassId').value.trim(),
    target_class_name: document.getElementById('roiTargetClassName').value.trim(),
    conf_threshold: document.getElementById('roiConf').value.trim() || 0.35,
    padding_ratio: document.getElementById('roiPadding').value.trim() || 0.05,
    select_policy: document.getElementById('roiSelectPolicy').value || 'conf_area'
  };
  if (payload.target_class_id === '') payload.target_class_id = null;

  const res = await fetch('/api/annotator/roi-cls/build-candidates', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || '生成 ROI 候选失败'); return; }
  setRoiProgress(5, data.message + '\njob_id=' + data.job_id);
  if (roiJobTimer) clearInterval(roiJobTimer);
  roiJobTimer = setInterval(() => pollRoiJob(data.job_id), 1500);
  pollRoiJob(data.job_id);
}

async function pollRoiJob(jobId) {
  const res = await fetch('/api/annotator/jobs/' + jobId);
  const data = await res.json();
  if (!res.ok) { setRoiProgress(100, data.detail || 'ROI 任务状态读取失败'); return; }
  const text = `任务：${data.name}\n状态：${data.status}\n进度：${data.progress || 0}%\n${data.message || ''}`;
  setRoiProgress(data.progress || 0, text);
  if (data.status === 'success' || data.status === 'failed') {
    if (roiJobTimer) clearInterval(roiJobTimer);
    roiJobTimer = null;
    await loadRoiClsInfo();
    if (data.status === 'success' && data.session_id) {
      await loadRoiSession(data.session_id, true);
    }
  }
}

async function loadRoiSession(sessionId, showMsg=true) {
  if (!sessionId) return;
  const res = await fetch('/api/annotator/roi-cls/sessions/' + encodeURIComponent(sessionId));
  const data = await res.json();
  if (!res.ok) { if(showMsg) alert(data.detail || '加载 ROI session 失败'); return; }
  roiSession = data.manifest;
  roiInfo = roiInfo || {};
  roiInfo.classes = data.classes || roiInfo.classes || [];
  roiCandidates = Array.isArray(roiSession.items) ? roiSession.items : [];
  const firstPending = roiCandidates.findIndex(x => x.status !== 'labeled' && x.status !== 'skipped');
  roiCurrentIndex = firstPending >= 0 ? firstPending : 0;
  renderRoiClassButtons();
  renderRoiCandidate();
}

function roiDetectorClassKey(item) {
  if (!item) return '';
  const cid = Number.isFinite(Number(item.det_class_id)) ? Number(item.det_class_id) : -1;
  const cname = String(item.det_class_name || cid).trim() || String(cid);
  return `${cid}:${cname}`;
}

function getRoiPolicyForItem(item) {
  if (!roiSession || !item) return null;
  const policy = roiSession.roi_policy || {};
  const byClass = policy.by_detector_class || {};
  const entry = byClass[roiDetectorClassKey(item)];
  if (entry && entry.enabled) return entry;
  return null;
}

function currentFineRoiBoxText() {
  const b = fineRoiState.box || {x1:0,y1:0,x2:1,y2:1};
  return `x1=${b.x1.toFixed(3)}, y1=${b.y1.toFixed(3)}, x2=${b.x2.toFixed(3)}, y2=${b.y2.toFixed(3)}`;
}

function setFineRoiMessage(text) {
  const el = document.getElementById('fineRoiMsg');
  if (el) el.textContent = text || '';
}

function updateFineRoiOverlay() {
  const boxEl = document.getElementById('fineRoiBox');
  const wrap = document.getElementById('roiCropWrap');
  const imgEl = document.getElementById('roiCropImg');
  const confirmBtn = document.getElementById('fineRoiConfirmBtn');
  if (!boxEl || !wrap || !imgEl) return;

  if (!fineRoiState.enabled || !imgEl.complete || !imgEl.naturalWidth) {
    boxEl.style.display = 'none';
    if (confirmBtn) confirmBtn.style.display = 'none';
    return;
  }

  const w = imgEl.clientWidth;
  const h = imgEl.clientHeight;
  if (!w || !h) return;

  const b = fineRoiState.box || {x1:0,y1:0,x2:1,y2:1};
  boxEl.style.display = 'block';
  boxEl.style.left = (b.x1 * w) + 'px';
  boxEl.style.top = (b.y1 * h) + 'px';
  boxEl.style.width = Math.max(10, (b.x2 - b.x1) * w) + 'px';
  boxEl.style.height = Math.max(10, (b.y2 - b.y1) * h) + 'px';
  if (confirmBtn) confirmBtn.style.display = 'inline-block';
  setFineRoiMessage('已启用精细 ROI。拖动橙色框或四角调整后，点击“确认精细 ROI”。当前比例：' + currentFineRoiBoxText());
}

function initFineRoiForCurrentItem() {
  const checkbox = document.getElementById('fineRoiEnable');
  const item = roiCandidates[roiCurrentIndex];
  const policy = getRoiPolicyForItem(item);

  if (policy && policy.relative_box) {
    fineRoiState.enabled = true;
    fineRoiState.box = {
      x1: Number(policy.relative_box.x1 ?? 0),
      y1: Number(policy.relative_box.y1 ?? 0),
      x2: Number(policy.relative_box.x2 ?? 1),
      y2: Number(policy.relative_box.y2 ?? 1)
    };
    if (checkbox) checkbox.checked = true;
    setFineRoiMessage(`当前检测类别 ${roiDetectorClassKey(item)} 已保存精细 ROI，可拖动后再次确认覆盖。比例：${currentFineRoiBoxText()}`);
  } else {
    fineRoiState.enabled = false;
    fineRoiState.box = {x1:0, y1:0, x2:1, y2:1};
    if (checkbox) checkbox.checked = false;
    setFineRoiMessage('默认不启用精细 ROI，直接使用检测框 + padding 的完整 ROI。');
  }
  updateFineRoiOverlay();
}

function onFineRoiToggle() {
  const checkbox = document.getElementById('fineRoiEnable');
  fineRoiState.enabled = !!(checkbox && checkbox.checked);
  if (fineRoiState.enabled) {
    // 默认完整覆盖当前 base ROI，相当于“不裁减”；用户可拖动四角变成更精细区域。
    if (!fineRoiState.box) fineRoiState.box = {x1:0, y1:0, x2:1, y2:1};
  }
  updateFineRoiOverlay();
  if (!fineRoiState.enabled) {
    setFineRoiMessage('默认不启用精细 ROI，直接使用检测框 + padding 的完整 ROI。');
  }
}

function getFineRoiMouseRel(evt) {
  const imgEl = document.getElementById('roiCropImg');
  const rect = imgEl.getBoundingClientRect();
  return {
    x: clamp((evt.clientX - rect.left) / Math.max(1, rect.width), 0, 1),
    y: clamp((evt.clientY - rect.top) / Math.max(1, rect.height), 0, 1)
  };
}

function normalizeFineRoiBox(b) {
  let x1 = clamp(Number(b.x1), 0, 1), y1 = clamp(Number(b.y1), 0, 1);
  let x2 = clamp(Number(b.x2), 0, 1), y2 = clamp(Number(b.y2), 0, 1);
  if (x2 < x1) [x1, x2] = [x2, x1];
  if (y2 < y1) [y1, y2] = [y2, y1];
  const minSize = 0.02;
  if (x2 - x1 < minSize) x2 = Math.min(1, x1 + minSize);
  if (y2 - y1 < minSize) y2 = Math.min(1, y1 + minSize);
  return {x1, y1, x2, y2};
}

function beginFineRoiDrag(evt) {
  if (!fineRoiState.enabled) return;
  evt.preventDefault();
  evt.stopPropagation();
  const handle = evt.target && evt.target.dataset ? evt.target.dataset.handle : '';
  fineRoiState.dragging = true;
  fineRoiState.dragMode = handle || 'move';
  fineRoiState.startMouse = getFineRoiMouseRel(evt);
  fineRoiState.startBox = {...fineRoiState.box};
}

function updateFineRoiDrag(evt) {
  if (!fineRoiState.dragging) return;
  evt.preventDefault();
  const p = getFineRoiMouseRel(evt);
  const s = fineRoiState.startMouse;
  const b0 = fineRoiState.startBox;
  const dx = p.x - s.x;
  const dy = p.y - s.y;
  let b = {...b0};

  if (fineRoiState.dragMode === 'move') {
    const width = b0.x2 - b0.x1;
    const height = b0.y2 - b0.y1;
    let nx1 = clamp(b0.x1 + dx, 0, 1 - width);
    let ny1 = clamp(b0.y1 + dy, 0, 1 - height);
    b = {x1:nx1, y1:ny1, x2:nx1 + width, y2:ny1 + height};
  } else {
    if (fineRoiState.dragMode.includes('n')) b.y1 = p.y;
    if (fineRoiState.dragMode.includes('s')) b.y2 = p.y;
    if (fineRoiState.dragMode.includes('w')) b.x1 = p.x;
    if (fineRoiState.dragMode.includes('e')) b.x2 = p.x;
  }

  fineRoiState.box = normalizeFineRoiBox(b);
  updateFineRoiOverlay();
}

function endFineRoiDrag() {
  if (!fineRoiState.dragging) return;
  fineRoiState.dragging = false;
  fineRoiState.dragMode = '';
  fineRoiState.startMouse = null;
  fineRoiState.startBox = null;
}

async function confirmFineRoiPolicy() {
  if (!roiSession || !roiCandidates.length) { alert('暂无 ROI 候选'); return; }
  const item = roiCandidates[roiCurrentIndex];
  const checkbox = document.getElementById('fineRoiEnable');
  if (!checkbox || !checkbox.checked) { alert('请先勾选“启用精细 ROI”'); return; }

  const classKey = roiDetectorClassKey(item);
  const ok = confirm(`确认为检测类别 ${classKey} 保存这一套精细 ROI？\n\n同一个检测类别只会保留一套精细 ROI。确认后会重新裁剪该检测类别下已经分类的训练样本。`);
  if (!ok) return;

  const res = await fetch('/api/annotator/roi-cls/roi-policy', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      session_id: roiSession.session_id,
      item_id: item.id,
      enabled: true,
      relative_box: fineRoiState.box
    })
  });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || '保存精细 ROI 失败'); return; }

  roiSession = data.manifest;
  roiCandidates = Array.isArray(roiSession.items) ? roiSession.items : [];
  roiInfo.classes = data.classes || roiInfo.classes || [];
  renderRoiClassButtons();
  renderRoiCandidate();
  setRoiProgress(0, data.message);
}

function renderRoiCandidate() {
  const info = document.getElementById('roiCandidateInfo');
  const preview = document.getElementById('roiPreviewImg');
  const crop = document.getElementById('roiCropImg');
  const cropWrap = document.getElementById('roiCropWrap');
  if (!info || !preview || !crop || !cropWrap) return;
  if (!roiCandidates.length) {
    info.textContent = '暂无候选。请先点击“生成 ROI 候选”。';
    preview.style.display = 'none';
    cropWrap.style.display = 'none';
    return;
  }
  roiCurrentIndex = Math.max(0, Math.min(roiCurrentIndex, roiCandidates.length - 1));
  const item = roiCandidates[roiCurrentIndex];
  const labeled = roiCandidates.filter(x => x.status === 'labeled').length;
  const skipped = roiCandidates.filter(x => x.status === 'skipped').length;
  const statusText = `${item.status || 'pending'}${item.assigned_label ? ' → ' + item.assigned_label : ''}`;
  const detText = `${item.det_class_name || item.det_class_id} · conf ${Number(item.det_conf || 0).toFixed(3)}`;
  const baseSize = (item.roi_size || []).join(' × ') || '-';
  const finalSize = (item.final_roi_size || []).join(' × ') || baseSize;
  const fileText = item.source_filename || '-';

  info.innerHTML = `
    <div class="roi-info-head">
      <strong>${roiCurrentIndex + 1} / ${roiCandidates.length}</strong>
      <span class="roi-info-chip">已分类 ${labeled}</span>
      <span class="roi-info-chip">跳过 ${skipped}</span>
      <span class="roi-info-chip">状态 ${statusText}</span>
    </div>
    <div class="roi-info-lines">
      <div class="roi-info-line"><b>检测：</b>${detText}</div>
      <div class="roi-info-line"><b>ROI：</b>基础 ${baseSize}　|　训练 ${finalSize}</div>
      <div class="roi-info-line roi-info-file"><b>来源：</b>${fileText}</div>
    </div>`;

  const sid = roiSession.session_id;
  preview.src = `/api/annotator/roi-cls/file/${encodeURIComponent(sid)}/previews/${encodeURIComponent(item.id + '.jpg')}?t=${Date.now()}`;
  crop.onload = () => { initFineRoiForCurrentItem(); };
  crop.src = `/api/annotator/roi-cls/file/${encodeURIComponent(sid)}/candidates/${encodeURIComponent(item.id + '.jpg')}?t=${Date.now()}`;
  preview.style.display = 'block';
  cropWrap.style.display = 'inline-block';
}

function nextUnfinishedRoiIndex(start) {
  if (!roiCandidates.length) return 0;
  for (let i = start; i < roiCandidates.length; i++) {
    if (roiCandidates[i].status !== 'labeled' && roiCandidates[i].status !== 'skipped') return i;
  }
  for (let i = 0; i < roiCandidates.length; i++) {
    if (roiCandidates[i].status !== 'labeled' && roiCandidates[i].status !== 'skipped') return i;
  }
  return Math.min(start, roiCandidates.length - 1);
}

function nextRoiCandidate() { if (!roiCandidates.length) return; roiCurrentIndex = Math.min(roiCandidates.length - 1, roiCurrentIndex + 1); renderRoiCandidate(); }
function prevRoiCandidate() { if (!roiCandidates.length) return; roiCurrentIndex = Math.max(0, roiCurrentIndex - 1); renderRoiCandidate(); }

async function labelRoiCandidate(label) {
  if (!roiSession || !roiCandidates.length) { alert('暂无 ROI 候选'); return; }
  const item = roiCandidates[roiCurrentIndex];
  const res = await fetch('/api/annotator/roi-cls/label', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({session_id:roiSession.session_id, item_id:item.id, label})
  });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || '保存 ROI 分类样本失败'); return; }
  roiSession = data.manifest;
  roiCandidates = Array.isArray(roiSession.items) ? roiSession.items : [];
  roiInfo.classes = data.classes || roiInfo.classes || [];
  renderRoiClassButtons();
  roiCurrentIndex = nextUnfinishedRoiIndex(roiCurrentIndex + 1);
  renderRoiCandidate();
  setRoiProgress(0, data.message);
}

async function skipRoiCandidate() {
  if (!roiSession || !roiCandidates.length) return;
  const item = roiCandidates[roiCurrentIndex];
  const res = await fetch('/api/annotator/roi-cls/skip', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({session_id:roiSession.session_id, item_id:item.id})
  });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || '跳过失败'); return; }
  roiSession = data.manifest;
  roiCandidates = Array.isArray(roiSession.items) ? roiSession.items : [];
  roiCurrentIndex = nextUnfinishedRoiIndex(roiCurrentIndex + 1);
  renderRoiCandidate();
}



function setupFineRoiEvents() {
  const box = document.getElementById('fineRoiBox');
  if (box && !box.dataset.bound) {
    box.dataset.bound = '1';
    box.addEventListener('mousedown', beginFineRoiDrag);
  }
  window.addEventListener('mousemove', updateFineRoiDrag);
  window.addEventListener('mouseup', endFineRoiDrag);
  window.addEventListener('resize', updateFineRoiOverlay);
}
setupFineRoiEvents();

let reviewJobTimer = null;

function setReviewJobMsg(text, visible=true) {
  const el = document.getElementById('reviewJobMsg');
  if (!el) return;
  el.style.display = visible ? 'block' : 'none';
  el.textContent = text || '';
}

async function pollReviewJob(jobId) {
  const res = await fetch('/api/jobs/' + jobId + '/logs');
  const data = await res.json();
  if (!res.ok) {
    setReviewJobMsg(data.detail || '审核入库任务状态读取失败');
    return;
  }
  const status = data.status || {};
  setReviewJobMsg(`审核入库任务：${status.status || '-'}
job_id=${jobId}`);
  if (status.status === 'success' || status.status === 'failed') {
    if (reviewJobTimer) clearInterval(reviewJobTimer);
    reviewJobTimer = null;
    setReviewJobMsg(`审核入库任务：${status.status}
${status.status === 'success' ? '已整理到训练数据目录。' : '请回控制台查看运行日志。'}`);
  }
}

async function acceptReviewedFromAnnotator() {
  const ok = confirm('确认当前 batch 的标注已经审核完成，并整理到对应的 raw_detection/raw_obb/raw_segmentation？');
  if (!ok) return;
  await saveCurrent(true);
  const res = await fetch('/api/accept-reviewed', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({task_type: currentTask()}) });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || '确认审核完成失败'); return; }
  setReviewJobMsg(`${data.message}
job_id=${data.job_id}`);
  if (reviewJobTimer) clearInterval(reviewJobTimer);
  reviewJobTimer = setInterval(() => pollReviewJob(data.job_id), 1200);
  pollReviewJob(data.job_id);
}

window.addEventListener('keydown', async e => {
  if (e.target && ['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) { if(e.key==='Enter' && e.target.id==='newClassInput') await confirmNewClass(); return; }
  const k=e.key.toLowerCase();
  if(k==='a') await prevImage();
  if(k==='d') await nextImage();
  if(k==='w') startDraw();
  if(k==='s') await saveCurrent(true);
  if(k==='q') deleteSelected();
  if(k==='e') await confirmAutoLabel();
  if(k==='enter') finishSegmentationPolygon();
  if(k==='escape'){ pendingAnnotation=null; closeClassModal(); closeRoiModal(); stopDraw(); }
});
window.addEventListener('resize', resizeCanvas);
init();
</script>
</body>
</html>
'''
