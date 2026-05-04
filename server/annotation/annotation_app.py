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
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_CONTEXT_PATH = DATA_DIR / "model_context" / "manifest.json"

QUICK_TRAIN_MIN_IMAGES = int(os.environ.get("VISIONOPS_QUICK_TRAIN_MIN_IMAGES", "5"))
QUICK_TRAIN_EPOCHS = int(os.environ.get("VISIONOPS_QUICK_TRAIN_EPOCHS", "50"))
QUICK_TRAIN_IMGSZ = int(os.environ.get("VISIONOPS_QUICK_TRAIN_IMGSZ", "640"))
QUICK_TRAIN_BATCH = int(os.environ.get("VISIONOPS_QUICK_TRAIN_BATCH", "2"))
QUICK_DET_MODEL = os.environ.get("VISIONOPS_QUICK_DET_MODEL", "models/pretrained/yolov8n.pt")
QUICK_OBB_MODEL = os.environ.get("VISIONOPS_QUICK_OBB_MODEL", "models/pretrained/yolov8n-obb.pt")
QUICK_YOLO_CMD = os.environ.get("VISIONOPS_QUICK_YOLO_CMD", "yolo")
AUTO_LABEL_CONF = float(os.environ.get("VISIONOPS_QUICK_AUTO_CONF", "0.25"))
QUICK_TRAIN_MIN_PER_CLASS = int(os.environ.get("VISIONOPS_QUICK_TRAIN_MIN_PER_CLASS", "3"))

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


def task_to_yolo_subcommand(task_type: str) -> str:
    return "obb" if task_type == "obb" else "detect"


def quick_model_for_task(task_type: str) -> str:
    return QUICK_OBB_MODEL if task_type == "obb" else QUICK_DET_MODEL


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
    data_yaml.write_text(
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
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
        "default_task_type": "detection",
    }


@router.get("/api/annotator/image/{index}")
def annotation_image(index: int) -> dict[str, Any]:
    try:
        _, images_dir, labels_dir, labels_auto_dir, _ = get_batch_paths()
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

    annotations = parse_yolo_label(active_label_path, image_w=image_w, image_h=image_h)

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
        _, _, _, _, classes_path = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    classes = payload.get("classes", [])
    if not isinstance(classes, list):
        raise HTTPException(status_code=400, detail="classes 必须是列表")

    save_annotation_classes(classes_path, [str(x) for x in classes])
    return {"message": "类别已保存", "classes_path": rel(classes_path), "num_classes": len(classes)}


@router.post("/api/annotator/save")
def annotation_save(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        _, images_dir, labels_dir, _, classes_path = get_batch_paths()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    filename = str(payload.get("filename", ""))
    if not filename:
        raise HTTPException(status_code=400, detail="filename 不能为空")

    image_path = images_dir / Path(filename).name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"图片不存在: {filename}")

    image_w, image_h = get_image_size(image_path)
    task_type = str(payload.get("task_type", "detection"))
    if task_type not in {"detection", "obb"}:
        raise HTTPException(status_code=400, detail=f"不支持的任务类型: {task_type}")

    annotations = payload.get("annotations", [])
    if not isinstance(annotations, list):
        raise HTTPException(status_code=400, detail="annotations 必须是列表")

    classes = payload.get("classes", [])
    if isinstance(classes, list):
        save_annotation_classes(classes_path, [str(x) for x in classes])

    label_path = labels_dir / f"{image_path.stem}.txt"
    save_yolo_label(label_path=label_path, annotations=annotations, image_w=image_w, image_h=image_h, task_type=task_type)
    return {"message": "已保存人工标注", "label_path": rel(label_path), "count": len(annotations)}


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
    task_type = str(payload.get("task_type", "detection"))
    if task_type not in {"detection", "obb"}:
        raise HTTPException(status_code=400, detail=f"不支持的任务类型: {task_type}")
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
    task_type = str(payload.get("task_type", "detection"))
    if task_type not in {"detection", "obb"}:
        raise HTTPException(status_code=400, detail=f"不支持的任务类型: {task_type}")
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
    .progress { height:14px; background:#e5e7eb; border-radius:999px; overflow:hidden; margin-top:10px; }
    .progress-inner { height:100%; width:0%; background:#2563eb; transition:width .3s ease; }
    .quick-msg { margin-top:8px; font-size:12px; color:#475569; line-height:1.45; }
    .modal-mask { display:none; position:fixed; inset:0; background:rgba(15,23,42,.45); z-index:10; align-items:center; justify-content:center; }
    .modal { width:420px; background:#fff; border-radius:16px; padding:18px; box-shadow:0 20px 60px rgba(15,23,42,.3); }
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
      </select>

      <div class="section-title">类别</div>
      <div id="classSummary" class="muted">还没有类别。画完第一个框后新增类别。</div>

      <div class="row">
        <button class="green" onclick="startDraw()">W 开始标注</button>
        <button class="orange" onclick="deleteSelected()">Q 删除</button>
        <button class="dark" onclick="saveCurrent(true)">S 保存</button>
      </div>

      <div id="autoNotice" class="auto-notice">
        自动标注待确认：无误按 <b>E</b>，或点击下方按钮复制到 labels。
        <div class="row"><button class="green" onclick="confirmAutoLabel()">E 确认自动标注</button></div>
      </div>

      <div class="hint">
        A/D：上一张/下一张。W：开始标注。Q：删除。S：保存。E：确认自动标注。Esc：取消。<br>
        Detection：W 后拖拽矩形框。OBB：W 后依次点击 4 个角点。鼠标滚轮缩放。<br>
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
let pendingAnnotation = null;
let baseScale = 1, zoom = 1, scale = 1, offsetX = 0, offsetY = 0;
let mouseCanvas = null;
let quickJobTimer = null;
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
  document.getElementById('batch').textContent = 'batch: ' + session.batch_id;
  renderClassSummary();
  resizeCanvas();
  await loadImage(0);
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
  annotations = data.annotations || [];
  labelSource = data.label_source || 'none';
  dirty = false;
  selectedIndex = -1;
  drawingMode = false;
  dragging = false;
  obbPoints = [];
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
  if (ann.type === 'obb') {
    const pts = ann.points || [];
    if (pts.length) {
      ctx.beginPath(); ctx.moveTo(sx(pts[0][0]), sy(pts[0][1]));
      for (let i=1;i<pts.length;i++) ctx.lineTo(sx(pts[i][0]), sy(pts[i][1]));
      if (pts.length === 4) ctx.closePath(); ctx.stroke();
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
  list.innerHTML = annotations.map((a,i)=>`<div class="boxitem ${i===selectedIndex?'active':''}" onclick="selectAnn(${i})">${i+1}. ${a.type} | ${a.class_id}: ${clsName(a.class_id)}</div>`).join('');
}
function selectAnn(i){selectedIndex=i; renderBoxList(); draw();}
function markDirty(){ dirty = true; if (labelSource === 'auto') document.getElementById('saveStatus').textContent = '已修改自动标注，需保存为人工标注'; }
function startDraw(){drawingMode=true; dragging=false; startPt=null; tempPt=null; obbPoints=[]; pendingAnnotation=null; canvas.classList.add('draw-ready'); document.getElementById('saveStatus').textContent='标注中'; draw();}
function stopDraw(){drawingMode=false; dragging=false; startPt=null; tempPt=null; obbPoints=[]; canvas.classList.remove('draw-ready'); draw();}
function onTaskChange(){stopDraw();}

canvas.addEventListener('mousedown', e => {
  if (!drawingMode) return;
  const p = toImgPt(e);
  if (currentTask() === 'detection') { dragging=true; startPt=p; tempPt=p; }
  else {
    obbPoints.push([p.x,p.y]);
    if (obbPoints.length === 4) { pendingAnnotation={type:'obb', class_id:-1, points:obbPoints.map(p=>[p[0],p[1]])}; obbPoints=[]; stopDraw(); openClassModal(pendingAnnotation); }
    draw();
  }
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
async function saveClassesOnly(){ await fetch('/api/annotator/classes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({classes})}); }

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
  const ok = confirm('确认当前 batch 的标注已经审核完成，并整理到 raw_detection/raw_obb？');
  if (!ok) return;
  await saveCurrent(true);
  const res = await fetch('/api/accept-reviewed', { method:'POST' });
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
  if(k==='escape'){ pendingAnnotation=null; closeClassModal(); stopDraw(); }
});
window.addEventListener('resize', resizeCanvas);
init();
</script>
</body>
</html>
'''
