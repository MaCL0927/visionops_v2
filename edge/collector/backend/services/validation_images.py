#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import os
import time
import threading
from pathlib import Path
from typing import Dict, List

from backend.config import DEFAULT_DATASET_NAME
from backend.services.storage import (
    FOLDER_TO_SUBDIR,
    IMAGE_SUFFIXES,
    dataset_path,
    default_dataset_name,
    ensure_dataset_dirs,
    sanitize_dataset_name,
)

VALIDATION_FOLDER = "all"
MAX_VALIDATION_IMAGES = 240
REALTIME_TMP_DIR = "validation_tmp"
REALTIME_IMAGE_NAME = "realtime_latest.jpg"

# 实时检测临时图清理策略：
# - 不再每帧扫描目录，避免 500ms 实时推理时目录扫描带来额外延迟；
# - 每个目录最短间隔 REALTIME_CLEANUP_INTERVAL_SEC 清理一次；
# - 唯一实时帧只保留最近 REALTIME_CLEANUP_KEEP 张，或超过 REALTIME_CLEANUP_MAX_AGE_SEC 即删除；
# - realtime_latest.* 仅作为兼容副本，默认不参与清理。
REALTIME_CLEANUP_KEEP = 30
REALTIME_CLEANUP_MAX_AGE_SEC = 60.0
REALTIME_CLEANUP_INTERVAL_SEC = 5.0
_REALTIME_CLEANUP_LOCK = threading.Lock()
_LAST_REALTIME_CLEANUP_TS: Dict[str, float] = {}


def _image_folder(dataset: str) -> Path:
    ds = sanitize_dataset_name(dataset or default_dataset_name())
    ensure_dataset_dirs(ds)
    return dataset_path(ds) / FOLDER_TO_SUBDIR[VALIDATION_FOLDER]


def list_validation_images(dataset: str = DEFAULT_DATASET_NAME, limit: int = MAX_VALIDATION_IMAGES) -> Dict:
    ds = sanitize_dataset_name(dataset or default_dataset_name())
    folder = _image_folder(ds)
    items: List[Dict] = []

    if folder.exists():
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[: max(1, int(limit))]:
            stat = p.stat()
            image_id = p.name
            items.append({
                "id": image_id,
                "name": p.name,
                "filename": p.name,
                "dataset": ds,
                "folder": VALIDATION_FOLDER,
                "size_bytes": stat.st_size,
                "mtime": int(stat.st_mtime),
                "url": f"/api/validation/image/{p.name}",
            })

    return {
        "ok": True,
        "dataset": ds,
        "folder": VALIDATION_FOLDER,
        "items": items,
    }


def get_validation_image_path(image_id: str, dataset: str = DEFAULT_DATASET_NAME) -> Path:
    ds = sanitize_dataset_name(dataset or default_dataset_name())
    safe_name = Path(image_id or "").name
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("非法图片名称")

    path = (_image_folder(ds) / safe_name).resolve()
    root = _image_folder(ds).resolve()
    if root not in path.parents and path != root:
        raise ValueError("非法图片路径")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"图片不存在: {safe_name}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("只支持 jpg/jpeg/png/webp 图片")
    return path


def _realtime_folder(dataset: str) -> Path:
    ds = sanitize_dataset_name(dataset or default_dataset_name())
    ensure_dataset_dirs(ds)
    folder = dataset_path(ds) / REALTIME_TMP_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """同目录临时文件写入 + os.replace 原子替换。"""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _is_realtime_frame_file(path: Path) -> bool:
    """是否为唯一实时帧文件。

    注意：realtime_latest.* 是兼容副本，不纳入数量统计，避免它影响 keep 计数。
    """
    name = path.name
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and name.startswith("realtime_")
        and not name.startswith("realtime_latest")
    )


def _cleanup_realtime_images(
    folder: Path,
    keep: int = REALTIME_CLEANUP_KEEP,
    max_age_sec: float = REALTIME_CLEANUP_MAX_AGE_SEC,
) -> int:
    """清理实时检测临时图片。

    优化后的策略：
    1. 只清理唯一实时帧 realtime_<timestamp>.jpg，不清理 realtime_latest.* 兼容副本；
    2. 数量优先：超过 keep 的旧帧直接删除；
    3. 时间兜底：超过 max_age_sec 的旧帧也删除；
    4. 返回删除数量，便于后续调试。
    """
    deleted = 0
    try:
        now = time.time()
        files = [p for p in folder.iterdir() if _is_realtime_frame_file(p)]

        def _mtime(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except FileNotFoundError:
                return 0.0
            except Exception:
                return 0.0

        files.sort(key=_mtime, reverse=True)

        keep = max(1, int(keep))
        max_age_sec = max(1.0, float(max_age_sec))

        for idx, path in enumerate(files):
            try:
                age = now - path.stat().st_mtime
                if idx >= keep or age > max_age_sec:
                    path.unlink()
                    deleted += 1
            except FileNotFoundError:
                pass
            except Exception:
                pass
    except Exception:
        pass
    return deleted


def maybe_cleanup_realtime_images(
    folder: Path,
    keep: int = REALTIME_CLEANUP_KEEP,
    max_age_sec: float = REALTIME_CLEANUP_MAX_AGE_SEC,
    interval_sec: float = REALTIME_CLEANUP_INTERVAL_SEC,
) -> int:
    """节流清理实时检测临时图片。

    500ms 实时推理时，如果每帧都 folder.iterdir() 扫描目录，会随着文件变多带来额外延迟。
    这里改成每个目录最多 interval_sec 秒清理一次。
    """
    key = str(folder.resolve())
    now = time.time()
    interval_sec = max(1.0, float(interval_sec))

    with _REALTIME_CLEANUP_LOCK:
        last = float(_LAST_REALTIME_CLEANUP_TS.get(key, 0.0))
        if now - last < interval_sec:
            return 0
        _LAST_REALTIME_CLEANUP_TS[key] = now

    return _cleanup_realtime_images(folder, keep=keep, max_age_sec=max_age_sec)


def _realtime_ext(ext: str) -> str:
    ext = (ext or ".jpg").lower()
    if ext not in IMAGE_SUFFIXES:
        ext = ".jpg"
    if ext == ".jpeg":
        ext = ".jpg"
    return ext


def save_realtime_image_bytes(dataset: str, image_bytes: bytes, ext: str = ".jpg") -> Dict:
    """保存实时检测临时帧。

    v2：不再让前端读取固定的 realtime_latest.jpg。
    每一帧写成唯一文件名 realtime_<timestamp>.jpg，然后把这个唯一 URL 返回给前端。

    原因：在 500ms 或更短间隔下，浏览器可能正在 GET realtime_latest.jpg，
    而下一轮推理又覆盖同一个文件，容易触发：
      RuntimeError: Response content shorter than Content-Length

    唯一文件名可以彻底避免“读同一个文件时被下一帧覆盖”。
    仍然额外维护 realtime_latest.jpg 作为兼容文件，但前端不再使用它。
    """
    ds = sanitize_dataset_name(dataset or default_dataset_name())
    if not image_bytes:
        raise ValueError("未收到实时检测图片数据")
    folder = _realtime_folder(ds)
    ext = _realtime_ext(ext)
    now_ns = time.time_ns()
    now_ms = int(time.time() * 1000)

    filename = f"realtime_{now_ns}{ext}"
    path = (folder / filename).resolve()
    root = folder.resolve()
    if root not in path.parents and path != root:
        raise ValueError("非法实时图片路径")

    # 唯一文件名：写入完成前不会被浏览器读取到。
    _atomic_write_bytes(path, image_bytes)

    # 兼容旧代码或人工调试：保留 latest 副本，但返回 URL 不再指向它。
    latest_name = REALTIME_IMAGE_NAME if ext in {".jpg", ".jpeg"} else f"realtime_latest{ext}"
    latest_path = (folder / latest_name).resolve()
    try:
        _atomic_write_bytes(latest_path, image_bytes)
    except Exception:
        pass

    maybe_cleanup_realtime_images(folder)

    return {
        "id": filename,
        "name": "实时画面",
        "filename": filename,
        "dataset": ds,
        "folder": REALTIME_TMP_DIR,
        "path": str(path),
        "url": f"/api/validation/realtime_image/{filename}?t={now_ms}",
        "mtime": now_ms,
        "size_bytes": path.stat().st_size,
    }

def save_realtime_image_data(dataset: str, image_data: str) -> Dict:
    """保存前端 canvas 截图作为实时检测临时帧。"""
    if not image_data or "," not in image_data:
        raise ValueError("未收到浏览器实时检测截图")
    header, b64_data = image_data.split(",", 1)
    ext = ".jpg"
    if "image/png" in header:
        ext = ".png"
    elif "image/webp" in header:
        ext = ".webp"
    raw = base64.b64decode(b64_data)
    return save_realtime_image_bytes(dataset, raw, ext=ext)


def get_realtime_image_path(filename: str, dataset: str = DEFAULT_DATASET_NAME) -> Path:
    ds = sanitize_dataset_name(dataset or default_dataset_name())
    safe_name = Path(filename or "").name
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("非法实时图片名称")
    folder = _realtime_folder(ds)
    path = (folder / safe_name).resolve()
    root = folder.resolve()
    if root not in path.parents and path != root:
        raise ValueError("非法实时图片路径")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"实时图片不存在: {safe_name}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("只支持 jpg/jpeg/png/webp 图片")
    return path
