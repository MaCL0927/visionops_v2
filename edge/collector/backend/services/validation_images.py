#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import time
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


def save_realtime_image_bytes(dataset: str, image_bytes: bytes, ext: str = ".jpg") -> Dict:
    """保存实时检测临时帧。只覆盖 validation_tmp/realtime_latest.jpg，不写入 all_images。"""
    ds = sanitize_dataset_name(dataset or default_dataset_name())
    if not image_bytes:
        raise ValueError("未收到实时检测图片数据")
    folder = _realtime_folder(ds)
    filename = REALTIME_IMAGE_NAME if ext.lower() in {".jpg", ".jpeg"} else f"realtime_latest{ext}"
    path = (folder / filename).resolve()
    root = folder.resolve()
    if root not in path.parents and path != root:
        raise ValueError("非法实时图片路径")
    path.write_bytes(image_bytes)
    now = int(time.time() * 1000)
    return {
        "id": filename,
        "name": "实时画面",
        "filename": filename,
        "dataset": ds,
        "folder": REALTIME_TMP_DIR,
        "path": str(path),
        "url": f"/api/validation/realtime_image/{filename}?t={now}",
        "mtime": now,
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
