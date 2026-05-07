#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from backend.config import MODELS_DIR
from backend.services.settings_store import get_effective_models_dir


def ensure_models_dir() -> Path:
    models_dir = get_effective_models_dir()
    # v2.3.0：模型目录动态读取，但不存在/不可读时 settings_store 会回退到 MODELS_DIR。
    return models_dir


def _format_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _list_from_names(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        def key_fn(k: Any):
            s = str(k)
            return (0, int(s)) if s.isdigit() else (1, s)
        return [str(value[k]) for k in sorted(value.keys(), key=key_fn)]
    return []


def _parse_meta(meta_path: Path) -> Dict[str, Any]:
    data = _safe_load_yaml(meta_path)
    model_meta = data.get("model") if isinstance(data.get("model"), dict) else {}
    dataset_meta = data.get("dataset") if isinstance(data.get("dataset"), dict) else {}
    deploy_meta = data.get("deploy") if isinstance(data.get("deploy"), dict) else {}

    task = str(data.get("task") or model_meta.get("task") or "").strip().lower()
    if task in {"obb", "oriented_detection", "rotated_detection", "yolo_obb", "yolov8_obb"}:
        task = "obb_detection"
    if task in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg", "yolov8_seg", "mask_segmentation"}:
        task = "segmentation"
    if task not in {"classification", "detection", "obb_detection", "segmentation"}:
        task = ""

    input_size = data.get("input_size") or model_meta.get("input_size") or []
    try:
        input_size = [int(input_size[0]), int(input_size[1])]
    except Exception:
        input_size = [224, 224] if task == "classification" else ([640, 640] if task in {"detection", "obb_detection", "segmentation"} else [])

    class_names = _list_from_names(data.get("class_names") or model_meta.get("class_names"))
    try:
        num_classes = int(data.get("num_classes") or model_meta.get("num_classes") or len(class_names))
    except Exception:
        num_classes = len(class_names)

    if not class_names and num_classes > 0:
        class_names = [str(i) for i in range(num_classes)]

    return {
        "raw": data,
        "task": task,
        "task_label": "分类模型" if task == "classification" else ("检测模型" if task == "detection" else ("旋转框检测模型" if task == "obb_detection" else ("实例分割模型" if task == "segmentation" else "未知任务"))),
        "input_size": input_size,
        "num_classes": num_classes,
        "class_names": class_names,
        "dataset": dataset_meta,
        "deploy": deploy_meta,
        "model": model_meta,
        "schema_version": data.get("schema_version"),
        "topk": int(data.get("topk") or model_meta.get("topk") or min(max(num_classes, 1), 5)) if task == "classification" else None,
        "conf_threshold": data.get("conf_threshold") or model_meta.get("conf_threshold"),
        "nms_threshold": data.get("nms_threshold") or model_meta.get("nms_threshold"),
    }


def _model_label(index: int, task: str, has_meta: bool) -> str:
    if not has_meta:
        return "配置缺失"
    prefix = "最新模型" if index == 0 else "历史模型"
    task_text = "分类" if task == "classification" else ("检测" if task == "detection" else ("旋转框检测" if task == "obb_detection" else ("实例分割" if task == "segmentation" else "未知")))
    return f"{prefix} · {task_text}"


def list_rknn_models() -> Dict[str, Any]:
    models_dir = ensure_models_dir()
    rknn_files = sorted(models_dir.glob("*.rknn"), key=lambda p: p.stat().st_mtime, reverse=True)

    items: List[Dict[str, Any]] = []
    for idx, path in enumerate(rknn_files):
        stat = path.stat()
        meta_path = path.with_suffix(".yaml")
        has_meta = meta_path.exists() and meta_path.is_file()
        meta = _parse_meta(meta_path) if has_meta else {
            "task": "",
            "task_label": "配置缺失",
            "input_size": [],
            "num_classes": 0,
            "class_names": [],
            "dataset": {},
            "deploy": {},
            "model": {},
        }
        dataset = meta.get("dataset") or {}
        deploy = meta.get("deploy") or {}
        task = meta.get("task") or ""
        customer_id = dataset.get("customer_id") or ""
        device_id = dataset.get("device_id") or ""
        dataset_id = dataset.get("dataset_id") or ""
        deployed_at = deploy.get("deployed_at") or ""

        items.append({
            "name": path.name,
            "stem": path.stem,
            "path": str(path),
            "meta_name": meta_path.name,
            "meta_path": str(meta_path),
            "has_meta": has_meta,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "mtime": int(stat.st_mtime),
            "mtime_text": _format_mtime(stat.st_mtime),
            "is_current": idx == 0,
            "label": _model_label(idx, task, has_meta),
            "task": task,
            "task_label": meta.get("task_label"),
            "input_size": meta.get("input_size", []),
            "num_classes": meta.get("num_classes", 0),
            "class_names": meta.get("class_names", []),
            "customer_id": customer_id,
            "device_id": device_id,
            "dataset_id": dataset_id,
            "deployed_at": deployed_at,
            "dataset": dataset,
            "deploy": deploy,
        })

    return {
        "ok": True,
        "models_dir": str(models_dir),
        "model_layout": "versioned_rknn_with_same_name_yaml",
        "items": items,
    }
