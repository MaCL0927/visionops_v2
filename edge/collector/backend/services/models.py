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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _resolve_bundle_path(bundle_dir: Path, value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    p = Path(str(value))
    return p if p.is_absolute() else (bundle_dir / p)


def _pipeline_task_label(task: str) -> str:
    if task == "roi_classification":
        return "ROI 分类双模型"
    return "组合模型"


def _pipeline_label(index: int, task: str) -> str:
    prefix = "最新模型" if index == 0 else "历史模型"
    if task == "roi_classification":
        return f"{prefix} · ROI分类双模型"
    return f"{prefix} · 组合模型"


def _stage_names(stage: Dict[str, Any]) -> List[str]:
    return _list_from_names(stage.get("class_names") or stage.get("names"))


def _stage_num_classes(stage: Dict[str, Any], names: List[str]) -> int:
    value = stage.get("num_classes")
    try:
        n = int(value)
    except Exception:
        n = len(names)
    return n if n > 0 else len(names)


def _pipeline_mtime(bundle_dir: Path, pipeline_path: Path, detector_path: Path | None, classifier_path: Path | None) -> float:
    times = [pipeline_path.stat().st_mtime]
    for p in [detector_path, classifier_path]:
        try:
            if p and p.exists():
                times.append(p.stat().st_mtime)
        except Exception:
            pass
    return max(times)


def _parse_roi_pipeline(bundle_dir: Path) -> Dict[str, Any] | None:
    pipeline_path = bundle_dir / "pipeline.yaml"
    if not pipeline_path.exists() or not pipeline_path.is_file():
        return None

    data = _safe_load_yaml(pipeline_path)
    pipeline_type = str(data.get("pipeline_type") or data.get("task") or "").strip().lower()
    if pipeline_type != "roi_classification":
        return None

    stage1 = data.get("stage1") if isinstance(data.get("stage1"), dict) else {}
    stage2 = data.get("stage2") if isinstance(data.get("stage2"), dict) else {}
    roi_cfg = data.get("roi") if isinstance(data.get("roi"), dict) else {}
    dataset_meta = data.get("dataset") if isinstance(data.get("dataset"), dict) else {}
    deploy_meta = data.get("deploy") if isinstance(data.get("deploy"), dict) else {}

    detector_names = _stage_names(stage1)
    classifier_names = _stage_names(stage2)
    detector_num = _stage_num_classes(stage1, detector_names)
    classifier_num = _stage_num_classes(stage2, classifier_names)

    detector_path = _resolve_bundle_path(bundle_dir, stage1.get("model_path") or "detector.rknn")
    classifier_path = _resolve_bundle_path(bundle_dir, stage2.get("model_path") or "classifier.rknn")
    detector_ok = bool(detector_path and detector_path.exists() and detector_path.is_file())
    classifier_ok = bool(classifier_path and classifier_path.exists() and classifier_path.is_file())
    has_meta = detector_ok and classifier_ok

    size_bytes = 0
    for p in [detector_path, classifier_path, pipeline_path]:
        try:
            if p and p.exists() and p.is_file():
                size_bytes += p.stat().st_size
        except Exception:
            pass

    mtime = _pipeline_mtime(bundle_dir, pipeline_path, detector_path, classifier_path)
    customer_id = dataset_meta.get("customer_id") or ""
    device_id = dataset_meta.get("device_id") or ""
    dataset_id = dataset_meta.get("dataset_id") or ""
    deployed_at = deploy_meta.get("deployed_at") or ""

    return {
        "name": bundle_dir.name,
        "stem": bundle_dir.name,
        "path": str(bundle_dir),
        "pipeline_config": str(pipeline_path),
        "meta_name": "pipeline.yaml",
        "meta_path": str(pipeline_path),
        "has_meta": has_meta,
        "is_pipeline": True,
        "model_kind": "pipeline",
        "size_mb": round(size_bytes / 1024 / 1024, 2),
        "mtime": int(mtime),
        "mtime_text": _format_mtime(mtime),
        "is_current": False,
        "label": "",
        "task": "roi_classification",
        "task_label": _pipeline_task_label("roi_classification"),
        "input_size": stage2.get("input_size") or [224, 224],
        "num_classes": classifier_num,
        "class_names": classifier_names or [str(i) for i in range(classifier_num)],
        "customer_id": customer_id,
        "device_id": device_id,
        "dataset_id": dataset_id,
        "deployed_at": deployed_at,
        "dataset": dataset_meta,
        "deploy": deploy_meta,
        "pipeline": {
            "pipeline_name": data.get("pipeline_name") or bundle_dir.name,
            "pipeline_config": str(pipeline_path),
            "detector_model": str(detector_path) if detector_path else "",
            "classifier_model": str(classifier_path) if classifier_path else "",
            "detector_ok": detector_ok,
            "classifier_ok": classifier_ok,
            "detector": {
                "task": "detection",
                "num_classes": detector_num,
                "class_names": detector_names,
                "input_size": stage1.get("input_size") or [640, 640],
                "conf_threshold": stage1.get("conf_threshold"),
                "nms_threshold": stage1.get("nms_threshold"),
                "target_class_id": stage1.get("target_class_id"),
                "target_class_name": stage1.get("target_class_name"),
            },
            "classifier": {
                "task": "classification",
                "num_classes": classifier_num,
                "class_names": classifier_names,
                "input_size": stage2.get("input_size") or [224, 224],
                "topk": stage2.get("topk"),
            },
            "roi": roi_cfg,
        },
    }


def list_rknn_models() -> Dict[str, Any]:
    models_dir = ensure_models_dir()

    raw_items: List[Dict[str, Any]] = []

    # 1) ROI Classification 双模型 bundle：
    #    /opt/visionops/models/<bundle_name>/pipeline.yaml
    #    在模型选择界面作为“一个模型”显示，避免用户同时选择 detector / classifier 出错。
    for bundle_dir in sorted([p for p in models_dir.iterdir() if p.is_dir()] if models_dir.exists() else [], key=lambda p: p.stat().st_mtime, reverse=True):
        item = _parse_roi_pipeline(bundle_dir)
        if item:
            raw_items.append(item)

    # 2) 兼容原来的单 RKNN 模型：
    #    /opt/visionops/models/<model_name>.rknn + 同名 yaml
    rknn_files = sorted(models_dir.glob("*.rknn"), key=lambda p: p.stat().st_mtime, reverse=True) if models_dir.exists() else []
    for path in rknn_files:
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

        raw_items.append({
            "name": path.name,
            "stem": path.stem,
            "path": str(path),
            "meta_name": meta_path.name,
            "meta_path": str(meta_path),
            "has_meta": has_meta,
            "is_pipeline": False,
            "model_kind": "rknn",
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "mtime": int(stat.st_mtime),
            "mtime_text": _format_mtime(stat.st_mtime),
            "is_current": False,
            "label": "",
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

    items = sorted(raw_items, key=lambda x: int(x.get("mtime") or 0), reverse=True)
    for idx, item in enumerate(items):
        item["is_current"] = idx == 0
        if item.get("is_pipeline"):
            item["label"] = _pipeline_label(idx, str(item.get("task") or ""))
        else:
            item["label"] = _model_label(idx, str(item.get("task") or ""), bool(item.get("has_meta")))

    return {
        "ok": True,
        "models_dir": str(models_dir),
        "model_layout": "mixed_rknn_and_roi_pipeline_bundle",
        "items": items,
    }

