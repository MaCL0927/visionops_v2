#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from backend.config import (
    CAMERA_JPEG_QUALITY,
    CAMERA_PREVIEW_WIDTH,
    CAMERA_RECONNECT_MAX_FAILS,
    CAMERA_SOURCE,
    CAMERA_STREAM_FPS,
    DATA_ROOT,
    DEVICE_ID,
    MODELS_DIR,
    RUNTIME_OVERRIDES_FILE,
    UPLOAD_ENABLED,
    UPLOAD_HOST,
    UPLOAD_PORT,
    UPLOAD_TARGET_DIR,
    UPLOAD_TIMEOUT_SEC,
    UPLOAD_USER,
    VALIDATION_INFER_PORT,
    VALIDATION_REALTIME_INTERVAL_MS,
    VALIDATION_TOPK,
)
from backend.services.settings_schema import VisionOpsRuntimeSettings, model_to_dict


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _migrate_upload_to_vision_box(data: Dict[str, Any]) -> Dict[str, Any]:
    """兼容 v2.0/v2.1 曾经写入的顶层 upload.*，统一迁移到 vision_box.upload。"""
    if not isinstance(data, dict):
        return data
    vision_box = data.setdefault("vision_box", {})
    if not isinstance(vision_box, dict):
        data["vision_box"] = vision_box = {}
    old_upload = data.get("upload")
    new_upload = vision_box.get("upload")
    if isinstance(old_upload, dict):
        if isinstance(new_upload, dict):
            # 视觉盒子下已有 upload 时，优先保留新位置；旧位置只补缺失字段。
            vision_box["upload"] = _deep_merge(old_upload, new_upload)
        else:
            vision_box["upload"] = old_upload
        data.pop("upload", None)
    return data


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_rtsp_from_source(data: Dict[str, Any], source: str) -> None:
    source = str(source or "").strip()
    if source.startswith("rtsp://"):
        data["camera"]["type"] = "rtsp"
        data["camera"]["rtsp"]["url"] = source
    elif source and source not in {"browser", "none", "false", "0"}:
        data["camera"]["type"] = "usb"
        data["camera"]["usb"]["device_node"] = source




def _parse_resolution(resolution: Any):
    """解析 1280x720 / 1280×720，兼容 Python 3.8，不使用 tuple[int, int] 写法。"""
    text = str(resolution or "").lower().replace("×", "x").strip()
    if "x" not in text:
        return 0, 0
    left, right = text.split("x", 1)
    try:
        return max(0, int(float(left.strip()))), max(0, int(float(right.strip())))
    except Exception:
        return 0, 0


def _build_standard_rtsp_url(rtsp: Dict[str, Any]) -> str:
    ip = str(rtsp.get("ip") or "").strip()
    if not ip:
        return str(rtsp.get("url") or "").strip()
    port = _as_int(rtsp.get("port", 554), 554)
    channel = str(rtsp.get("channel") or "102").strip() or "102"
    username = str(rtsp.get("username") or "admin").strip() or "admin"
    password = str(rtsp.get("password") or "").strip()
    # 默认使用 Hikvision 常见小写 channels；如果用户填写了完整 url，会优先保留用户 url。
    return f"rtsp://{username}:{password}@{ip}:{port}/Streaming/channels/{channel}"


def normalize_rtsp_settings_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """确保 RTSP 配置稳定。

    v2.2 调整：优先保留用户填写的完整 rtsp.url。只有 url 为空时，才根据
    IP/端口/通道/账号密码拼接，避免默认密码被保存后反复连接并触发相机锁定。
    """
    if not isinstance(data, dict):
        return data
    camera = data.setdefault("camera", {})
    if not isinstance(camera, dict):
        return data
    if str(camera.get("type") or "rtsp").lower() != "rtsp":
        return data
    rtsp = camera.setdefault("rtsp", {})
    if not isinstance(rtsp, dict):
        camera["rtsp"] = rtsp = {}
    existing_url = str(rtsp.get("url") or "").strip()
    if existing_url.startswith("rtsp://"):
        rtsp["url"] = existing_url
    else:
        standard_url = _build_standard_rtsp_url(rtsp)
        if standard_url.startswith("rtsp://"):
            rtsp["url"] = standard_url
    transport = str(rtsp.get("transport") or "tcp").lower()
    rtsp["transport"] = "udp" if transport == "udp" else "tcp"
    camera["type"] = "rtsp"
    return data


def get_camera_runtime_config(settings: VisionOpsRuntimeSettings = None) -> Dict[str, Any]:
    """给 CameraService 使用的轻量配置。"""
    if settings is None:
        settings = load_settings()
    data = normalize_rtsp_settings_dict(model_to_dict(settings))
    camera = data.get("camera", {}) if isinstance(data, dict) else {}
    common = camera.get("common", {}) if isinstance(camera, dict) else {}
    rtsp = camera.get("rtsp", {}) if isinstance(camera, dict) else {}
    camera_type = str(camera.get("type") or "rtsp").lower()

    if camera_type == "rtsp":
        source = str(rtsp.get("url") or _build_standard_rtsp_url(rtsp) or "").strip()
    elif camera_type == "usb":
        usb = camera.get("usb", {}) if isinstance(camera, dict) else {}
        source = str(usb.get("device_node") or CAMERA_SOURCE or "").strip()
    else:
        source = str(CAMERA_SOURCE or "").strip()

    width, height = _parse_resolution(common.get("resolution", ""))
    return {
        "enabled": bool(source) and source.lower() not in {"browser", "none", "false", "0"},
        "type": camera_type,
        "source": source,
        "rtsp_transport": str(rtsp.get("transport") or "tcp").lower(),
        "stream_fps": _as_float(common.get("fps", CAMERA_STREAM_FPS), 6.0),
        "preview_width": _as_int(common.get("preview_width", CAMERA_PREVIEW_WIDTH), 960),
        "jpeg_quality": _as_int(common.get("jpeg_quality", CAMERA_JPEG_QUALITY), 75),
        "reconnect_max_fails": _as_int(common.get("reconnect_max_fails", CAMERA_RECONNECT_MAX_FAILS), 30),
        "resolution_width": width,
        "resolution_height": height,
    }


def get_upload_runtime_config(settings: VisionOpsRuntimeSettings = None) -> Dict[str, Any]:
    """给打包上传流程使用的即时上传配置，保存后无需重启 Collector。"""
    if settings is None:
        settings = load_settings()
    data = _migrate_upload_to_vision_box(model_to_dict(settings))
    upload = data.get("vision_box", {}).get("upload", {}) if isinstance(data, dict) else {}
    if not isinstance(upload, dict):
        upload = {}
    return {
        "enabled": bool(upload.get("enabled", True)),
        "host": str(upload.get("host") or "").strip(),
        "user": str(upload.get("user") or "pc").strip(),
        "port": _as_int(upload.get("port", 22), 22),
        "target_dir": str(upload.get("target_dir") or "/home/pc/桌面/visionops_v2/data/incoming").strip(),
        "timeout_sec": _as_int(upload.get("timeout_sec", 120), 120),
    }



def _clamp_int(value: Any, default: int, min_value: int = None, max_value: int = None) -> int:
    result = _as_int(value, default)
    if min_value is not None:
        result = max(min_value, result)
    if max_value is not None:
        result = min(max_value, result)
    return result


def _clamp_float(value: Any, default: float, min_value: float = None, max_value: float = None) -> float:
    result = _as_float(value, default)
    if min_value is not None:
        result = max(min_value, result)
    if max_value is not None:
        result = min(max_value, result)
    return result


def _safe_task_name(task: Any) -> str:
    name = str(task or "").strip().lower()
    if name in {"cls", "classify", "classification", "image_classification"}:
        return "classification"
    if name in {"det", "detect", "detection", "object_detection"}:
        return "detection"
    if name in {"obb", "obb_detection", "oriented_detection", "rotated_detection", "yolo_obb", "yolov8_obb"}:
        return "obb_detection"
    if name in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg", "yolov8_seg", "mask_segmentation"}:
        return "segmentation"
    return name or "detection"


def get_algorithm_runtime_config(settings: VisionOpsRuntimeSettings = None) -> Dict[str, Any]:
    """读取设置界面的算法运行时配置。

    返回的是轻量 dict，可被前端、生产模式和验证推理服务即时读取。
    """
    if settings is None:
        settings = load_settings()
    data = model_to_dict(settings).get("algorithm", {})
    common = data.get("common", {}) if isinstance(data, dict) else {}
    classification = data.get("classification", {}) if isinstance(data, dict) else {}
    detection = data.get("detection", {}) if isinstance(data, dict) else {}
    obb = data.get("obb_detection", {}) if isinstance(data, dict) else {}
    segmentation = data.get("segmentation", {}) if isinstance(data, dict) else {}

    fps_limit = _clamp_int(common.get("production_fps_limit", 5), 5, 1, 30)
    interval_from_fps = int(round(1000.0 / max(1, fps_limit)))
    # 界面当前暴露的是“生产 FPS 上限”，因此生产检测间隔由 FPS 自动换算。
    production_interval = interval_from_fps

    return {
        "common": {
            "model_mode": str(common.get("model_mode") or "auto"),
            "input_size": str(common.get("input_size") or "640x640"),
            "npu_core": str(common.get("npu_core") or "auto"),
            "realtime_interval_ms": _clamp_int(common.get("realtime_interval_ms", 1000), 1000, 100, 60000),
            "production_fps_limit": fps_limit,
            "production_detect_interval_ms": _clamp_int(production_interval, interval_from_fps, 100, 60000),
            "warmup_runs": _clamp_int(common.get("warmup_runs", 3), 3, 0, 20),
            "label_display": str(common.get("label_display") or "class_conf"),
            "max_results": _clamp_int(common.get("max_results", 100), 100, 1, 1000),
            "log_level": str(common.get("log_level") or "INFO").upper(),
        },
        "classification": {
            "topk": _clamp_int(classification.get("topk", 5), 5, 1, 100),
            "score_threshold": _clamp_float(classification.get("score_threshold", 0.5), 0.5, 0.0, 1.0),
            "low_confidence_policy": str(classification.get("low_confidence_policy") or "review"),
        },
        "detection": {
            "conf_threshold": _clamp_float(detection.get("conf_threshold", 0.25), 0.25, 0.0, 1.0),
            "nms_threshold": _clamp_float(detection.get("nms_threshold", 0.45), 0.45, 0.0, 1.0),
            "show_center": bool(detection.get("show_center", True)),
            "max_detections": _clamp_int(detection.get("max_detections", 100), 100, 1, 1000),
        },
        "obb_detection": {
            "conf_threshold": _clamp_float(obb.get("conf_threshold", 0.25), 0.25, 0.0, 1.0),
            "nms_threshold": _clamp_float(obb.get("nms_threshold", 0.45), 0.45, 0.0, 1.0),
            "nms_mode": str(obb.get("nms_mode") or "rotated"),
            "show_angle": bool(obb.get("show_angle", True)),
            "show_polygon": bool(obb.get("show_polygon", True)),
        },
        "segmentation": {
            "conf_threshold": _clamp_float(segmentation.get("conf_threshold", 0.25), 0.25, 0.0, 1.0),
            "nms_threshold": _clamp_float(segmentation.get("nms_threshold", 0.45), 0.45, 0.0, 1.0),
            "mask_threshold": _clamp_float(segmentation.get("mask_threshold", 0.5), 0.5, 0.0, 1.0),
            "mask_alpha": _clamp_float(segmentation.get("mask_alpha", 0.35), 0.35, 0.0, 1.0),
            "show_mask": bool(segmentation.get("show_mask", True)),
            "show_box": bool(segmentation.get("show_box", True)),
            "show_mode": str(segmentation.get("show_mode") or "mask_box"),
        },
    }


def get_algorithm_effective_config(task: Any = None, model_meta: Dict[str, Any] = None, settings: VisionOpsRuntimeSettings = None) -> Dict[str, Any]:
    """合成某个任务实际要传给 engine.py 的算法参数。"""
    algo = get_algorithm_runtime_config(settings)
    common = dict(algo.get("common", {}))
    model_meta = model_meta or {}
    task_name = _safe_task_name(task or model_meta.get("task") or model_meta.get("model", {}).get("task"))

    result = {
        "task": task_name,
        "npu_core": common["npu_core"],
        "warmup_runs": common["warmup_runs"],
        "realtime_interval_ms": common["realtime_interval_ms"],
        "production_detect_interval_ms": common["production_detect_interval_ms"],
        "production_fps_limit": common["production_fps_limit"],
        "max_results": common["max_results"],
        "label_display": common["label_display"],
    }

    if task_name == "classification":
        result.update(algo["classification"])
        result.setdefault("conf_threshold", 0.25)
        result.setdefault("nms_threshold", 0.45)
        result.setdefault("mask_threshold", 0.5)
    elif task_name == "obb_detection":
        result.update(algo["obb_detection"])
        result.setdefault("topk", algo["classification"]["topk"])
        result.setdefault("mask_threshold", 0.5)
    elif task_name == "segmentation":
        result.update(algo["segmentation"])
        result.setdefault("topk", algo["classification"]["topk"])
    else:
        result.update(algo["detection"])
        result.setdefault("topk", algo["classification"]["topk"])
        result.setdefault("mask_threshold", 0.5)

    result["signature"] = make_algorithm_signature(result)
    return result


def make_algorithm_signature(effective: Dict[str, Any]) -> str:
    keys = [
        "task", "npu_core", "warmup_runs", "topk", "score_threshold",
        "conf_threshold", "nms_threshold", "mask_threshold", "nms_mode",
    ]
    payload = {key: effective.get(key) for key in keys if key in effective}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def write_runtime_algorithm_env(settings: VisionOpsRuntimeSettings = None) -> Path:
    """写 edge/runtime/runtime_algorithm.env，供 switch_model.sh 选择模型时读取。"""
    if settings is None:
        settings = load_settings()
    algo = get_algorithm_runtime_config(settings)
    runtime_dir = get_settings_path().parent
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / "runtime_algorithm.env"

    common = algo["common"]
    cls = algo["classification"]
    det = algo["detection"]
    obb = algo["obb_detection"]
    seg = algo["segmentation"]
    lines = [
        "# Auto generated by VisionOps Collector settings v2.2",
        f"VISIONOPS_RUNTIME_NPU_CORE={common['npu_core']}",
        f"VISIONOPS_RUNTIME_WARMUP_RUNS={common['warmup_runs']}",
        f"VISIONOPS_RUNTIME_REALTIME_INTERVAL_MS={common['realtime_interval_ms']}",
        f"VISIONOPS_RUNTIME_PRODUCTION_DETECT_INTERVAL_MS={common['production_detect_interval_ms']}",
        f"VISIONOPS_RUNTIME_MAX_RESULTS={common['max_results']}",
        f"VISIONOPS_RUNTIME_CLASSIFICATION_TOPK={cls['topk']}",
        f"VISIONOPS_RUNTIME_CLASSIFICATION_SCORE_THRESHOLD={cls['score_threshold']}",
        f"VISIONOPS_RUNTIME_DETECTION_CONF_THRESHOLD={det['conf_threshold']}",
        f"VISIONOPS_RUNTIME_DETECTION_NMS_THRESHOLD={det['nms_threshold']}",
        f"VISIONOPS_RUNTIME_OBB_CONF_THRESHOLD={obb['conf_threshold']}",
        f"VISIONOPS_RUNTIME_OBB_NMS_THRESHOLD={obb['nms_threshold']}",
        f"VISIONOPS_RUNTIME_OBB_NMS_MODE={obb['nms_mode']}",
        f"VISIONOPS_RUNTIME_SEGMENTATION_CONF_THRESHOLD={seg['conf_threshold']}",
        f"VISIONOPS_RUNTIME_SEGMENTATION_NMS_THRESHOLD={seg['nms_threshold']}",
        f"VISIONOPS_RUNTIME_SEGMENTATION_MASK_THRESHOLD={seg['mask_threshold']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def build_default_settings() -> VisionOpsRuntimeSettings:
    """从当前运行配置生成默认回显值。"""
    defaults = model_to_dict(VisionOpsRuntimeSettings())

    # 注意：这里不强制把 schema 默认 RTSP 改成 browser，避免“恢复默认”后现场画面变成笔记本摄像头。
    _extract_rtsp_from_source(defaults, CAMERA_SOURCE)
    defaults["camera"]["common"]["fps"] = _as_int(CAMERA_STREAM_FPS, 6)
    defaults["camera"]["common"]["preview_width"] = _as_int(CAMERA_PREVIEW_WIDTH, 960)
    defaults["camera"]["common"]["jpeg_quality"] = _as_int(CAMERA_JPEG_QUALITY, 75)
    defaults["camera"]["common"]["reconnect_max_fails"] = _as_int(CAMERA_RECONNECT_MAX_FAILS, 30)

    defaults["vision_box"]["device_id"] = str(DEVICE_ID)
    defaults["vision_box"]["models_dir"] = str(MODELS_DIR)
    defaults["vision_box"]["data_dir"] = str(DATA_ROOT)
    defaults["vision_box"]["validation_port"] = _as_int(VALIDATION_INFER_PORT, 8082)

    defaults["algorithm"]["common"]["realtime_interval_ms"] = _as_int(VALIDATION_REALTIME_INTERVAL_MS, 1000)
    defaults["algorithm"]["classification"]["topk"] = _as_int(VALIDATION_TOPK, 5)

    defaults.setdefault("vision_box", {}).setdefault("upload", {})
    defaults["vision_box"]["upload"] = {
        "enabled": bool(UPLOAD_ENABLED),
        "host": str(UPLOAD_HOST),
        "user": str(UPLOAD_USER),
        "port": _as_int(UPLOAD_PORT, 22),
        "target_dir": str(UPLOAD_TARGET_DIR),
        "timeout_sec": _as_int(UPLOAD_TIMEOUT_SEC, 120),
    }

    defaults = _migrate_upload_to_vision_box(normalize_rtsp_settings_dict(defaults))
    defaults["version"] = "2.2"
    return VisionOpsRuntimeSettings(**defaults)


def get_settings_path() -> Path:
    return RUNTIME_OVERRIDES_FILE


def load_raw_overrides() -> Dict[str, Any]:
    path = get_settings_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return _migrate_upload_to_vision_box(data) if isinstance(data, dict) else {}


def load_settings() -> VisionOpsRuntimeSettings:
    merged = _migrate_upload_to_vision_box(_deep_merge(model_to_dict(build_default_settings()), load_raw_overrides()))
    return VisionOpsRuntimeSettings(**merged)


def save_settings(settings: VisionOpsRuntimeSettings) -> VisionOpsRuntimeSettings:
    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _migrate_upload_to_vision_box(normalize_rtsp_settings_dict(model_to_dict(settings)))
    data["version"] = "2.2"

    fd, tmp_name = tempfile.mkstemp(prefix=".runtime_overrides.", suffix=".yaml", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        if path.exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    saved = load_settings()
    try:
        write_runtime_algorithm_env(saved)
    except Exception:
        pass
    return saved


def reset_settings() -> VisionOpsRuntimeSettings:
    """恢复默认值。v2.0 不删除配置，而是写入默认配置，避免摄像头回退到 browser。"""
    default_settings = build_default_settings()
    return save_settings(default_settings)
