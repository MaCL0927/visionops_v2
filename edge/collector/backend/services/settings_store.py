#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    password = str(rtsp.get("password") or "password").strip()
    return f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}"


def normalize_rtsp_settings_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """确保 RTSP 类型下 url 与 IP/端口/通道/账号密码保持一致。"""
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
    # v2.1 当前只正式支持海康/大华常见 RTSP 拼接；用户自定义 URL 仍可直接写入该字段。
    # 只要 IP 存在，就以后端拼接结果为准，避免改了密码或通道但 URL 仍是旧值。
    standard_url = _build_standard_rtsp_url(rtsp)
    if standard_url:
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
    defaults["version"] = "2.1.1"
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
    data["version"] = "2.1.1"

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
    return load_settings()


def reset_settings() -> VisionOpsRuntimeSettings:
    """恢复默认值。v2.0 不删除配置，而是写入默认配置，避免摄像头回退到 browser。"""
    default_settings = build_default_settings()
    return save_settings(default_settings)
