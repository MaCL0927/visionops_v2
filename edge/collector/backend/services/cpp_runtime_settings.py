#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C++ inference runtime settings helpers.

v0.7.3.1: keep the old Python camera settings intact, but add an independent
C++ camera settings path for the C++ inference service.

Design:
- Persistent editable settings are stored under runtime_overrides.yaml:
    cpp_inference:
      camera: {...}
- Effective process settings are written to edge/runtime/cpp.env.
- Applying settings restarts visionops-inference-cpp.service so the C++ binary
  reloads cpp.env through its systemd start script.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML should exist in VisionOps venv
    yaml = None

try:
    from backend.config import CPP_INFERENCE_URL
except Exception:  # pragma: no cover
    CPP_INFERENCE_URL = "http://127.0.0.1:18080"


INSTALL_DIR = Path(os.environ.get("VISIONOPS_INSTALL_DIR", "/opt/visionops"))
RUNTIME_DIR = Path(os.environ.get("VISIONOPS_RUNTIME_DIR", str(INSTALL_DIR / "edge" / "runtime")))
CPP_ENV_PATH = Path(os.environ.get("VISIONOPS_CPP_ENV_PATH", str(RUNTIME_DIR / "cpp.env")))
RUNTIME_SETTINGS_PATH = Path(
    os.environ.get("VISIONOPS_RUNTIME_SETTINGS_PATH", str(RUNTIME_DIR / "runtime_overrides.yaml"))
)
CPP_SERVICE_NAME = os.environ.get("VISIONOPS_CPP_SERVICE_NAME", "visionops-inference-cpp")


_CPP_ENV_TO_SETTING = {
    "VISIONOPS_CPP_CAMERA_TYPE": "camera_type",
    "VISIONOPS_CPP_CAMERA_SOURCE": "camera_source",
    "VISIONOPS_CAMERA_SOURCE": "camera_source",
    "VISIONOPS_CPP_CAMERA_WIDTH": "camera_width",
    "VISIONOPS_CPP_CAMERA_HEIGHT": "camera_height",
    "VISIONOPS_CPP_CAMERA_FPS": "camera_fps",
    "VISIONOPS_CPP_CAMERA_BUFFER_SIZE": "camera_buffer_size",
    "VISIONOPS_CPP_CAMERA_FOURCC": "camera_fourcc",
    "VISIONOPS_CPP_STREAM_BACKEND": "stream_backend",
    "VISIONOPS_CPP_STREAM_CODEC": "stream_codec",
    "VISIONOPS_CPP_STREAM_AUTO_START": "stream_auto_start",
    "VISIONOPS_CPP_CAMERA_READ_FPS": "camera_read_fps",
    "VISIONOPS_CPP_DETECT_FPS": "detect_fps",
    "VISIONOPS_CPP_SNAPSHOT_FPS": "snapshot_fps",
    "VISIONOPS_CPP_ENABLE_SNAPSHOT": "enable_snapshot",
    "VISIONOPS_CPP_ENABLE_ANNOTATED": "enable_annotated",
    "VISIONOPS_CPP_RTSP_TRANSPORT": "rtsp_transport",
    "VISIONOPS_CPP_RTSP_TIMEOUT_MS": "rtsp_timeout_ms",
    "VISIONOPS_CPP_GST_LATENCY_MS": "gst_latency_ms",
    "VISIONOPS_CPP_QUIET_FFMPEG_LOG": "quiet_ffmpeg_log",
}

_SETTING_TO_CPP_ENV = {
    "camera_type": "VISIONOPS_CPP_CAMERA_TYPE",
    "camera_source": "VISIONOPS_CPP_CAMERA_SOURCE",
    "camera_width": "VISIONOPS_CPP_CAMERA_WIDTH",
    "camera_height": "VISIONOPS_CPP_CAMERA_HEIGHT",
    "camera_fps": "VISIONOPS_CPP_CAMERA_FPS",
    "camera_buffer_size": "VISIONOPS_CPP_CAMERA_BUFFER_SIZE",
    "camera_fourcc": "VISIONOPS_CPP_CAMERA_FOURCC",
    "stream_backend": "VISIONOPS_CPP_STREAM_BACKEND",
    "stream_codec": "VISIONOPS_CPP_STREAM_CODEC",
    "stream_auto_start": "VISIONOPS_CPP_STREAM_AUTO_START",
    "camera_read_fps": "VISIONOPS_CPP_CAMERA_READ_FPS",
    "detect_fps": "VISIONOPS_CPP_DETECT_FPS",
    "snapshot_fps": "VISIONOPS_CPP_SNAPSHOT_FPS",
    "enable_snapshot": "VISIONOPS_CPP_ENABLE_SNAPSHOT",
    "enable_annotated": "VISIONOPS_CPP_ENABLE_ANNOTATED",
    "rtsp_transport": "VISIONOPS_CPP_RTSP_TRANSPORT",
    "rtsp_timeout_ms": "VISIONOPS_CPP_RTSP_TIMEOUT_MS",
    "gst_latency_ms": "VISIONOPS_CPP_GST_LATENCY_MS",
    "quiet_ffmpeg_log": "VISIONOPS_CPP_QUIET_FFMPEG_LOG",
}

_NUMERIC_KEYS = {
    "camera_width",
    "camera_height",
    "camera_fps",
    "camera_buffer_size",
    "camera_read_fps",
    "detect_fps",
    "snapshot_fps",
    "rtsp_timeout_ms",
    "gst_latency_ms",
}
_BOOL_KEYS = {
    "stream_auto_start",
    "enable_snapshot",
    "enable_annotated",
    "quiet_ffmpeg_log",
}

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "camera_type": "auto",            # auto | rtsp | usb
    "camera_source": "",             # rtsp://... | /dev/video7 | /dev/v4l/by-id/...
    "camera_width": 0,
    "camera_height": 0,
    "camera_fps": 0,
    "camera_buffer_size": 1,
    "camera_fourcc": "",             # YUYV | MJPG | empty
    "stream_backend": "opencv",      # opencv | gst-mpp
    "stream_codec": "h264",
    "stream_auto_start": False,
    "camera_read_fps": 10,
    "detect_fps": 10,
    "snapshot_fps": 1,
    "enable_snapshot": True,
    "enable_annotated": True,
    "rtsp_transport": "tcp",
    "rtsp_timeout_ms": 5000,
    "gst_latency_ms": 100,
    "quiet_ffmpeg_log": True,
}


class CppRuntimeSettingsError(RuntimeError):
    """Raised when C++ runtime settings cannot be read/applied."""


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value).strip()))
    except Exception:
        return default


def _normalize_camera_type(value: Any) -> str:
    s = str(value or "auto").strip().lower()
    return s if s in {"auto", "rtsp", "usb"} else "auto"


def _normalize_stream_backend(value: Any) -> str:
    s = str(value or "opencv").strip().lower()
    return s if s in {"opencv", "gst-mpp"} else "opencv"


def _normalize_fourcc(value: Any) -> str:
    s = str(value or "").strip().upper()
    # OpenCV FOURCC is 4 characters. Accept common user typo MJPEG -> MJPG.
    if s == "MJPEG":
        s = "MJPG"
    if not s:
        return ""
    return s if len(s) == 4 else ""


def _normalize_settings(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    data = dict(_DEFAULT_SETTINGS)
    if raw:
        data.update({k: v for k, v in dict(raw).items() if v is not None})

    data["camera_type"] = _normalize_camera_type(data.get("camera_type"))
    data["camera_source"] = str(data.get("camera_source") or "").strip()
    data["stream_backend"] = _normalize_stream_backend(data.get("stream_backend"))
    data["stream_codec"] = str(data.get("stream_codec") or "h264").strip().lower() or "h264"
    data["rtsp_transport"] = str(data.get("rtsp_transport") or "tcp").strip().lower()
    if data["rtsp_transport"] not in {"tcp", "udp"}:
        data["rtsp_transport"] = "tcp"
    data["camera_fourcc"] = _normalize_fourcc(data.get("camera_fourcc"))

    for key in _NUMERIC_KEYS:
        default = int(_DEFAULT_SETTINGS.get(key, 0))
        data[key] = _as_int(data.get(key), default)

    if data["camera_buffer_size"] <= 0:
        data["camera_buffer_size"] = 1

    for key in _BOOL_KEYS:
        data[key] = _as_bool(data.get(key), bool(_DEFAULT_SETTINGS.get(key, False)))

    # Safety: USB must use OpenCV in current v0.7.x. gst-mpp is RTSP-only.
    if data["camera_type"] == "usb" and data["stream_backend"] == "gst-mpp":
        data["stream_backend"] = "opencv"

    return data


def _validate_settings(settings: Mapping[str, Any]) -> None:
    camera_type = str(settings.get("camera_type") or "auto")
    source = str(settings.get("camera_source") or "").strip()
    if not source:
        raise CppRuntimeSettingsError("camera_source 不能为空")

    if camera_type == "usb":
        if not (source.startswith("/dev/video") or source.startswith("/dev/v4l/") or source.isdigit()):
            raise CppRuntimeSettingsError("USB 相机源应为 /dev/videoX、/dev/v4l/... 或数字索引")
        if str(settings.get("stream_backend") or "opencv") != "opencv":
            raise CppRuntimeSettingsError("USB C++ 相机当前只支持 stream_backend=opencv")

    if camera_type == "rtsp":
        if not source.lower().startswith("rtsp://"):
            raise CppRuntimeSettingsError("RTSP 相机源应以 rtsp:// 开头")


def _parse_env_line(line: str) -> Optional[tuple[str, str]]:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    try:
        parts = shlex.split(line, comments=False, posix=True)
        if parts:
            token = parts[0]
        else:
            token = line
    except Exception:
        token = line
    if "=" not in token:
        return None
    key, value = token.split("=", 1)
    return key.strip(), value


def read_cpp_env(path: Path = CPP_ENV_PATH) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = _parse_env_line(line)
        if parsed:
            key, value = parsed
            env[key] = value
    return env


def _env_to_settings(env: Mapping[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for env_key, setting_key in _CPP_ENV_TO_SETTING.items():
        if env_key in env and setting_key not in out:
            out[setting_key] = env[env_key]
    return _normalize_settings(out)


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_yaml_file(path: Path, data: Mapping[str, Any]) -> None:
    if yaml is None:
        raise CppRuntimeSettingsError("当前 Python 环境未安装 PyYAML，无法写入 runtime_overrides.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(data), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def get_saved_cpp_camera_settings() -> Dict[str, Any]:
    data = _load_yaml_file(RUNTIME_SETTINGS_PATH)
    cpp = data.get("cpp_inference") if isinstance(data.get("cpp_inference"), dict) else {}
    camera = cpp.get("camera") if isinstance(cpp.get("camera"), dict) else {}
    return _normalize_settings(camera)


def save_cpp_camera_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_settings(settings)
    _validate_settings(normalized)

    data = _load_yaml_file(RUNTIME_SETTINGS_PATH)
    cpp = data.setdefault("cpp_inference", {})
    if not isinstance(cpp, dict):
        cpp = {}
        data["cpp_inference"] = cpp
    cpp["camera"] = normalized
    _write_yaml_file(RUNTIME_SETTINGS_PATH, data)
    return normalized


def get_effective_cpp_env_settings() -> Dict[str, Any]:
    return _env_to_settings(read_cpp_env())


def get_cpp_settings_payload() -> Dict[str, Any]:
    env = read_cpp_env()
    effective = _env_to_settings(env)
    saved = get_saved_cpp_camera_settings()
    # If saved settings are still default and env exists, expose env as editable default.
    editable = saved if saved.get("camera_source") else effective
    return {
        "ok": True,
        "settings": editable,
        "saved_settings": saved,
        "effective_settings": effective,
        "cpp_env_path": str(CPP_ENV_PATH),
        "runtime_settings_path": str(RUNTIME_SETTINGS_PATH),
        "service_name": CPP_SERVICE_NAME,
    }


def _format_env_value(value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif value is None:
        value = ""
    else:
        value = str(value)
    return shlex.quote(value)




# -----------------------------------------------------------------------------
# v0.8.1 C++ current model config helpers
# -----------------------------------------------------------------------------

_CPP_MODEL_ENV_KEYS = {
    "VISIONOPS_CPP_BIN",
    "VISIONOPS_CPP_MODEL_PATH",
    "VISIONOPS_CPP_CLASS_NAMES_FILE",
    "VISIONOPS_CPP_TASK",
    "VISIONOPS_CPP_PIPELINE_CONFIG",
    "VISIONOPS_CPP_PORT",
    "VISIONOPS_CPP_NPU_CORE",
    "VISIONOPS_CPP_NUM_CLASSES",
    "VISIONOPS_CPP_INPUT_SIZE",
    "VISIONOPS_CPP_CONF_THRESHOLD",
    "VISIONOPS_CPP_NMS_THRESHOLD",
    "VISIONOPS_CPP_TOPK",
    "VISIONOPS_CPP_MAX_DET",
    "VISIONOPS_CPP_OUTPUT_MODE",
    "VISIONOPS_CPP_PREPROCESS_BACKEND",
    "VISIONOPS_CPP_RGA_MODE",
}


def _normalize_model_task(value: Any) -> str:
    task = str(value or "").strip().lower()
    aliases = {
        "cls": "classification",
        "classify": "classification",
        "classification": "classification",
        "det": "detection",
        "detect": "detection",
        "detection": "detection",
        "obb": "obb_detection",
        "oriented_detection": "obb_detection",
        "rotated_detection": "obb_detection",
        "yolo_obb": "obb_detection",
        "yolov8_obb": "obb_detection",
        "seg": "segmentation",
        "segment": "segmentation",
        "segmentation": "segmentation",
        "instance_segmentation": "segmentation",
        "yolo_seg": "segmentation",
        "yolov8_seg": "segmentation",
        "roi": "roi_classification",
        "roi_classification": "roi_classification",
        "roi_detection_classification": "roi_classification",
    }
    return aliases.get(task, task)


def _parse_input_size(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except Exception:
            return []
    s = str(value or "").strip().replace("\\,", ",")
    if not s:
        return []
    for sep in [",", "x", "X", " "]:
        if sep in s:
            parts = [x for x in s.replace("x", sep).replace("X", sep).split(sep) if x.strip()]
            if len(parts) >= 2:
                try:
                    return [int(float(parts[0])), int(float(parts[1]))]
                except Exception:
                    return []
    return []


def _format_input_size_for_env(value: Any) -> str:
    size = _parse_input_size(value)
    if len(size) == 2:
        return f"{size[0]},{size[1]}"
    return str(value or "").strip()


def _as_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip())
    except Exception:
        return None


def get_cpp_current_model_config() -> Dict[str, Any]:
    """Read the configured C++ model from edge/runtime/cpp.env.

    v0.8.1 treats cpp.env as the single persistent source of truth for the
    C++ inference model. This function intentionally does not call the running
    C++ service: it still works when visionops-inference-cpp.service is stopped.
    """
    env = read_cpp_env()
    model_path = str(env.get("VISIONOPS_CPP_MODEL_PATH") or "").strip()
    meta_path = str(env.get("VISIONOPS_CPP_CLASS_NAMES_FILE") or "").strip()
    pipeline_config = str(env.get("VISIONOPS_CPP_PIPELINE_CONFIG") or "").strip()
    task = _normalize_model_task(env.get("VISIONOPS_CPP_TASK"))
    input_size = _parse_input_size(env.get("VISIONOPS_CPP_INPUT_SIZE"))
    num_classes = _as_int(env.get("VISIONOPS_CPP_NUM_CLASSES"), 0)

    missing = []
    if task == "roi_classification":
        if not pipeline_config:
            missing.append("VISIONOPS_CPP_PIPELINE_CONFIG")
    else:
        if not model_path:
            missing.append("VISIONOPS_CPP_MODEL_PATH")
        if not meta_path:
            missing.append("VISIONOPS_CPP_CLASS_NAMES_FILE")
    if not task:
        missing.append("VISIONOPS_CPP_TASK")
    if num_classes <= 0:
        missing.append("VISIONOPS_CPP_NUM_CLASSES")
    if len(input_size) != 2:
        missing.append("VISIONOPS_CPP_INPUT_SIZE")

    raw_model_env = {k: env.get(k, "") for k in sorted(_CPP_MODEL_ENV_KEYS) if k in env}

    return {
        "ok": True,
        "source": "cpp.env",
        "cpp_env_path": str(CPP_ENV_PATH),
        "cpp_env_exists": CPP_ENV_PATH.exists(),
        "service_name": CPP_SERVICE_NAME,
        "valid": len(missing) == 0,
        "missing": missing,
        "model_path": model_path,
        "pipeline_config": pipeline_config,
        "class_names_file": meta_path,
        "meta_path": meta_path,
        "task": task,
        "num_classes": num_classes,
        "input_size": input_size,
        "port": _as_int(env.get("VISIONOPS_CPP_PORT"), 18080),
        "npu_core": str(env.get("VISIONOPS_CPP_NPU_CORE") or "auto"),
        "conf_threshold": _as_float_or_none(env.get("VISIONOPS_CPP_CONF_THRESHOLD")),
        "nms_threshold": _as_float_or_none(env.get("VISIONOPS_CPP_NMS_THRESHOLD")),
        "mask_threshold": _as_float_or_none(env.get("VISIONOPS_CPP_MASK_THRESHOLD")),
        "topk": _as_int(env.get("VISIONOPS_CPP_TOPK"), 5),
        "max_det": _as_int(env.get("VISIONOPS_CPP_MAX_DET"), 100),
        "output_mode": str(env.get("VISIONOPS_CPP_OUTPUT_MODE") or "float"),
        "preprocess_backend": str(env.get("VISIONOPS_CPP_PREPROCESS_BACKEND") or "auto"),
        "rga_mode": str(env.get("VISIONOPS_CPP_RGA_MODE") or "resize_color"),
        "raw_env": raw_model_env,
    }



def write_cpp_pipeline_env(pipeline_config: Mapping[str, Any]) -> Path:
    """v0.8.4: Update cpp.env for ROI classification pipeline bundles.

    The C++ binary uses --task roi_classification and --pipeline-config. Camera,
    stream and preprocessing settings are preserved from existing cpp.env.
    """
    env = read_cpp_env()
    cfg_path = str(
        pipeline_config.get("pipeline_config")
        or pipeline_config.get("config_path")
        or pipeline_config.get("path")
        or ""
    ).strip()
    task = _normalize_model_task(pipeline_config.get("task") or "roi_classification")
    if task != "roi_classification":
        raise CppRuntimeSettingsError(f"当前 pipeline 切换仅支持 roi_classification，收到: {task or 'empty'}")
    if not cfg_path:
        raise CppRuntimeSettingsError("pipeline_config 不能为空")

    env["VISIONOPS_CPP_TASK"] = "roi_classification"
    env["VISIONOPS_CPP_PIPELINE_CONFIG"] = cfg_path
    # Keep single-model keys for compatibility/diagnostics, but the C++ binary
    # ignores them in roi_classification mode.
    env.setdefault("VISIONOPS_CPP_MODEL_PATH", str(INSTALL_DIR / "models" / "unused_roi_pipeline.rknn"))
    env.setdefault("VISIONOPS_CPP_CLASS_NAMES_FILE", cfg_path)
    env.setdefault("VISIONOPS_CPP_NUM_CLASSES", "1")
    env.setdefault("VISIONOPS_CPP_INPUT_SIZE", "640,640")
    env.setdefault("VISIONOPS_CPP_BIN", str(INSTALL_DIR / "bin" / "visionops_inference_cpp"))
    env.setdefault("VISIONOPS_CPP_PORT", "18080")
    env.setdefault("VISIONOPS_CPP_NPU_CORE", "auto")

    CPP_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-generated/updated by VisionOps Collector C++ ROI pipeline API"]
    for key in sorted(env.keys()):
        lines.append(f"{key}={_format_env_value(env[key])}")
    CPP_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return CPP_ENV_PATH

def write_cpp_model_env(model_config: Mapping[str, Any]) -> Path:
    """Update model-related keys in cpp.env while preserving camera settings.

    This helper is added in v0.8.1 for v0.8.2 model switching. v0.8.1 itself
    only exposes current config; it does not call this function from the UI.
    """
    env = read_cpp_env()

    model_path = str(model_config.get("model_path") or model_config.get("path") or "").strip()
    meta_path = str(
        model_config.get("class_names_file")
        or model_config.get("meta_path")
        or model_config.get("yaml_path")
        or ""
    ).strip()
    task = _normalize_model_task(model_config.get("task"))
    input_size = _format_input_size_for_env(model_config.get("input_size") or env.get("VISIONOPS_CPP_INPUT_SIZE") or "")
    num_classes = _as_int(model_config.get("num_classes"), _as_int(env.get("VISIONOPS_CPP_NUM_CLASSES"), 0))

    if not model_path:
        raise CppRuntimeSettingsError("model_path 不能为空")
    if not meta_path:
        raise CppRuntimeSettingsError("class_names_file/meta_path 不能为空")
    if task not in {"classification", "detection", "obb_detection", "segmentation"}:
        raise CppRuntimeSettingsError(f"当前单模型切换不支持任务类型: {task or 'empty'}")
    if num_classes <= 0:
        raise CppRuntimeSettingsError("num_classes 必须大于 0")
    if len(_parse_input_size(input_size)) != 2:
        raise CppRuntimeSettingsError("input_size 必须为形如 640,640 或 [640, 640] 的格式")

    env["VISIONOPS_CPP_MODEL_PATH"] = model_path
    env["VISIONOPS_CPP_CLASS_NAMES_FILE"] = meta_path
    env["VISIONOPS_CPP_TASK"] = task
    env["VISIONOPS_CPP_NUM_CLASSES"] = str(num_classes)
    env["VISIONOPS_CPP_INPUT_SIZE"] = input_size

    # Optional algorithm parameters. Omit means preserve existing cpp.env values.
    optional_key_map = {
        "conf_threshold": "VISIONOPS_CPP_CONF_THRESHOLD",
        "nms_threshold": "VISIONOPS_CPP_NMS_THRESHOLD",
        "mask_threshold": "VISIONOPS_CPP_MASK_THRESHOLD",
        "topk": "VISIONOPS_CPP_TOPK",
        "max_det": "VISIONOPS_CPP_MAX_DET",
        "output_mode": "VISIONOPS_CPP_OUTPUT_MODE",
        "preprocess_backend": "VISIONOPS_CPP_PREPROCESS_BACKEND",
        "rga_mode": "VISIONOPS_CPP_RGA_MODE",
    }
    for cfg_key, env_key in optional_key_map.items():
        if cfg_key in model_config and model_config.get(cfg_key) is not None:
            env[env_key] = str(model_config.get(cfg_key))

    env.setdefault("VISIONOPS_CPP_BIN", str(INSTALL_DIR / "bin" / "visionops_inference_cpp"))
    env.setdefault("VISIONOPS_CPP_PORT", "18080")
    env.setdefault("VISIONOPS_CPP_NPU_CORE", "auto")

    CPP_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-generated/updated by VisionOps Collector C++ model API"]
    for key in sorted(env.keys()):
        lines.append(f"{key}={_format_env_value(env[key])}")
    CPP_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return CPP_ENV_PATH


def write_cpp_env(settings: Mapping[str, Any]) -> Path:
    normalized = _normalize_settings(settings)
    _validate_settings(normalized)

    env = read_cpp_env()
    for setting_key, env_key in _SETTING_TO_CPP_ENV.items():
        env[env_key] = str(normalized.get(setting_key, ""))

    # Keep legacy/shared source variable in sync for scripts and diagnostics.
    env["VISIONOPS_CAMERA_SOURCE"] = str(normalized.get("camera_source") or "")

    # If cpp.env does not exist yet, provide safe defaults for the keys needed by the start script.
    env.setdefault("VISIONOPS_CPP_BIN", str(INSTALL_DIR / "bin" / "visionops_inference_cpp"))
    env.setdefault("VISIONOPS_CPP_PORT", "18080")
    env.setdefault("VISIONOPS_CPP_STREAM_BACKEND", str(normalized.get("stream_backend", "opencv")))

    CPP_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-generated/updated by VisionOps Collector C++ settings API"]
    for key in sorted(env.keys()):
        lines.append(f"{key}={_format_env_value(env[key])}")
    CPP_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return CPP_ENV_PATH


def _run_command(cmd: list[str], timeout: float = 15.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def restart_cpp_service() -> Dict[str, Any]:
    commands = [
        ["sudo", "-n", "systemctl", "restart", f"{CPP_SERVICE_NAME}.service"],
        ["systemctl", "restart", f"{CPP_SERVICE_NAME}.service"],
    ]
    last: Optional[subprocess.CompletedProcess] = None
    for cmd in commands:
        last = _run_command(cmd, timeout=20.0)
        if last.returncode == 0:
            return {"ok": True, "command": " ".join(cmd), "stdout": last.stdout.strip()}
    stderr = (last.stderr if last else "").strip()
    raise CppRuntimeSettingsError(
        f"重启 {CPP_SERVICE_NAME}.service 失败：{stderr or 'unknown error'}。"
        "请确认 Collector 运行用户具备 sudo -n systemctl restart 权限。"
    )


def wait_cpp_health(timeout_sec: float = 12.0) -> Dict[str, Any]:
    base = str(CPP_INFERENCE_URL or "http://127.0.0.1:18080").rstrip("/")
    url = base + "/health"
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            import json

            data = json.loads(raw) if raw.strip() else {}
            if isinstance(data, dict):
                return data
            return {"data": data}
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise CppRuntimeSettingsError(f"C++ 服务重启后 health 检查超时：{last_error}")


def apply_cpp_camera_settings(settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if settings is not None:
        normalized = save_cpp_camera_settings(settings)
    else:
        normalized = get_saved_cpp_camera_settings()
        _validate_settings(normalized)

    env_path = write_cpp_env(normalized)
    restart_info = restart_cpp_service()
    health = wait_cpp_health()
    return {
        "ok": True,
        "message": "C++ 相机设置已写入 cpp.env，并已重启 C++ 推理服务",
        "settings": normalized,
        "cpp_env_path": str(env_path),
        "restart": restart_info,
        "health": health,
    }
