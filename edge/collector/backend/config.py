#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

COLLECTOR_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = COLLECTOR_DIR / "static"

# v2.0 设置保存/回显框架：Web 设置写入运行时覆盖文件。
# 默认使用 edge/runtime/runtime_overrides.yaml；单独运行 collector.zip 时也可以通过
# VISIONOPS_RUNTIME_DIR 或 VISIONOPS_RUNTIME_OVERRIDES_FILE 指定位置。
RUNTIME_DIR = Path(os.getenv("VISIONOPS_RUNTIME_DIR", COLLECTOR_DIR.parent / "runtime")).resolve()
RUNTIME_OVERRIDES_FILE = Path(
    os.getenv("VISIONOPS_RUNTIME_OVERRIDES_FILE", RUNTIME_DIR / "runtime_overrides.yaml")
).resolve()

def _load_runtime_overrides() -> dict:
    try:
        import yaml  # type: ignore
        if not RUNTIME_OVERRIDES_FILE.exists():
            return {}
        with RUNTIME_OVERRIDES_FILE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        # 设置文件损坏时不影响主界面启动。
        return {}

_RUNTIME_OVERRIDES = _load_runtime_overrides()

def _override(path: str, default=None):
    cur = _RUNTIME_OVERRIDES
    for key in path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def _override_first(paths, default=None):
    for path in paths:
        cur = _RUNTIME_OVERRIDES
        found = True
        for key in path.split('.'):
            if not isinstance(cur, dict) or key not in cur:
                found = False
                break
            cur = cur[key]
        if found:
            return cur
    return default

def _override_first_non_empty(paths, default=None):
    for path in paths:
        value = _override_first([path], None)
        if value is not None and str(value).strip() != "":
            return value
    return default

def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", "关闭", "否"}

def _as_int(value, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default

def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _runtime_camera_source(default: str) -> str:
    camera_type = str(_override("camera.type", "") or "").strip().lower()
    if camera_type == "rtsp":
        url = str(_override("camera.rtsp.url", "") or "").strip()
        if url:
            return url
        ip = str(_override("camera.rtsp.ip", "") or "").strip()
        port = _as_int(_override("camera.rtsp.port", 554), 554)
        channel = str(_override("camera.rtsp.channel", "102") or "102").strip()
        username = str(_override("camera.rtsp.username", "admin") or "admin").strip()
        password = str(_override("camera.rtsp.password", "password") or "password").strip()
        if ip:
            return f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}"
    if camera_type == "usb":
        node = str(_override("camera.usb.device_node", "") or "").strip()
        if node:
            return node
    return default

# v4.5 起不再暴露数据集选择给工人；后台固定使用一个本地采集目录。
DATA_ROOT = Path(os.getenv("VISIONOPS_COLLECTOR_DATA_DIR", COLLECTOR_DIR / "data")).resolve()
DEFAULT_DATASET_NAME = os.getenv("VISIONOPS_DEFAULT_DATASET", "local_dataset")

DEVICE_ID = str(_override("vision_box.device_id", os.getenv("VISIONOPS_DEVICE_ID", "rk3588-001")))
USER_ID = os.getenv("VISIONOPS_USER_ID", "operator-001")
_CUSTOMER_ID = str(_override("vision_box.customer_id", os.getenv("VISIONOPS_CUSTOMER_ID", "CUST-001")))
_ENV_CAMERA_SOURCE = os.getenv("VISIONOPS_CAMERA_SOURCE", "browser")
CAMERA_SOURCE = _runtime_camera_source(_ENV_CAMERA_SOURCE)
CAMERA_STREAM_FPS = _as_float(_override("camera.common.fps", os.getenv("VISIONOPS_CAMERA_STREAM_FPS", "6")), 6.0)
CAMERA_PREVIEW_WIDTH = _as_int(_override("camera.common.preview_width", os.getenv("VISIONOPS_CAMERA_PREVIEW_WIDTH", "960")), 960)
CAMERA_JPEG_QUALITY = _as_int(_override("camera.common.jpeg_quality", os.getenv("VISIONOPS_CAMERA_JPEG_QUALITY", "75")), 75)
CAMERA_RECONNECT_MAX_FAILS = _as_int(_override("camera.common.reconnect_max_fails", os.getenv("VISIONOPS_CAMERA_RECONNECT_MAX_FAILS", "30")), 30)

# v5.0 打包上传配置：确认上传后，将本地 tar.gz 包通过 SSH/SCP 上传到电脑端。
# 推荐在 RK3588 上配置免密 SSH 到电脑。未配置 VISIONOPS_UPLOAD_HOST 时，只本地打包不远程上传。
# v2.1.1：上传配置归入“视觉盒子设置 -> 服务端上传配置”。
# 兼容旧版 runtime_overrides.yaml 的顶层 upload.*，并保留 collector.env 环境变量兜底。
UPLOAD_ENABLED = _as_bool(_override_first(["vision_box.upload.enabled", "upload.enabled"], os.getenv("VISIONOPS_UPLOAD_ENABLED", "1")), True)
UPLOAD_HOST = str(_override_first_non_empty(["vision_box.upload.host", "upload.host"], os.getenv("VISIONOPS_UPLOAD_HOST", ""))).strip()
UPLOAD_USER = str(_override_first_non_empty(["vision_box.upload.user", "upload.user"], os.getenv("VISIONOPS_UPLOAD_USER", "pc"))).strip()
UPLOAD_PORT = _as_int(_override_first(["vision_box.upload.port", "upload.port"], os.getenv("VISIONOPS_UPLOAD_PORT", "22")), 22)
UPLOAD_TARGET_DIR = str(_override_first_non_empty(["vision_box.upload.target_dir", "upload.target_dir"], os.getenv("VISIONOPS_UPLOAD_TARGET_DIR", "/home/pc/桌面/visionops_v2/data/incoming"))).strip()
UPLOAD_TIMEOUT_SEC = _as_int(_override_first(["vision_box.upload.timeout_sec", "upload.timeout_sec"], os.getenv("VISIONOPS_UPLOAD_TIMEOUT_SEC", "120")), 120)

# v6.5 模型验证页配置：读取 /opt/visionops/models 下的版本化模型。
# 每个模型由同名 .rknn + .yaml 组成，例如 xxx.rknn / xxx.yaml。
MODELS_DIR = Path(str(_override("vision_box.models_dir", os.getenv("VISIONOPS_MODELS_DIR", "/opt/visionops/models")))).resolve()

# v6.5 验证推理服务配置：任务、类别、输入尺寸均从选中模型的同名 YAML 读取。
VALIDATION_INFER_HOST = os.getenv("VISIONOPS_VALIDATION_INFER_HOST", "127.0.0.1")
VALIDATION_INFER_PORT = _as_int(_override("vision_box.validation_port", os.getenv("VISIONOPS_VALIDATION_INFER_PORT", "8082")), 8082)
VALIDATION_INFER_TIMEOUT_SEC = float(os.getenv("VISIONOPS_VALIDATION_INFER_TIMEOUT_SEC", "20"))
VALIDATION_NPU_CORE = os.getenv("VISIONOPS_VALIDATION_NPU_CORE", "auto")
VALIDATION_TOPK = _as_int(_override("algorithm.classification.topk", os.getenv("VISIONOPS_VALIDATION_TOPK", "5")), 5)
VALIDATION_WARMUP_RUNS = _as_int(_override("algorithm.common.warmup_runs", os.getenv("VISIONOPS_VALIDATION_WARMUP_RUNS", "1")), 1)
VALIDATION_ENGINE_PATH = os.getenv(
    "VISIONOPS_VALIDATION_ENGINE_PATH",
    str((COLLECTOR_DIR.parent / "inference" / "engine.py").resolve()),
)

# v6.4+ 低频实时检测配置。前端默认每 1 秒请求一次单帧推理。
VALIDATION_REALTIME_INTERVAL_MS = _as_int(_override("algorithm.common.realtime_interval_ms", os.getenv("VISIONOPS_VALIDATION_REALTIME_INTERVAL_MS", "1000")), 1000)
# v0.5.0 C++ inference service proxy.
# Collector 只代理 C++ 服务；不要在 Python 中参与实时 RTSP 解码和逐帧推理。
CPP_INFERENCE_ENABLED = _as_bool(
    os.getenv("VISIONOPS_CPP_INFERENCE_ENABLED", "1"),
    True,
)
CPP_INFERENCE_URL = str(
    os.getenv(
        "VISIONOPS_CPP_SERVICE_URL",
        os.getenv("VISIONOPS_CPP_INFERENCE_URL", "http://127.0.0.1:18080"),
    )
).rstrip("/")
CPP_INFERENCE_TIMEOUT_SEC = _as_float(
    os.getenv("VISIONOPS_CPP_TIMEOUT_SEC", "5"),
    5.0,
)
CPP_INFERENCE_IMAGE_TIMEOUT_SEC = _as_float(
    os.getenv("VISIONOPS_CPP_IMAGE_TIMEOUT_SEC", "15"),
    15.0,
)


# 生产模式连续检测 + Gateway 推送配置
# Web/Collector 作为唯一 RTSP 拉流进程，从 latest_frame 取图后调用推理服务，
# 再将检测结果 POST 到 Gateway 的 HTTP 接收口。
PRODUCTION_INFER_URL = os.getenv(
    "VISIONOPS_PRODUCTION_INFER_URL",
    f"http://{VALIDATION_INFER_HOST}:{VALIDATION_INFER_PORT}/infer",
)

PRODUCTION_GATEWAY_PUSH_URL = os.getenv(
    "VISIONOPS_PRODUCTION_GATEWAY_PUSH_URL",
    "http://127.0.0.1:9101/push_result",
)

PRODUCTION_DETECT_INTERVAL_MS = _as_int(
    _override("algorithm.common.production_detect_interval_ms", _override("algorithm.common.production_interval_ms", os.getenv("VISIONOPS_PRODUCTION_DETECT_INTERVAL_MS", str(VALIDATION_REALTIME_INTERVAL_MS)))),
    VALIDATION_REALTIME_INTERVAL_MS,
)

PRODUCTION_CAMERA_ID = int(os.getenv("VISIONOPS_PRODUCTION_CAMERA_ID", "1"))

# Web 端全局检测结果推送配置。
# 只要 Web 接口完成一次模型推理，就会 best-effort POST 到 Gateway HTTP 接收口，
# 再由 Gateway 通过 9100 TCP 推给上位机。
GATEWAY_PUSH_ENABLED = os.getenv("VISIONOPS_GATEWAY_PUSH_ENABLED", "1") not in {"0", "false", "False", "no", "NO"}
GATEWAY_PUSH_URL = os.getenv("VISIONOPS_GATEWAY_PUSH_URL", PRODUCTION_GATEWAY_PUSH_URL)
GATEWAY_PUSH_CAMERA_ID = int(os.getenv("VISIONOPS_GATEWAY_PUSH_CAMERA_ID", str(PRODUCTION_CAMERA_ID)))
GATEWAY_PUSH_TIMEOUT_SEC = float(os.getenv("VISIONOPS_GATEWAY_PUSH_TIMEOUT_SEC", "0.8"))

UI_VERSION = "setting-v2.3.0-light-vision-box-runtime"
