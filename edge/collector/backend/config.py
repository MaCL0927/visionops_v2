#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

COLLECTOR_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = COLLECTOR_DIR / "static"

# v4.5 起不再暴露数据集选择给工人；后台固定使用一个本地采集目录。
DATA_ROOT = Path(os.getenv("VISIONOPS_COLLECTOR_DATA_DIR", COLLECTOR_DIR / "data")).resolve()
DEFAULT_DATASET_NAME = os.getenv("VISIONOPS_DEFAULT_DATASET", "local_dataset")

DEVICE_ID = os.getenv("VISIONOPS_DEVICE_ID", "rk3588-001")
USER_ID = os.getenv("VISIONOPS_USER_ID", "operator-001")
CAMERA_SOURCE = os.getenv("VISIONOPS_CAMERA_SOURCE", "browser")
CAMERA_STREAM_FPS = float(os.getenv("VISIONOPS_CAMERA_STREAM_FPS", "6"))
CAMERA_PREVIEW_WIDTH = int(os.getenv("VISIONOPS_CAMERA_PREVIEW_WIDTH", "960"))
CAMERA_JPEG_QUALITY = int(os.getenv("VISIONOPS_CAMERA_JPEG_QUALITY", "75"))
CAMERA_RECONNECT_MAX_FAILS = int(os.getenv("VISIONOPS_CAMERA_RECONNECT_MAX_FAILS", "30"))

# v5.0 打包上传配置：确认上传后，将本地 tar.gz 包通过 SSH/SCP 上传到电脑端。
# 推荐在 RK3588 上配置免密 SSH 到电脑。未配置 VISIONOPS_UPLOAD_HOST 时，只本地打包不远程上传。
UPLOAD_ENABLED = os.getenv("VISIONOPS_UPLOAD_ENABLED", "1") not in {"0", "false", "False", "no", "NO"}
UPLOAD_HOST = os.getenv("VISIONOPS_UPLOAD_HOST", "").strip()
UPLOAD_USER = os.getenv("VISIONOPS_UPLOAD_USER", "pc").strip()
UPLOAD_PORT = int(os.getenv("VISIONOPS_UPLOAD_PORT", "22"))
UPLOAD_TARGET_DIR = os.getenv("VISIONOPS_UPLOAD_TARGET_DIR", "/home/pc/桌面/visionops_v2/data").strip()
UPLOAD_TIMEOUT_SEC = int(os.getenv("VISIONOPS_UPLOAD_TIMEOUT_SEC", "120"))

# v6.5 模型验证页配置：读取 /opt/visionops/models 下的版本化模型。
# 每个模型由同名 .rknn + .yaml 组成，例如 xxx.rknn / xxx.yaml。
MODELS_DIR = Path(os.getenv("VISIONOPS_MODELS_DIR", "/opt/visionops/models")).resolve()

# v6.5 验证推理服务配置：任务、类别、输入尺寸均从选中模型的同名 YAML 读取。
VALIDATION_INFER_HOST = os.getenv("VISIONOPS_VALIDATION_INFER_HOST", "127.0.0.1")
VALIDATION_INFER_PORT = int(os.getenv("VISIONOPS_VALIDATION_INFER_PORT", "8082"))
VALIDATION_INFER_TIMEOUT_SEC = float(os.getenv("VISIONOPS_VALIDATION_INFER_TIMEOUT_SEC", "20"))
VALIDATION_NPU_CORE = os.getenv("VISIONOPS_VALIDATION_NPU_CORE", "auto")
VALIDATION_TOPK = int(os.getenv("VISIONOPS_VALIDATION_TOPK", "5"))
VALIDATION_WARMUP_RUNS = int(os.getenv("VISIONOPS_VALIDATION_WARMUP_RUNS", "1"))
VALIDATION_ENGINE_PATH = os.getenv(
    "VISIONOPS_VALIDATION_ENGINE_PATH",
    str((COLLECTOR_DIR.parent / "inference" / "engine.py").resolve()),
)

# v6.4+ 低频实时检测配置。前端默认每 1 秒请求一次单帧推理。
VALIDATION_REALTIME_INTERVAL_MS = int(os.getenv("VISIONOPS_VALIDATION_REALTIME_INTERVAL_MS", "1000"))


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

PRODUCTION_DETECT_INTERVAL_MS = int(
    os.getenv("VISIONOPS_PRODUCTION_DETECT_INTERVAL_MS", str(VALIDATION_REALTIME_INTERVAL_MS))
)

PRODUCTION_CAMERA_ID = int(os.getenv("VISIONOPS_PRODUCTION_CAMERA_ID", "1"))

# Web 端全局检测结果推送配置。
# 只要 Web 接口完成一次模型推理，就会 best-effort POST 到 Gateway HTTP 接收口，
# 再由 Gateway 通过 9100 TCP 推给上位机。
GATEWAY_PUSH_ENABLED = os.getenv("VISIONOPS_GATEWAY_PUSH_ENABLED", "1") not in {"0", "false", "False", "no", "NO"}
GATEWAY_PUSH_URL = os.getenv("VISIONOPS_GATEWAY_PUSH_URL", PRODUCTION_GATEWAY_PUSH_URL)
GATEWAY_PUSH_CAMERA_ID = int(os.getenv("VISIONOPS_GATEWAY_PUSH_CAMERA_ID", str(PRODUCTION_CAMERA_ID)))
GATEWAY_PUSH_TIMEOUT_SEC = float(os.getenv("VISIONOPS_GATEWAY_PUSH_TIMEOUT_SEC", "0.8"))

UI_VERSION = "v6.8-obb-polygon-validation"
