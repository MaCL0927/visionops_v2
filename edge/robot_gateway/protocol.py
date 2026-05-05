#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisionOps Gateway 协议工具。

协议格式：
    *{JSON}#

当前用途：
1. Gateway 接收 Web/Collector 发来的检测结果
2. Gateway 转换为 *JSON# TCP 帧
3. Gateway 主动推送给上位机或其他系统
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List


FRAME_START = b"*"
FRAME_END = b"#"


def now_timestamp() -> List[int]:
    """
    返回 [秒, 毫秒]。
    """
    t = time.time()
    sec = int(t)
    ms = int((t - sec) * 1000)
    return [sec, ms]


def encode_frame(data: Dict[str, Any]) -> bytes:
    """
    dict -> *{JSON}#
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return FRAME_START + payload.encode("utf-8") + FRAME_END


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一单个检测框字段。

    支持 engine.py 常见输出：
        class_id
        class_name
        confidence
        bbox
        center / center_x / center_y
    """
    bbox = pred.get("bbox")
    center = pred.get("center")

    center_x = pred.get("center_x")
    center_y = pred.get("center_y")

    if center is None and isinstance(bbox, list) and len(bbox) >= 4:
        x1 = _safe_float(bbox[0], 0.0)
        y1 = _safe_float(bbox[1], 0.0)
        x2 = _safe_float(bbox[2], 0.0)
        y2 = _safe_float(bbox[3], 0.0)
        center_x = round((x1 + x2) / 2.0, 2)
        center_y = round((y1 + y2) / 2.0, 2)
        center = [center_x, center_y]

    if center is not None and isinstance(center, list) and len(center) >= 2:
        center_x = center[0]
        center_y = center[1]

    return {
        "class_id": pred.get("class_id"),
        "class_name": pred.get("class_name"),
        "confidence": pred.get("confidence"),
        "bbox": bbox,
        "center": center,
        "center_x": center_x,
        "center_y": center_y,
    }


def build_detection_frame(
    inference_result: Dict[str, Any],
    camera_id: int = 1,
    frame_id: int | None = None,
) -> bytes:
    """
    将检测结果封装为 TCP 推送帧。

    输出示例：
        *{
          "function": "result",
          "timestamp": [sec, ms],
          "frame_id": 1,
          "camera_id": 1,
          "result": 0,
          "task": "detection",
          "latency_ms": 64.5,
          "count": 1,
          "predictions": [...]
        }#
    """
    predictions = inference_result.get("predictions", [])
    if not isinstance(predictions, list):
        predictions = []

    normalized_predictions = [
        _normalize_prediction(p)
        for p in predictions
        if isinstance(p, dict)
    ]

    result_code = int(inference_result.get(
        "result",
        0 if normalized_predictions else 1,
    ))

    frame = {
        "function": "result",
        "timestamp": inference_result.get("timestamp", now_timestamp()),
        "frame_id": inference_result.get("frame_id", frame_id),
        "camera_id": int(inference_result.get("camera_id", camera_id)),
        "result": result_code,
        "task": inference_result.get("task", "detection"),
        "latency_ms": inference_result.get("latency_ms"),
        "count": len(normalized_predictions),
        "predictions": normalized_predictions,
    }

    return encode_frame(frame)


def build_error_frame(message: str, camera_id: int = 1, frame_id: int | None = None) -> bytes:
    """
    错误帧。
    """
    frame = {
        "function": "result",
        "timestamp": now_timestamp(),
        "frame_id": frame_id,
        "camera_id": camera_id,
        "result": -1,
        "task": "unknown",
        "latency_ms": None,
        "count": 0,
        "error": message,
        "predictions": [],
    }
    return encode_frame(frame)
