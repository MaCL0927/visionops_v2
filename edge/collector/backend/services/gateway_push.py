#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web/Collector 全局 Gateway 推送服务。

职责：
1. 所有 Web 端模型推理完成后，统一调用 push_result_to_gateway()
2. 将推理结果整理成 Gateway /push_result 接口需要的轻量 JSON
3. 推送失败不影响 Web 端原有检测返回，只把失败信息写入 gateway_push 字段

数据流：
    collector 8090 -> POST http://127.0.0.1:9101/push_result -> gateway 9100 TCP 推送
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from backend.config import (
    GATEWAY_PUSH_CAMERA_ID,
    GATEWAY_PUSH_ENABLED,
    GATEWAY_PUSH_TIMEOUT_SEC,
    GATEWAY_PUSH_URL,
)

logger = logging.getLogger("visionops.gateway_push")

_frame_counter = itertools.count(1)
_counter_lock = threading.Lock()


def _next_frame_id() -> int:
    with _counter_lock:
        return int(next(_frame_counter))


def _now_timestamp() -> List[int]:
    t = time.time()
    sec = int(t)
    ms = int((t - sec) * 1000)
    return [sec, ms]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gateway HTTP 错误: {e.code}, detail={detail}") from e
    except Exception as e:
        raise RuntimeError(f"推送到 Gateway 失败: {url}, error={e}") from e


def _normalize_prediction_item(pred: Dict[str, Any]) -> Dict[str, Any]:
    """只保留上位机常用字段，避免把 raw/mask 等大字段推送出去。"""
    bbox = pred.get("bbox")
    center = pred.get("center")
    center_x = pred.get("center_x")
    center_y = pred.get("center_y")

    if center is None and isinstance(bbox, list) and len(bbox) >= 4:
        x1 = _safe_float(bbox[0], 0.0) or 0.0
        y1 = _safe_float(bbox[1], 0.0) or 0.0
        x2 = _safe_float(bbox[2], 0.0) or 0.0
        y2 = _safe_float(bbox[3], 0.0) or 0.0
        center_x = round((x1 + x2) / 2.0, 2)
        center_y = round((y1 + y2) / 2.0, 2)
        center = [center_x, center_y]

    item = {
        "class_id": pred.get("class_id"),
        "class_name": pred.get("class_name") or pred.get("class") or pred.get("label"),
        "confidence": pred.get("confidence", pred.get("score")),
        "bbox": bbox,
        "center": center,
        "center_x": center_x,
        "center_y": center_y,
    }

    # OBB / segmentation 可选字段：不强依赖，但有就保留。
    if pred.get("points") is not None:
        item["points"] = pred.get("points")
    if pred.get("angle") is not None:
        item["angle"] = pred.get("angle")
    if pred.get("segments") is not None:
        item["segments"] = pred.get("segments")
    if pred.get("mask_area") is not None:
        item["mask_area"] = pred.get("mask_area")

    return item


def _classification_predictions(infer_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """分类任务没有 bbox，这里把 topk/top1 也转换成 predictions，方便统一推送。"""
    topk = infer_result.get("topk")
    if isinstance(topk, list) and topk:
        return [_normalize_prediction_item(p) for p in topk if isinstance(p, dict)]

    result = infer_result.get("result")
    if isinstance(result, dict):
        return [
            _normalize_prediction_item(
                {
                    "class_id": result.get("class_id"),
                    "class_name": result.get("class_name"),
                    "confidence": result.get("confidence"),
                }
            )
        ]
    return []


def _extract_predictions(infer_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    preds = infer_result.get("predictions")
    if isinstance(preds, list):
        return [_normalize_prediction_item(p) for p in preds if isinstance(p, dict)]
    return _classification_predictions(infer_result)


def build_gateway_payload(
    infer_result: Dict[str, Any],
    *,
    source: str,
    dataset: str = "",
    model_name: str = "",
    image_id: str = "",
    camera_id: int = GATEWAY_PUSH_CAMERA_ID,
    frame_id: Optional[int] = None,
) -> Dict[str, Any]:
    """将 Web 端一次推理结果转换成 Gateway /push_result 的轻量 JSON。"""
    predictions = _extract_predictions(infer_result)
    frame_id = int(frame_id or infer_result.get("frame_id") or _next_frame_id())
    task = str(infer_result.get("task") or "unknown")

    payload: Dict[str, Any] = {
        "function": "result",
        "timestamp": infer_result.get("timestamp") or _now_timestamp(),
        "frame_id": frame_id,
        "camera_id": int(infer_result.get("camera_id") or camera_id),
        "result": 0 if predictions else 1,
        "task": task,
        "latency_ms": infer_result.get("latency_ms"),
        "count": len(predictions),
        "predictions": predictions,
        "source": source,
        "dataset": dataset,
        "model_name": model_name,
        "image_id": image_id,
        "mode": infer_result.get("mode"),
    }

    # 分类结果额外保留一个 classification 字段，方便上位机区分。
    if isinstance(infer_result.get("result"), dict):
        payload["classification"] = infer_result.get("result")

    # 已有 captured/realtime 时带上文件名/URL，便于追踪，不带图片本体。
    for key in ("captured", "realtime"):
        item = infer_result.get(key)
        if isinstance(item, dict):
            payload[key] = {
                "id": item.get("id"),
                "name": item.get("name"),
                "filename": item.get("filename"),
                "url": item.get("url"),
                "dataset": item.get("dataset"),
            }

    return payload


def push_result_to_gateway(
    infer_result: Dict[str, Any],
    *,
    source: str,
    dataset: str = "",
    model_name: str = "",
    image_id: str = "",
    camera_id: int = GATEWAY_PUSH_CAMERA_ID,
    gateway_url: Optional[str] = None,
    enabled: Optional[bool] = None,
    timeout: float = GATEWAY_PUSH_TIMEOUT_SEC,
    frame_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    统一推送入口。

    返回值始终可 JSON 序列化。推送失败不会抛出到业务接口外层。
    """
    use_enabled = GATEWAY_PUSH_ENABLED if enabled is None else bool(enabled)
    url = gateway_url or GATEWAY_PUSH_URL

    if not use_enabled:
        return {"ok": False, "skipped": True, "reason": "disabled", "url": url}

    payload = build_gateway_payload(
        infer_result,
        source=source,
        dataset=dataset,
        model_name=model_name,
        image_id=image_id,
        camera_id=camera_id,
        frame_id=frame_id,
    )

    try:
        response = _post_json(url, payload, timeout=timeout)
        return {
            "ok": True,
            "url": url,
            "frame_id": payload.get("frame_id"),
            "count": payload.get("count"),
            "response": response,
        }
    except Exception as e:
        logger.warning("Gateway 推送失败: source=%s model=%s image=%s err=%s", source, model_name, image_id, e)
        return {
            "ok": False,
            "url": url,
            "frame_id": payload.get("frame_id"),
            "count": payload.get("count"),
            "error": str(e),
        }
