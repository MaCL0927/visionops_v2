#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C++ latest_result -> robot_gateway 自动推送服务。

数据流：
    C++ service /stream/latest_result
        -> Collector cpp_result_push_service
        -> HTTP POST http://127.0.0.1:9101/push_result
        -> robot_gateway TCP 9100 *JSON#

设计原则：
- 不把 TCP 推送、重连、广播逻辑塞进 C++ 推理进程；
- Collector 只做轻量 JSON 转换与定时转发；
- robot_gateway 继续负责 9101 HTTP 接收与 9100 TCP 广播；
- 支持 detection / classification / obb_detection / segmentation / roi_classification。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from backend.services.cpp_inference_client import CppInferenceError, get_json

logger = logging.getLogger("visionops.cpp_result_push")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


DEFAULT_GATEWAY_URL = os.getenv(
    "VISIONOPS_CPP_RESULT_PUSH_URL",
    os.getenv("VISIONOPS_GATEWAY_PUSH_URL", "http://127.0.0.1:9101/push_result"),
)
DEFAULT_CAMERA_ID = _env_int(
    "VISIONOPS_CPP_RESULT_PUSH_CAMERA_ID",
    _env_int("VISIONOPS_GATEWAY_PUSH_CAMERA_ID", 1),
)
DEFAULT_PUSH_FPS = _env_float("VISIONOPS_CPP_RESULT_PUSH_FPS", 5.0)
DEFAULT_TIMEOUT_SEC = _env_float("VISIONOPS_CPP_RESULT_PUSH_TIMEOUT_SEC", 0.8)
DEFAULT_PUSH_EMPTY = _env_bool("VISIONOPS_CPP_RESULT_PUSH_EMPTY", True)
DEFAULT_DEDUPE = _env_bool("VISIONOPS_CPP_RESULT_PUSH_DEDUPE", True)
# v0.8.5.1: 默认常驻启动推送线程。
# 线程常驻不等于一直推送：默认只有 C++ detect/推理流正在运行时才真正 POST 到 robot_gateway。
# 如需禁用常驻，设置 VISIONOPS_CPP_RESULT_PUSH_AUTOSTART=0。
DEFAULT_AUTOSTART = _env_bool("VISIONOPS_CPP_RESULT_PUSH_AUTOSTART", True)
DEFAULT_REQUIRE_INFERENCE = _env_bool("VISIONOPS_CPP_RESULT_PUSH_REQUIRE_INFERENCE", True)
DEFAULT_SCHEMA = os.getenv("VISIONOPS_CPP_RESULT_PUSH_SCHEMA", "visionops_gateway_v1")


def now_timestamp() -> List[int]:
    t = time.time()
    sec = int(t)
    ms = int((t - sec) * 1000)
    return [sec, ms]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _round_float(value: Any, ndigits: int = 3) -> Any:
    f = _safe_float(value, None)
    if f is None:
        return value
    return round(f, ndigits)


def _json_safe(value: Any, *, max_depth: int = 6) -> Any:
    """把来自 C++/Python 的对象规整成可 JSON 序列化的基础类型。"""
    if max_depth <= 0:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v, max_depth=max_depth - 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, max_depth=max_depth - 1) for v in value]
    try:
        return value.item()  # numpy scalar
    except Exception:
        return str(value)


def _bbox_dict(bbox: Any, center: Any = None, center_x: Any = None, center_y: Any = None) -> Optional[Dict[str, Any]]:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    x1 = _safe_float(bbox[0], 0.0) or 0.0
    y1 = _safe_float(bbox[1], 0.0) or 0.0
    x2 = _safe_float(bbox[2], 0.0) or 0.0
    y2 = _safe_float(bbox[3], 0.0) or 0.0

    if isinstance(center, list) and len(center) >= 2:
        cx = _safe_float(center[0], None)
        cy = _safe_float(center[1], None)
    else:
        cx = _safe_float(center_x, None)
        cy = _safe_float(center_y, None)

    if cx is None:
        cx = (x1 + x2) / 2.0
    if cy is None:
        cy = (y1 + y2) / 2.0

    return {
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "cx": round(cx, 2),
        "cy": round(cy, 2),
        "width": round(max(0.0, x2 - x1), 2),
        "height": round(max(0.0, y2 - y1), 2),
        "xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
    }


def _simple_class_item(pred: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class_id": _safe_int(pred.get("class_id"), pred.get("class_id")),
        "class_name": pred.get("class_name") or pred.get("label") or pred.get("class"),
        "score": _round_float(pred.get("confidence", pred.get("score")), 6),
        "confidence": _round_float(pred.get("confidence", pred.get("score")), 6),
        "logit": _round_float(pred.get("logit"), 6) if pred.get("logit") is not None else None,
    }


def _limit_points(points: Any, limit: int = 200) -> Any:
    if not isinstance(points, list):
        return points
    if len(points) <= limit:
        return _json_safe(points)
    if limit <= 0:
        return []
    step = max(1, len(points) // limit)
    out = points[::step][:limit]
    return _json_safe(out)


def _normalize_detection_like_object(pred: Dict[str, Any], idx: int) -> Dict[str, Any]:
    bbox = _bbox_dict(pred.get("bbox"), pred.get("center"), pred.get("center_x"), pred.get("center_y"))
    obj: Dict[str, Any] = {
        "id": idx,
        "class_id": _safe_int(pred.get("class_id"), pred.get("class_id")),
        "class_name": pred.get("class_name") or pred.get("label") or pred.get("class"),
        "score": _round_float(pred.get("confidence", pred.get("score")), 6),
        "confidence": _round_float(pred.get("confidence", pred.get("score")), 6),
        "bbox": bbox,
    }

    if isinstance(pred.get("obb"), dict):
        obb = pred.get("obb") or {}
        obj["obb"] = {
            "cx": _round_float(obb.get("cx"), 2),
            "cy": _round_float(obb.get("cy"), 2),
            "w": _round_float(obb.get("w"), 2),
            "h": _round_float(obb.get("h"), 2),
            "angle": _round_float(obb.get("angle"), 6),
            "angle_unit": obb.get("angle_unit", "radian"),
            "points": _limit_points(obb.get("points"), 32),
        }
    elif pred.get("points") is not None:
        obj["points"] = _limit_points(pred.get("points"), 32)

    if isinstance(pred.get("mask"), dict):
        mask = pred.get("mask") or {}
        obj["mask"] = {
            "area": _round_float(mask.get("area"), 2),
            "threshold": _round_float(mask.get("threshold"), 3),
            "shape": _json_safe(mask.get("shape")),
            "polygon": _limit_points(mask.get("polygon"), 200),
            "segments": _json_safe(mask.get("segments"), max_depth=5),
        }
    elif pred.get("segments") is not None or pred.get("mask_area") is not None:
        obj["mask"] = {
            "area": _round_float(pred.get("mask_area"), 2),
            "segments": _json_safe(pred.get("segments"), max_depth=5),
        }

    # ROI pipeline 的 prediction 中通常包含 detector/classifier/roi 三块。
    if isinstance(pred.get("detector"), dict):
        det = pred.get("detector") or {}
        obj["detector"] = {
            "class_id": _safe_int(det.get("class_id"), det.get("class_id")),
            "class_name": det.get("class_name"),
            "score": _round_float(det.get("confidence", det.get("score")), 6),
            "confidence": _round_float(det.get("confidence", det.get("score")), 6),
            "bbox": _bbox_dict(det.get("bbox"), det.get("center"), det.get("center_x"), det.get("center_y")),
        }
    if isinstance(pred.get("classifier"), dict):
        obj["classifier"] = _simple_class_item(pred.get("classifier") or {})
    if isinstance(pred.get("roi"), dict):
        roi = pred.get("roi") or {}
        obj["roi"] = {
            "mode": roi.get("mode"),
            "pipeline_mode": roi.get("pipeline_mode"),
            "bbox": _bbox_dict(roi.get("bbox")) or _json_safe(roi.get("bbox")),
            "base_bbox": _bbox_dict(roi.get("base_bbox")) or _json_safe(roi.get("base_bbox")),
            "relative_box": _json_safe(roi.get("relative_box")),
            "padding_ratio": _round_float(roi.get("padding_ratio"), 6),
            "class_key": roi.get("class_key"),
            "matched_class_key": roi.get("matched_class_key"),
            "source": roi.get("source"),
        }

    return obj


def _normalize_classification_objects(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    prediction = result.get("prediction")
    topk = result.get("topk")
    objects: List[Dict[str, Any]] = []
    if isinstance(prediction, dict):
        obj = _simple_class_item(prediction)
        obj["id"] = 0
        if isinstance(topk, list):
            obj["topk"] = [_simple_class_item(x) for x in topk if isinstance(x, dict)]
        objects.append(obj)
    elif isinstance(topk, list) and topk:
        first = topk[0]
        if isinstance(first, dict):
            obj = _simple_class_item(first)
            obj["id"] = 0
            obj["topk"] = [_simple_class_item(x) for x in topk if isinstance(x, dict)]
            objects.append(obj)
    return objects


def _extract_objects(latest_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    task = str(latest_result.get("task") or "").strip().lower()
    if task == "classification":
        return _normalize_classification_objects(latest_result)

    preds = latest_result.get("predictions")
    if not isinstance(preds, list):
        preds = []
    return [_normalize_detection_like_object(p, idx) for idx, p in enumerate(preds) if isinstance(p, dict)]


def _result_code(latest_result: Dict[str, Any], objects: List[Dict[str, Any]]) -> int:
    explicit = latest_result.get("result")
    if explicit is not None:
        n = _safe_int(explicit, None)
        if n is not None:
            return int(n)
    status = str(latest_result.get("status") or "").strip().lower()
    if status in {"error", "failed", "exception"}:
        return -1
    if status in {"no_target", "no_result", "empty"}:
        return 1
    return 0 if objects else 1


def _infer_frame_id(latest_result: Dict[str, Any], fallback: int) -> int:
    for key in ("frame_id", "frame", "sequence", "seq"):
        v = _safe_int(latest_result.get(key), None)
        if v is not None:
            return int(v)
    return int(fallback)


def _dedupe_key(latest_result: Dict[str, Any], payload: Dict[str, Any]) -> str:
    frame_id = latest_result.get("frame_id")
    if frame_id is not None:
        return f"frame:{frame_id}"
    # 有些 C++ 版本没有 frame_id，退化为 payload 关键内容 hash。
    slim = {
        "task": payload.get("task"),
        "status": payload.get("status"),
        "count": payload.get("count"),
        "latency_ms": payload.get("latency_ms"),
        "objects": payload.get("objects"),
    }
    raw = json.dumps(slim, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "hash:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_cpp_gateway_payload(
    latest_result: Dict[str, Any],
    *,
    frame_id: int,
    camera_id: int,
    source: str = "cpp_latest_result",
    schema: str = DEFAULT_SCHEMA,
) -> Dict[str, Any]:
    objects = _extract_objects(latest_result)
    result_code = _result_code(latest_result, objects)
    task = str(latest_result.get("task") or "unknown")

    payload: Dict[str, Any] = {
        "schema": schema,
        "function": "result",
        "timestamp": latest_result.get("timestamp") or now_timestamp(),
        "frame_id": _infer_frame_id(latest_result, frame_id),
        "camera_id": int(latest_result.get("camera_id") or camera_id),
        "source": source,
        "task": task,
        "status": latest_result.get("status", "ok" if result_code == 0 else "no_result"),
        "result": int(result_code),
        "latency_ms": _round_float(latest_result.get("latency_ms"), 3),
        "count": len(objects),
        "objects": objects,
        # 兼容旧上位机/旧 protocol.py：同时保留 predictions。
        "predictions": objects,
    }

    for key in (
        "final_decision",
        "final_label",
        "final_confidence",
        "message",
        "image_width",
        "image_height",
        "input_width",
        "input_height",
        "pipeline_config",
        "model",
    ):
        if key in latest_result:
            payload[key] = _json_safe(latest_result.get(key))

    if isinstance(latest_result.get("detector"), dict):
        det = latest_result.get("detector") or {}
        payload["detector"] = {
            "count": det.get("count"),
            "selected": _json_safe(det.get("selected")),
        }
    if isinstance(latest_result.get("classifier"), dict):
        cls = latest_result.get("classifier") or {}
        payload["classifier"] = {
            "prediction": _json_safe(cls.get("prediction")),
            "topk": _json_safe(cls.get("topk")),
        }
    if isinstance(latest_result.get("roi"), dict):
        payload["roi"] = _json_safe(latest_result.get("roi"))

    timing = latest_result.get("timing_ms") or latest_result.get("timing")
    if isinstance(timing, dict):
        payload["timing"] = _json_safe(timing)

    return payload


def _stream_inference_active(status: Dict[str, Any]) -> bool:
    """判断 C++ stream 是否处于需要推送结果的推理模式。

    目标：Collector 推送线程可以常驻，但只有 Web 端启动 C++ detect/推理流时才推送，
    preview 模式或 stream 停止时不重复推送旧结果。
    """
    if not isinstance(status, dict):
        return False

    running = bool(status.get("running"))
    if not running:
        return False

    mode = str(status.get("stream_mode") or status.get("mode") or "").strip().lower()
    inference_enabled = status.get("inference_enabled")
    if inference_enabled is None:
        inference_enabled = mode in {"detect", "inference", "infer"}

    return bool(inference_enabled) or mode in {"detect", "inference", "infer"}


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "User-Agent": "VisionOps-Collector-CppResultPush/0.8.5",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {"ok": True, "status": resp.status}
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"data": data}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gateway HTTP {exc.code}: {detail or exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(f"POST 到 Gateway 失败: {url}, error={exc}") from exc


class CppResultPushService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.running = False
        self.gateway_url = DEFAULT_GATEWAY_URL
        self.fps = DEFAULT_PUSH_FPS
        self.interval_s = 1.0 / max(0.1, DEFAULT_PUSH_FPS)
        self.camera_id = DEFAULT_CAMERA_ID
        self.timeout_sec = DEFAULT_TIMEOUT_SEC
        self.push_empty = DEFAULT_PUSH_EMPTY
        self.dedupe = DEFAULT_DEDUPE
        self.require_inference = DEFAULT_REQUIRE_INFERENCE
        self.source = "cpp_latest_result"
        self.schema = DEFAULT_SCHEMA

        self.frame_counter = 0
        self.push_success = 0
        self.push_failed = 0
        self.push_skipped = 0
        self.loop_errors = 0
        self.last_pushed_key = ""
        self.latest_result: Optional[Dict[str, Any]] = None
        self.latest_payload: Optional[Dict[str, Any]] = None
        self.latest_response: Optional[Dict[str, Any]] = None
        self.latest_stream_status: Optional[Dict[str, Any]] = None
        self.latest_error = ""
        self.started_at = 0.0
        self.updated_at = 0.0

    def start(
        self,
        *,
        gateway_url: Optional[str] = None,
        fps: Optional[float] = None,
        camera_id: Optional[int] = None,
        timeout_sec: Optional[float] = None,
        push_empty: Optional[bool] = None,
        dedupe: Optional[bool] = None,
        require_inference: Optional[bool] = None,
        source: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.stop(join_timeout=1.0)

        fps_value = max(0.1, float(fps if fps is not None else DEFAULT_PUSH_FPS))
        with self._lock:
            self.gateway_url = str(gateway_url or DEFAULT_GATEWAY_URL)
            self.fps = fps_value
            self.interval_s = 1.0 / fps_value
            self.camera_id = int(camera_id if camera_id is not None else DEFAULT_CAMERA_ID)
            self.timeout_sec = float(timeout_sec if timeout_sec is not None else DEFAULT_TIMEOUT_SEC)
            self.push_empty = bool(DEFAULT_PUSH_EMPTY if push_empty is None else push_empty)
            self.dedupe = bool(DEFAULT_DEDUPE if dedupe is None else dedupe)
            self.require_inference = bool(DEFAULT_REQUIRE_INFERENCE if require_inference is None else require_inference)
            self.source = str(source or "cpp_latest_result")
            self.schema = str(schema or DEFAULT_SCHEMA)
            self.frame_counter = 0
            self.push_success = 0
            self.push_failed = 0
            self.push_skipped = 0
            self.loop_errors = 0
            self.last_pushed_key = ""
            self.latest_result = None
            self.latest_payload = None
            self.latest_response = None
            self.latest_stream_status = None
            self.latest_error = ""
            self.started_at = time.time()
            self.updated_at = 0.0
            self.running = True
            self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._loop,
            name="visionops-cpp-result-push",
            daemon=True,
        )
        self._thread.start()
        return self.status()

    def stop(self, join_timeout: float = 1.0) -> Dict[str, Any]:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=join_timeout)
        with self._lock:
            self.running = False
            self._thread = None
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "running": self.running,
                "gateway_url": self.gateway_url,
                "fps": self.fps,
                "interval_ms": round(self.interval_s * 1000.0, 1),
                "camera_id": self.camera_id,
                "timeout_sec": self.timeout_sec,
                "push_empty": self.push_empty,
                "dedupe": self.dedupe,
                "require_inference": self.require_inference,
                "source": self.source,
                "schema": self.schema,
                "frame_counter": self.frame_counter,
                "push_success": self.push_success,
                "push_failed": self.push_failed,
                "push_skipped": self.push_skipped,
                "loop_errors": self.loop_errors,
                "last_pushed_key": self.last_pushed_key,
                "latest_error": self.latest_error,
                "latest_response": self.latest_response,
                "latest_stream_status": self.latest_stream_status,
                "latest_payload": self.latest_payload,
                "started_at": self.started_at,
                "updated_at": self.updated_at,
                "autostart_enabled": DEFAULT_AUTOSTART,
            }

    def run_once(self, *, force: bool = True) -> Dict[str, Any]:
        return self._run_once(force=force)

    def _loop(self) -> None:
        logger.info("C++ latest_result Gateway 自动推送启动: url=%s fps=%s", self.gateway_url, self.fps)
        while not self._stop_event.is_set():
            started = time.time()
            try:
                self._run_once(force=False)
            except Exception as exc:
                logger.exception("C++ latest_result 自动推送失败")
                with self._lock:
                    self.loop_errors += 1
                    self.latest_error = str(exc)
                    self.updated_at = time.time()
            elapsed = time.time() - started
            wait_s = max(0.02, self.interval_s - elapsed)
            self._stop_event.wait(timeout=wait_s)
        logger.info("C++ latest_result Gateway 自动推送已停止")

    def _run_once(self, *, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            self.frame_counter += 1
            frame_id = self.frame_counter
            gateway_url = self.gateway_url
            camera_id = self.camera_id
            timeout_sec = self.timeout_sec
            push_empty = self.push_empty
            dedupe = self.dedupe
            require_inference = self.require_inference
            source = self.source
            schema = self.schema
            last_key = self.last_pushed_key

        stream_status: Dict[str, Any] = {}
        if require_inference and not force:
            try:
                stream_status = get_json("/stream/status")
            except CppInferenceError as exc:
                with self._lock:
                    self.push_skipped += 1
                    self.latest_stream_status = {"error": str(exc)}
                    self.latest_error = f"skipped: C++ stream/status unavailable: {exc}"
                    self.updated_at = time.time()
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "stream_status_unavailable",
                    "error": str(exc),
                }

            if not _stream_inference_active(stream_status):
                with self._lock:
                    self.push_skipped += 1
                    self.latest_stream_status = stream_status
                    self.latest_error = "skipped: C++ inference stream is not running"
                    self.updated_at = time.time()
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "inference_stream_not_running",
                    "stream_status": stream_status,
                }

        try:
            latest = get_json("/stream/latest_result")
        except CppInferenceError as exc:
            raise RuntimeError(f"读取 C++ latest_result 失败: {exc}") from exc

        if not isinstance(latest, dict):
            raise RuntimeError("C++ latest_result 不是 JSON object")

        payload = build_cpp_gateway_payload(
            latest,
            frame_id=frame_id,
            camera_id=camera_id,
            source=source,
            schema=schema,
        )
        key = _dedupe_key(latest, payload)
        result_code = int(payload.get("result", 1))

        if result_code == 1 and not push_empty and not force:
            with self._lock:
                self.push_skipped += 1
                self.latest_result = latest
                self.latest_stream_status = stream_status or self.latest_stream_status
                self.latest_payload = payload
                self.latest_error = "skipped empty result"
                self.updated_at = time.time()
            return {"ok": True, "skipped": True, "reason": "empty_result", "payload": payload}

        if dedupe and not force and key == last_key:
            with self._lock:
                self.push_skipped += 1
                self.latest_result = latest
                self.latest_stream_status = stream_status or self.latest_stream_status
                self.latest_payload = payload
                self.latest_error = "skipped duplicate frame"
                self.updated_at = time.time()
            return {"ok": True, "skipped": True, "reason": "duplicate", "dedupe_key": key, "payload": payload}

        try:
            response = post_json(gateway_url, payload, timeout=timeout_sec)
            with self._lock:
                self.push_success += 1
                self.last_pushed_key = key
                self.latest_result = latest
                self.latest_stream_status = stream_status or self.latest_stream_status
                self.latest_payload = payload
                self.latest_response = response
                self.latest_error = ""
                self.updated_at = time.time()
            return {
                "ok": True,
                "skipped": False,
                "gateway_url": gateway_url,
                "dedupe_key": key,
                "payload": payload,
                "response": response,
            }
        except Exception as exc:
            with self._lock:
                self.push_failed += 1
                self.latest_result = latest
                self.latest_stream_status = stream_status or self.latest_stream_status
                self.latest_payload = payload
                self.latest_error = str(exc)
                self.updated_at = time.time()
            return {
                "ok": False,
                "gateway_url": gateway_url,
                "dedupe_key": key,
                "payload": payload,
                "error": str(exc),
            }


cpp_result_push_service = CppResultPushService()


def cpp_result_push_autostart_enabled() -> bool:
    return bool(DEFAULT_AUTOSTART)
