#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生产模式连续检测 + Gateway 推送服务。

职责：
1. 从 Web 后端 camera_service 获取 latest_frame
2. 调用现有 RKNN 推理服务
3. 通过 gateway_push.py 统一推送到 Gateway /push_result
4. 保存 latest_result，供前端生产模式页面轮询显示
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import (
    DEFAULT_DATASET_NAME,
    PRODUCTION_CAMERA_ID,
    PRODUCTION_DETECT_INTERVAL_MS,
    PRODUCTION_GATEWAY_PUSH_URL,
)
from backend.services.camera import backend_camera_enabled, camera_service
from backend.services.gateway_push import push_result_to_gateway
from backend.services.validation_images import (
    get_realtime_image_path,
    save_realtime_image_bytes,
)
from backend.services.validation_infer import classify_image_with_model
from backend.services.settings_store import get_algorithm_runtime_config


logger = logging.getLogger("visionops.production_push")


def _now_timestamp() -> list[int]:
    t = time.time()
    sec = int(t)
    ms = int((t - sec) * 1000)
    return [sec, ms]


def _default_production_interval_ms() -> int:
    try:
        algo = get_algorithm_runtime_config()
        return max(100, int(algo.get("common", {}).get("production_detect_interval_ms") or PRODUCTION_DETECT_INTERVAL_MS))
    except Exception:
        return int(PRODUCTION_DETECT_INTERVAL_MS)


class ProductionPushService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.running = False
        self.model_name = ""
        self.dataset = DEFAULT_DATASET_NAME
        self.gateway_url = PRODUCTION_GATEWAY_PUSH_URL
        self.interval_ms = _default_production_interval_ms()
        self.camera_id = PRODUCTION_CAMERA_ID

        self.frame_id = 0
        self.latest_result: Optional[Dict[str, Any]] = None
        self.latest_error = ""
        self.latest_push_response: Optional[Dict[str, Any]] = None
        self.started_at = 0.0
        self.updated_at = 0.0

    def start(
        self,
        model_name: str,
        dataset: str = DEFAULT_DATASET_NAME,
        gateway_url: str = PRODUCTION_GATEWAY_PUSH_URL,
        interval_ms: int = None,
        camera_id: int = PRODUCTION_CAMERA_ID,
    ) -> Dict[str, Any]:
        model_name = Path(model_name or "").name
        if not model_name:
            raise ValueError("缺少 model_name")

        interval_ms = max(100, int(interval_ms or _default_production_interval_ms()))

        # 避免两个生产检测线程同时运行。
        self.stop(join_timeout=2.0)

        with self._lock:
            self.model_name = model_name
            self.dataset = dataset or DEFAULT_DATASET_NAME
            self.gateway_url = gateway_url or PRODUCTION_GATEWAY_PUSH_URL
            self.interval_ms = interval_ms
            self.camera_id = int(camera_id or PRODUCTION_CAMERA_ID)
            self.frame_id = 0
            self.latest_result = None
            self.latest_error = ""
            self.latest_push_response = None
            self.started_at = time.time()
            self.updated_at = 0.0
            self.running = True
            self._stop_event.clear()

        if backend_camera_enabled():
            camera_service.start()

        self._thread = threading.Thread(
            target=self._loop,
            name="visionops-production-push",
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
                "model_name": self.model_name,
                "dataset": self.dataset,
                "gateway_url": self.gateway_url,
                "interval_ms": self.interval_ms,
                "camera_id": self.camera_id,
                "frame_id": self.frame_id,
                "latest_error": self.latest_error,
                "latest_push_response": self.latest_push_response,
                "latest_result": self.latest_result,
                "started_at": self.started_at,
                "updated_at": self.updated_at,
                "camera": camera_service.status(),
            }

    def _loop(self) -> None:
        logger.info(
            "生产连续检测启动: model=%s, dataset=%s, gateway=%s, interval=%sms",
            self.model_name,
            self.dataset,
            self.gateway_url,
            self.interval_ms,
        )

        while not self._stop_event.is_set():
            started = time.time()

            try:
                self._run_once()
            except Exception as e:
                logger.exception("生产连续检测单帧失败")
                with self._lock:
                    self.latest_error = str(e)
                    self.updated_at = time.time()

            elapsed = time.time() - started
            sleep_s = max(0.02, self.interval_ms / 1000.0 - elapsed)
            self._stop_event.wait(timeout=sleep_s)

        logger.info("生产连续检测已停止")

    def _run_once(self) -> None:
        with self._lock:
            model_name = self.model_name
            dataset = self.dataset
            gateway_url = self.gateway_url
            camera_id = self.camera_id
            self.frame_id += 1
            frame_id = self.frame_id

        if not backend_camera_enabled():
            raise RuntimeError("后端摄像头未启用：请设置 VISIONOPS_CAMERA_SOURCE")

        # 1. 从 Web 后端单例摄像头服务取 latest_frame。
        jpeg = camera_service.get_latest_frame_jpeg(quality=90, timeout=2.0)

        # 2. 保存为实时临时图，便于前端生产模式显示和复用现有推理接口。
        frame = save_realtime_image_bytes(dataset, jpeg, ".jpg")
        image_path = get_realtime_image_path(frame["filename"], dataset)

        # 3. 调用现有 RKNN 推理服务。
        infer_result = classify_image_with_model(model_name, image_path)
        infer_result["mode"] = "production"
        infer_result["realtime"] = frame
        infer_result["frame_id"] = frame_id
        infer_result["camera_id"] = camera_id
        infer_result["timestamp"] = _now_timestamp()

        # 4. 统一推送到 Gateway。推送失败不影响最新检测结果保存。
        push_response = push_result_to_gateway(
            infer_result,
            source="production_realtime",
            dataset=dataset,
            model_name=model_name,
            image_id=frame.get("filename", ""),
            camera_id=camera_id,
            gateway_url=gateway_url,
            frame_id=frame_id,
        )
        infer_result["gateway_push"] = push_response

        # 5. 保存最新状态给前端轮询。
        with self._lock:
            self.latest_result = infer_result
            self.latest_error = "" if push_response.get("ok") else str(push_response.get("error") or "")
            self.latest_push_response = push_response
            self.updated_at = time.time()


production_push_service = ProductionPushService()
