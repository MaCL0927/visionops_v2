#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Disabled legacy Python camera service.

VisionOps on LB3576 now uses the C++ camera path only:
- HP60C Angstrong SDK bridge: http://127.0.0.1:18181
- C++ inference service reads: http://127.0.0.1:18181/stream.mjpeg

This module keeps the old symbols so existing imports do not break, but it
never opens RTSP/USB sources and never starts a Python OpenCV reader thread.
"""

from typing import Any, Dict, Generator, Optional


def backend_camera_enabled() -> bool:
    return False


class DisabledCameraService:
    def enabled(self) -> bool:
        return False

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def reload_from_runtime(self, start: bool = False) -> Dict[str, Any]:
        return self.status()

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "type": "disabled",
            "source": "hp60c_sdk_cpp",
            "status": "disabled",
            "error": "旧 Python RTSP/USB 摄像头服务已禁用；请使用 C++/HP60C SDK 相机链路。",
            "has_frame": False,
            "latest_age_ms": None,
            "stream_fps": 0.0,
            "preview_width": 0,
            "jpeg_quality": 0,
            "rtsp_transport": "disabled",
            "usb_backend": "disabled",
            "usb_buffer_size": 0,
            "resolution": "disabled",
        }

    def get_latest_frame_jpeg(self, *args: Any, **kwargs: Any) -> bytes:
        raise RuntimeError("旧 Python 摄像头服务已禁用；请使用 C++/HP60C SDK snapshot 接口。")

    def get_latest_jpeg(self, *args: Any, **kwargs: Any) -> bytes:
        return self.get_latest_frame_jpeg(*args, **kwargs)

    def mjpeg_stream(self) -> Generator[bytes, None, None]:
        raise RuntimeError("旧 Python MJPEG 摄像头流已禁用；请使用 C++/HP60C SDK 预览。")


camera_service = DisabledCameraService()


def read_one_jpeg(source: Optional[str] = None) -> bytes:
    raise RuntimeError("旧 Python 摄像头单帧读取已禁用；请使用 /api/cpp/hp60c_sdk/stream/snapshot.jpg。")


def mjpeg_stream(source: Optional[str] = None) -> Generator[bytes, None, None]:
    raise RuntimeError("旧 Python 摄像头 MJPEG 流已禁用；请使用 /api/cpp/hp60c_sdk/stream/snapshot.jpg。")
