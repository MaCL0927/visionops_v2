#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单例摄像头服务。

v2.1 重点：RTSP 网络相机参数 + 预览参数从 runtime_overrides.yaml 动态生效。
1. Web 设置保存后可以调用 camera_service.reload_from_runtime() 立即重连；
2. RTSP IP/端口/通道/账号/密码/URL/TCP/UDP 生效；
3. 预览 FPS、预览宽度、JPEG 质量生效；
4. 保持原来的 latest_frame / MJPEG 预览 / 采集逻辑不变。
"""

import os
import threading
import time
from typing import Any, Dict, Generator, Optional, Tuple

from backend.config import (
    CAMERA_SOURCE,
    CAMERA_STREAM_FPS,
    CAMERA_PREVIEW_WIDTH,
    CAMERA_JPEG_QUALITY,
    CAMERA_RECONNECT_MAX_FAILS,
)


def _load_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def _normalize_source(source: str):
    source = (source or CAMERA_SOURCE or "browser").strip()
    if source.isdigit():
        return int(source)
    return source


def _fallback_camera_config() -> Dict[str, Any]:
    return {
        "enabled": str(CAMERA_SOURCE or "browser").strip().lower() not in {"", "browser", "none", "false", "0"},
        "type": "rtsp" if str(CAMERA_SOURCE or "").startswith("rtsp://") else "usb",
        "source": CAMERA_SOURCE,
        "usb_backend": "opencv",
        "usb_buffer_size": 1,
        "rtsp_transport": "tcp",
        "stream_fps": float(CAMERA_STREAM_FPS),
        "preview_width": int(CAMERA_PREVIEW_WIDTH),
        "jpeg_quality": int(CAMERA_JPEG_QUALITY),
        "reconnect_max_fails": int(CAMERA_RECONNECT_MAX_FAILS),
        "resolution_width": 0,
        "resolution_height": 0,
    }


def _runtime_camera_config() -> Dict[str, Any]:
    """延迟导入 settings_store，避免模块导入阶段形成循环依赖。"""
    try:
        from backend.services.settings_store import get_camera_runtime_config
        return get_camera_runtime_config()
    except Exception:
        return _fallback_camera_config()


def backend_camera_enabled() -> bool:
    return bool(_runtime_camera_config().get("enabled", False))


def _is_rtsp_source(source: Any) -> bool:
    return isinstance(source, str) and source.strip().lower().startswith("rtsp://")


def _set_ffmpeg_rtsp_options(transport: str) -> None:
    transport = (transport or "tcp").strip().lower()
    if transport not in {"tcp", "udp"}:
        transport = "tcp"
    # stimeout 单位为微秒，避免 RTSP 断流时长时间卡住。
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}|stimeout;5000000"


def _open_video_capture(
    cv2,
    source: Any,
    rtsp_transport: str = "tcp",
    resolution: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = None,
    buffer_size: int = 1,
):
    """打开摄像头源。RTSP 优先使用 FFMPEG backend，USB/OpenCV 使用普通 VideoCapture。"""
    if _is_rtsp_source(source):
        _set_ffmpeg_rtsp_options(rtsp_transport)
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) if hasattr(cv2, "CAP_FFMPEG") else cv2.VideoCapture(source)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size or 1))
    except Exception:
        pass

    if fps and float(fps) > 0:
        try:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        except Exception:
            pass

    if resolution:
        width, height = resolution
        if width > 0:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            except Exception:
                pass
        if height > 0:
            try:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            except Exception:
                pass
    return cap


class CameraService:
    """后台单例读帧服务。"""

    def __init__(self, source: Optional[str] = None):
        self.cv2 = _load_cv2()
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_ts = 0.0
        self._status = "stopped"
        self._error = ""
        self._last_runtime_config: Dict[str, Any] = {}
        self._apply_config(_runtime_camera_config(), override_source=source)

    def _apply_config(self, cfg: Dict[str, Any], override_source: Optional[str] = None) -> None:
        self._last_runtime_config = dict(cfg or {})
        self.camera_type = str(cfg.get("type") or "rtsp").strip().lower()
        self.source = _normalize_source(override_source or str(cfg.get("source") or CAMERA_SOURCE))
        self.usb_backend = str(cfg.get("usb_backend") or "opencv").strip().lower()
        self.usb_buffer_size = int(cfg.get("usb_buffer_size") or 1)
        self.rtsp_transport = str(cfg.get("rtsp_transport") or "tcp").lower()
        self.stream_fps = float(cfg.get("stream_fps") or CAMERA_STREAM_FPS or 6.0)
        self.preview_width = int(cfg.get("preview_width") or CAMERA_PREVIEW_WIDTH or 960)
        self.jpeg_quality = int(cfg.get("jpeg_quality") or CAMERA_JPEG_QUALITY or 75)
        self.reconnect_max_fails = int(cfg.get("reconnect_max_fails") or CAMERA_RECONNECT_MAX_FAILS or 30)
        self.resolution_width = int(cfg.get("resolution_width") or 0)
        self.resolution_height = int(cfg.get("resolution_height") or 0)
        self._enabled = bool(cfg.get("enabled", True)) and str(self.source).strip().lower() not in {"", "browser", "none", "false", "0"}

    def enabled(self) -> bool:
        return bool(self._enabled)

    def reload_from_runtime(self, start: bool = True) -> Dict[str, Any]:
        """重新读取 runtime_overrides.yaml 并重连摄像头。"""
        was_running = self._thread is not None and self._thread.is_alive()
        self.stop()
        self._apply_config(_runtime_camera_config())
        with self._lock:
            self._latest_frame = None
            self._latest_jpeg = None
            self._latest_ts = 0.0
        if start or was_running:
            self.start()
        return self.status()

    def start(self) -> None:
        # 每次启动前重新读取一次配置，确保外部修改配置文件后也能生效。
        if not (self._thread and self._thread.is_alive()):
            self._apply_config(_runtime_camera_config())

        if not self.enabled():
            self._status = "disabled"
            return
        if self.cv2 is None:
            self._status = "error"
            self._error = "当前 Python 环境未安装 cv2，无法读取 RTSP/本机摄像头"
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, name="visionops-camera-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._release_cap()
        self._status = "stopped"

    def status(self) -> dict:
        with self._lock:
            has_frame = self._latest_frame is not None
            age_ms = int((time.time() - self._latest_ts) * 1000) if has_frame else None
        return {
            "enabled": self.enabled(),
            "type": self.camera_type,
            "source": str(self.source if self.enabled() else "browser"),
            "status": self._status,
            "error": self._error,
            "has_frame": has_frame,
            "latest_age_ms": age_ms,
            "stream_fps": self.stream_fps,
            "preview_width": self.preview_width,
            "jpeg_quality": self.jpeg_quality,
            "rtsp_transport": self.rtsp_transport,
            "usb_backend": self.usb_backend,
            "usb_buffer_size": self.usb_buffer_size,
            "resolution": f"{self.resolution_width}x{self.resolution_height}" if self.resolution_width and self.resolution_height else "auto",
        }

    def _open_cap(self):
        assert self.cv2 is not None
        self._release_cap()
        self._status = "connecting"
        cap = _open_video_capture(
            self.cv2,
            self.source,
            rtsp_transport=self.rtsp_transport,
            resolution=(self.resolution_width, self.resolution_height),
            fps=self.stream_fps,
            buffer_size=self.usb_buffer_size if self.camera_type == "usb" else 1,
        )
        if not cap.isOpened():
            self._status = "error"
            self._error = f"无法打开摄像头源：{self.source}"
            cap.release()
            return None
        self._status = "running"
        self._error = ""
        return cap

    def _release_cap(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _reader_loop(self) -> None:
        fail_count = 0
        while not self._stop_event.is_set():
            try:
                if self._cap is None:
                    self._cap = self._open_cap()
                    if self._cap is None:
                        time.sleep(1.0)
                        continue

                ok, frame = self._cap.read()
                if not ok or frame is None:
                    fail_count += 1
                    self._error = f"读取帧失败，连续失败 {fail_count} 次"
                    if fail_count >= self.reconnect_max_fails:
                        self._status = "reconnecting"
                        self._release_cap()
                        fail_count = 0
                        time.sleep(1.0)
                    else:
                        time.sleep(0.03)
                    continue

                fail_count = 0
                now = time.time()
                preview = self._resize_for_preview(frame)
                ok_jpg, buf = self.cv2.imencode(
                    ".jpg",
                    preview,
                    [int(self.cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
                )
                if ok_jpg:
                    with self._lock:
                        self._latest_frame = frame.copy()
                        self._latest_jpeg = buf.tobytes()
                        self._latest_ts = now
                        self._status = "running"
                        self._error = ""
            except Exception as e:
                self._status = "error"
                self._error = str(e)
                self._release_cap()
                time.sleep(1.0)

        self._release_cap()

    def _resize_for_preview(self, frame):
        if not self.preview_width or self.preview_width <= 0:
            return frame
        h, w = frame.shape[:2]
        if w <= self.preview_width:
            return frame
        scale = self.preview_width / float(w)
        new_h = max(1, int(h * scale))
        return self.cv2.resize(frame, (int(self.preview_width), new_h))

    def get_latest_jpeg(self, timeout: float = 3.0) -> bytes:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._latest_jpeg:
                    return bytes(self._latest_jpeg)
            time.sleep(0.03)
        raise RuntimeError(f"暂未获取到摄像头画面：{self._error or self._status}")

    def get_latest_frame_jpeg(self, quality: int = 90, timeout: float = 3.0) -> bytes:
        if self.cv2 is None:
            raise RuntimeError("当前 Python 环境未安装 cv2，无法读取 RTSP/本机摄像头")
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is not None:
                ok, buf = self.cv2.imencode(".jpg", frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                if not ok:
                    raise RuntimeError("摄像头帧 JPEG 编码失败")
                return buf.tobytes()
            time.sleep(0.03)
        raise RuntimeError(f"暂未获取到摄像头画面：{self._error or self._status}")

    def mjpeg_stream(self) -> Generator[bytes, None, None]:
        self.start()
        delay = max(0.05, 1.0 / max(float(self.stream_fps or 1.0), 1.0))
        last_sent = 0.0
        while True:
            now = time.time()
            wait = delay - (now - last_sent)
            if wait > 0:
                time.sleep(wait)
            try:
                jpeg = self.get_latest_jpeg(timeout=1.0)
                last_sent = time.time()
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            except Exception:
                time.sleep(0.2)


camera_service = CameraService()


def read_one_jpeg(source: Optional[str] = None) -> bytes:
    if source:
        cv2 = _load_cv2()
        if cv2 is None:
            raise RuntimeError("当前 Python 环境未安装 cv2，无法读取 RTSP/本机摄像头")
        cfg = _runtime_camera_config()
        cap = _open_video_capture(
            cv2,
            _normalize_source(source),
            rtsp_transport=str(cfg.get("rtsp_transport") or "tcp"),
            resolution=(int(cfg.get("resolution_width") or 0), int(cfg.get("resolution_height") or 0)),
            fps=float(cfg.get("stream_fps") or 0),
            buffer_size=int(cfg.get("usb_buffer_size") or 1),
        )
        try:
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头源：{source}")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头已打开，但读取帧失败")
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise RuntimeError("摄像头帧 JPEG 编码失败")
            return buf.tobytes()
        finally:
            cap.release()
    if not backend_camera_enabled():
        raise RuntimeError("后端摄像头未启用：请在系统设置中配置 RTSP 或 USB 相机")
    camera_service.start()
    return camera_service.get_latest_frame_jpeg()


def mjpeg_stream(source: Optional[str] = None) -> Generator[bytes, None, None]:
    if source:
        temp_service = CameraService(source)
        temp_service.start()
        try:
            yield from temp_service.mjpeg_stream()
        finally:
            temp_service.stop()
        return
    if not backend_camera_enabled():
        raise RuntimeError("后端摄像头未启用：请在系统设置中配置 RTSP 或 USB 相机")
    yield from camera_service.mjpeg_stream()
