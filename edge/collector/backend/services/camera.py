#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单例摄像头服务。

v4.7 目标：
1. 后端只保留一个 RTSP/OpenCV 读取源；
2. 后台线程持续刷新 latest_frame；
3. 网页 MJPEG 只用 latest_frame 做低帧率预览；
4. 采集按钮直接保存 latest_frame，不再由浏览器截图回传 RTSP 画面。
"""

import threading
import time
from typing import Generator, Optional

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


def backend_camera_enabled() -> bool:
    src = (CAMERA_SOURCE or "browser").strip().lower()
    return src not in {"", "browser", "none", "false", "0"}


def _normalize_source(source: str):
    source = (source or CAMERA_SOURCE or "browser").strip()
    if source.isdigit():
        return int(source)
    return source


class CameraService:
    """后台单例读帧服务。"""

    def __init__(self, source: Optional[str] = None):
        self.source = _normalize_source(source or CAMERA_SOURCE)
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

    def enabled(self) -> bool:
        return backend_camera_enabled()

    def start(self) -> None:
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
        self._release_cap()
        self._status = "stopped"

    def status(self) -> dict:
        with self._lock:
            has_frame = self._latest_frame is not None
            age_ms = int((time.time() - self._latest_ts) * 1000) if has_frame else None
        return {
            "enabled": self.enabled(),
            "source": str(CAMERA_SOURCE if self.enabled() else "browser"),
            "status": self._status,
            "error": self._error,
            "has_frame": has_frame,
            "latest_age_ms": age_ms,
            "stream_fps": CAMERA_STREAM_FPS,
            "preview_width": CAMERA_PREVIEW_WIDTH,
        }

    def _open_cap(self):
        assert self.cv2 is not None
        self._release_cap()
        self._status = "connecting"
        cap = self.cv2.VideoCapture(self.source)
        # 尽量降低 OpenCV 内部缓存，减少“越播越延迟”。不同后端不一定全部生效。
        try:
            cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            self._status = "error"
            self._error = f"无法打开摄像头源：{CAMERA_SOURCE}"
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
                    if fail_count >= CAMERA_RECONNECT_MAX_FAILS:
                        self._status = "reconnecting"
                        self._release_cap()
                        fail_count = 0
                        time.sleep(1.0)
                    else:
                        time.sleep(0.03)
                    continue

                fail_count = 0
                now = time.time()
                # 保存原始 latest_frame，取图时用；preview_jpeg 单独缩小编码。
                preview = self._resize_for_preview(frame)
                ok_jpg, buf = self.cv2.imencode(
                    ".jpg",
                    preview,
                    [int(self.cv2.IMWRITE_JPEG_QUALITY), int(CAMERA_JPEG_QUALITY)],
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
        if not CAMERA_PREVIEW_WIDTH or CAMERA_PREVIEW_WIDTH <= 0:
            return frame
        h, w = frame.shape[:2]
        if w <= CAMERA_PREVIEW_WIDTH:
            return frame
        scale = CAMERA_PREVIEW_WIDTH / float(w)
        new_h = max(1, int(h * scale))
        return self.cv2.resize(frame, (int(CAMERA_PREVIEW_WIDTH), new_h))

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
        delay = max(0.05, 1.0 / max(CAMERA_STREAM_FPS, 1.0))
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
                # 暂无帧时不要断开浏览器连接，稍后自动恢复。
                time.sleep(0.2)


camera_service = CameraService()


def read_one_jpeg(source: Optional[str] = None) -> bytes:
    # 保留兼容：如果传入临时 source，则单次读取；正常 RTSP 使用单例 latest_frame。
    if source:
        cv2 = _load_cv2()
        if cv2 is None:
            raise RuntimeError("当前 Python 环境未安装 cv2，无法读取 RTSP/本机摄像头")
        cap = cv2.VideoCapture(_normalize_source(source))
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
        raise RuntimeError("后端摄像头未启用：请设置 VISIONOPS_CAMERA_SOURCE 为 RTSP 地址或摄像头编号")
    camera_service.start()
    return camera_service.get_latest_frame_jpeg()


def mjpeg_stream(source: Optional[str] = None) -> Generator[bytes, None, None]:
    if source:
        # v4.7 不建议每个请求单独打开 source，这里只为兼容保留。
        temp_service = CameraService(source)
        temp_service.start()
        try:
            yield from temp_service.mjpeg_stream()
        finally:
            temp_service.stop()
        return
    if not backend_camera_enabled():
        raise RuntimeError("后端摄像头未启用：请设置 VISIONOPS_CAMERA_SOURCE 为 RTSP 地址或摄像头编号")
    yield from camera_service.mjpeg_stream()
