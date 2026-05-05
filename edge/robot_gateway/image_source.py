#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot Gateway 图像来源。

当前支持：
1. 固定测试图片：--image-path
2. 边缘端相机接口：--camera-frame-url，例如 http://127.0.0.1:8090/api/camera/frame
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Any, Dict


def prepare_image_for_request(
    req: Dict[str, Any],
    image_path: str | None = None,
    camera_frame_url: str | None = None,
    frame_dir: str = "/tmp/visionops_robot_gateway_frames",
    timeout: float = 3.0,
) -> str:
    """
    为一次机械臂触发准备待推理图片。

    优先级：
    1. 如果传了 camera_frame_url，则从相机接口抓一帧
    2. 否则使用 image_path 固定图片
    """
    if camera_frame_url:
        return fetch_camera_frame(
            req=req,
            camera_frame_url=camera_frame_url,
            frame_dir=frame_dir,
            timeout=timeout,
        )

    if image_path:
        p = Path(image_path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"测试图片不存在: {p}")
        return str(p)

    raise ValueError("缺少图像来源：请设置 --camera-frame-url 或 --image-path")


def fetch_camera_frame(
    req: Dict[str, Any],
    camera_frame_url: str,
    frame_dir: str = "/tmp/visionops_robot_gateway_frames",
    timeout: float = 3.0,
) -> str:
    """
    从边缘端相机接口获取一帧 JPEG，并保存到临时目录。

    推荐相机接口：
        http://127.0.0.1:8090/api/camera/frame
    """
    out_dir = Path(frame_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    triggerindex = req.get("triggerindex", "unknown")
    triggerpos = req.get("triggerpos", "unknown")
    ts_ms = int(time.time() * 1000)

    filename = f"trigger_{triggerindex}_pos_{triggerpos}_{ts_ms}.jpg"
    out_path = out_dir / filename

    request = urllib.request.Request(
        camera_frame_url,
        method="GET",
        headers={
            "User-Agent": "VisionOps-Robot-Gateway/0.1",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()

    except Exception as e:
        raise RuntimeError(f"获取相机帧失败: url={camera_frame_url}, error={e}") from e

    if not data:
        raise RuntimeError("获取相机帧失败：返回内容为空")

    # 不强制要求 Content-Type 必须是 image/jpeg，避免某些服务没带 header。
    out_path.write_bytes(data)
    return str(out_path)

import cv2
import logging

logger = logging.getLogger("image_source")

def get_camera_stream(rtsp_url: str, timeout: float = 5.0) -> bytes:
    """
    从RTSP相机流中抓取一帧JPEG。
    
    Args:
        rtsp_url: RTSP地址，例如 rtsp://admin:password@192.168.1.64:554/Streaming/Channels/1
        timeout: 超时时间（秒）
    
    Returns:
        JPEG格式的字节数据
    """
    logger.debug("打开RTSP流: %s", rtsp_url)
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        raise RuntimeError(f"无法打开RTSP流: {rtsp_url}")
    
    try:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("读取帧失败")
        
        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    
    finally:
        cap.release()
