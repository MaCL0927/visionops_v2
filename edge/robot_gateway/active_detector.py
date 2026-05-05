#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主动检测器。

从RTSP相机持续取帧，定时调用推理服务，通过回调输出结果。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from pathlib import Path
from typing import Callable, Dict, Any, Union, Awaitable

from image_source import get_camera_stream
from inference_client import post_image_to_infer

logger = logging.getLogger("active_detector")


class ActiveDetector:
    """主动检测器"""
    
    def __init__(
        self,
        camera_url: str,
        infer_url: str,
        interval: float = 0.5,
        infer_timeout: float = 10.0,
        frame_dir: str = "/tmp/visionops_robot_gateway_frames",
    ):
        self.camera_url = camera_url
        self.infer_url = infer_url
        self.interval = interval
        self.infer_timeout = infer_timeout
        
        self.frame_dir = Path(frame_dir).expanduser().resolve()
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        
        self._running = False
        self._frame_count = 0
    
    async def run(self, on_detection: Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]):
        """
        开始主动检测循环。
        
        Args:
            on_detection: 回调函数，可以是同步函数或异步函数，接收推理结果JSON
        """
        self._running = True
        logger.info("主动检测器启动: camera=%s, infer=%s, interval=%.1fs", 
                     self.camera_url, self.infer_url, self.interval)
        
        try:
            while self._running:
                start_time = time.time()
                
                try:
                    # 1. 从相机抓取一帧
                    img_path = await self._capture_frame()
                    
                    # 2. 调用推理服务
                    result = await self._infer(img_path)
                    
                    # 3. 回调输出（自动识别同步/异步函数）
                    if inspect.iscoroutinefunction(on_detection):
                        await on_detection(result)
                    else:
                        on_detection(result)
                    
                except Exception as e:
                    logger.error("主动检测循环出错: %s", e)
                
                # 4. 控制检测频率
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        finally:
            self._running = False
            logger.info("主动检测器已停止，共检测 %d 帧", self._frame_count)
    
    async def _capture_frame(self) -> str:
        """从RTSP相机抓取一帧并保存"""
        self._frame_count += 1
        ts = int(time.time() * 1000)
        filename = f"active_frame_{self._frame_count:06d}_{ts}.jpg"
        out_path = self.frame_dir / filename
        
        # 在线程池中执行OpenCV操作（OpenCV是同步的）
        loop = asyncio.get_event_loop()
        frame_data = await loop.run_in_executor(
            None, get_camera_stream, self.camera_url
        )
        
        out_path.write_bytes(frame_data)
        logger.debug("抓取第 %d 帧: %s", self._frame_count, out_path)
        
        return str(out_path)
    
    async def _infer(self, image_path: str) -> Dict[str, Any]:
        """调用推理服务"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, post_image_to_infer, image_path, self.infer_url, self.infer_timeout
        )
        return result
    
    def stop(self):
        """停止检测"""
        self._running = False
