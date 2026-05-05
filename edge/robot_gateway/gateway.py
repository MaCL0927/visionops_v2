#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisionOps Robot Gateway V0.2

运行模式：
  1. 触发模式（原有）：
     python edge/robot_gateway/gateway.py --host 0.0.0.0 --port 9100 --infer-url http://127.0.0.1:8082/infer
     
  2. 主动检测模式（新增）：
     python edge/robot_gateway/gateway.py --mode active --camera-url rtsp://admin:pass@192.168.1.64/Streaming/Channels/1 --infer-url http://127.0.0.1:8082/infer --interval 0.5
     
  3. 混合模式：
     python edge/robot_gateway/gateway.py --mode hybrid  # 同时运行TCP服务端+主动检测
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import json
from typing import Optional, Dict, Any

from protocol import (
    extract_frames,
    decode_frame,
    encode_frame,
    validate_camera_request,
    build_mock_result,
    build_error_result,
    normalize_camera_ids,
    build_detection_frame,  # 新增
)

from inference_client import post_image_to_infer
from result_mapper import build_result_from_inference
from image_source import prepare_image_for_request, get_camera_stream  # 新增
from active_detector import ActiveDetector  # 新增

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] robot_gateway: %(message)s",
)

logger = logging.getLogger("robot_gateway")


# ===== 连接管理（主动模式推送） =====
class ClientManager:
    """管理所有连接的客户端，用于主动推送"""
    
    def __init__(self):
        self.clients: set[asyncio.StreamWriter] = set()
    
    def add(self, writer: asyncio.StreamWriter):
        self.clients.add(writer)
    
    def remove(self, writer: asyncio.StreamWriter):
        self.clients.discard(writer)
    
    async def broadcast(self, data: bytes):
        """向所有连接的客户端广播数据"""
        disconnected = set()
        for writer in self.clients:
            try:
                writer.write(data)
                await writer.drain()
            except Exception:
                disconnected.add(writer)
        
        # 移除断开的客户端
        for writer in disconnected:
            self.remove(writer)


# ===== 触发模式处理（原有逻辑） =====
async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    infer_url: str | None = None,
    image_path: str | None = None,
    camera_frame_url: str | None = None,
    frame_dir: str = "/tmp/visionops_robot_gateway_frames",
    camera_timeout: float = 3.0,
    infer_timeout: float = 10.0,
) -> None:
    """原有的触发式检测处理逻辑（保持不变）"""
    peer = writer.get_extra_info("peername")
    logger.info("[触发模式] 客户端已连接: %s", peer)

    buffer = b""

    try:
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                logger.info("[触发模式] 客户端断开: %s", peer)
                break

            buffer += chunk
            frames, buffer = extract_frames(buffer)

            for payload in frames:
                req: Optional[Dict[str, Any]] = None

                try:
                    req = decode_frame(payload)
                    validate_camera_request(req)

                    camera_ids = normalize_camera_ids(req.get("camera"))
                    logger.info(
                        "[RX] triggerindex=%s triggerpos=%s camera=%s",
                        req.get("triggerindex"),
                        req.get("triggerpos"),
                        camera_ids,
                    )

                    if infer_url and (image_path or camera_frame_url):
                        current_image_path = prepare_image_for_request(
                            req=req,
                            image_path=image_path,
                            camera_frame_url=camera_frame_url,
                            frame_dir=frame_dir,
                            timeout=camera_timeout,
                        )

                        logger.info("[IMG] triggerindex=%s image=%s", req.get("triggerindex"), current_image_path)

                        raw = post_image_to_infer(
                            image_path=current_image_path,
                            infer_url=infer_url,
                            timeout=infer_timeout,
                        )

                        result = build_result_from_inference(req, raw)
                    else:
                        result = build_mock_result(req)

                except Exception as e:
                    logger.exception("处理请求失败")
                    result = build_error_result(req, str(e))

                data = encode_frame(result)
                writer.write(data)
                await writer.drain()

                logger.info(
                    "[TX] triggerindex=%s result=%s bytes=%s",
                    result.get("triggerindex"),
                    result.get("result"),
                    len(data),
                )

    except ConnectionResetError:
        logger.warning("[触发模式] 连接被客户端重置: %s", peer)

    except Exception:
        logger.exception("[触发模式] 客户端处理异常: %s", peer)

    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


# ===== 混合模式：TCP服务端 + 主动推送 =====
async def handle_client_hybrid(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    client_manager: ClientManager,
    infer_url: str | None = None,
    image_path: str | None = None,
    camera_frame_url: str | None = None,
    frame_dir: str = "/tmp/visionops_robot_gateway_frames",
    camera_timeout: float = 3.0,
    infer_timeout: float = 10.0,
) -> None:
    """混合模式下的客户端处理：保留触发功能 + 注册到广播列表"""
    peer = writer.get_extra_info("peername")
    logger.info("[混合模式] 客户端已连接: %s", peer)
    
    # 注册到广播列表，用于接收主动检测结果
    client_manager.add(writer)
    
    try:
        # 同时处理该客户端的触发式请求
        await handle_client(
            reader, writer,
            infer_url=infer_url,
            image_path=image_path,
            camera_frame_url=camera_frame_url,
            frame_dir=frame_dir,
            camera_timeout=camera_timeout,
            infer_timeout=infer_timeout,
        )
    finally:
        client_manager.remove(writer)


# ===== 主动检测模式回调 =====
def build_active_detection_frame(detection_result: Dict[str, Any], camera_id: int = 1) -> bytes:
    """将主动检测结果封装为协议帧"""
    return build_detection_frame(detection_result, camera_id)


# ===== 主函数 =====
async def main_async(
    host: str = "0.0.0.0",
    port: int = 9100,
    mode: str = "trigger",  # trigger / active / hybrid
    infer_url: str | None = None,
    image_path: str | None = None,
    camera_frame_url: str | None = None,
    camera_url: str | None = None,  # 新增：RTSP地址
    frame_dir: str = "/tmp/visionops_robot_gateway_frames",
    camera_timeout: float = 3.0,
    infer_timeout: float = 10.0,
    interval: float = 0.5,  # 新增：主动检测间隔（秒）
) -> None:
    
    logger.info("=" * 60)
    logger.info("VisionOps Robot Gateway V0.2 启动")
    logger.info("运行模式: %s", mode)
    logger.info("=" * 60)
    
    if mode == "active":
        # ===== 纯主动检测模式 =====
        if not infer_url:
            raise ValueError("主动检测模式需要设置 --infer-url")
        if not camera_url:
            raise ValueError("主动检测模式需要设置 --camera-url")
        
        logger.info("相机源: %s", camera_url)
        logger.info("推理服务: %s", infer_url)
        logger.info("检测间隔: %.1fs", interval)
        
        detector = ActiveDetector(
            camera_url=camera_url,
            infer_url=infer_url,
            interval=interval,
            infer_timeout=infer_timeout,
            frame_dir=frame_dir,
        )
        
        # 纯主动模式：只打印到日志（改为同步函数）
        def on_detection(result):
            frame = build_active_detection_frame(result)
            logger.info("[主动检测] 推送: %s", frame[:200].decode("utf-8", errors="replace"))
        
        await detector.run(on_detection)
    
    elif mode == "trigger":
        # ===== 纯触发模式（原有逻辑） =====
        async def _client_handler(reader, writer):
            await handle_client(
                reader, writer,
                infer_url=infer_url,
                image_path=image_path,
                camera_frame_url=camera_frame_url,
                frame_dir=frame_dir,
                camera_timeout=camera_timeout,
                infer_timeout=infer_timeout,
            )
        
        server = await asyncio.start_server(_client_handler, host=host, port=port)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        logger.info("触发模式 TCP 服务已启动: %s", addrs)
        
        async with server:
            await server.serve_forever()
    
    elif mode == "hybrid":
        # ===== 混合模式 =====
        if not infer_url:
            raise ValueError("混合模式需要设置 --infer-url")
        if not camera_url:
            raise ValueError("混合模式需要设置 --camera-url (主动检测) 或 --camera-frame-url (触发) 或 --image-path (触发)")
        
        client_manager = ClientManager()
        
        async def _hybrid_client_handler(reader, writer):
            await handle_client_hybrid(
                reader, writer,
                client_manager=client_manager,
                infer_url=infer_url,
                image_path=image_path,
                camera_frame_url=camera_frame_url,
                frame_dir=frame_dir,
                camera_timeout=camera_timeout,
                infer_timeout=infer_timeout,
            )
        
        # 启动TCP服务端
        server = await asyncio.start_server(_hybrid_client_handler, host=host, port=port)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        logger.info("混合模式 TCP 服务已启动: %s", addrs)
        
        # 启动主动检测器
        detector = ActiveDetector(
            camera_url=camera_url,
            infer_url=infer_url,
            interval=interval,
            infer_timeout=infer_timeout,
            frame_dir=frame_dir,
        )
        
        async def on_detection_hybrid(result):
            frame = build_active_detection_frame(result)
            # 同时日志输出和广播
            logger.info("[混合模式] 广播检测结果到 %d 个客户端", len(client_manager.clients))
            await client_manager.broadcast(frame)
        
        # 并行运行TCP服务和主动检测
        async with server:
            await asyncio.gather(
                server.serve_forever(),
                detector.run(on_detection_hybrid),
            )
    
    else:
        raise ValueError(f"未知运行模式: {mode}，支持 trigger / active / hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--mode", default="trigger", 
                       choices=["trigger", "active", "hybrid"],
                       help="运行模式: trigger(触发式) / active(主动检测) / hybrid(混合)")
    
    # 推理相关
    parser.add_argument("--infer-url", default=None, 
                       help="VisionOps 推理服务地址，例如 http://127.0.0.1:8082/infer")
    parser.add_argument("--infer-timeout", type=float, default=10.0)
    
    # 触发模式图像源
    parser.add_argument("--image-path", default=None, 
                       help="触发模式用固定测试图片")
    parser.add_argument("--camera-frame-url", default=None, 
                       help="触发模式相机取帧接口")
    parser.add_argument("--camera-timeout", type=float, default=3.0, 
                       help="相机取帧超时时间")
    
    # 主动模式参数（新增）
    parser.add_argument("--camera-url", default=None, 
                       help="主动检测模式 RTSP 相机地址，例如 rtsp://admin:pass@192.168.1.64/Streaming/Channels/1")
    parser.add_argument("--interval", type=float, default=0.5, 
                       help="主动检测间隔（秒）")
    
    parser.add_argument("--frame-dir", default="/tmp/visionops_robot_gateway_frames", 
                       help="临时图片保存目录")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        main_async(
            host=args.host,
            port=args.port,
            mode=args.mode,
            infer_url=args.infer_url,
            image_path=args.image_path,
            camera_frame_url=args.camera_frame_url,
            camera_url=args.camera_url,
            frame_dir=args.frame_dir,
            camera_timeout=args.camera_timeout,
            infer_timeout=args.infer_timeout,
            interval=args.interval,
        )
    )


if __name__ == "__main__":
    main()

