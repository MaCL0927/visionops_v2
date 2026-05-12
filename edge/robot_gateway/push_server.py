#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisionOps Gateway Push Server

职责：
1. TCP 9100：上位机连接后，持续接收检测结果推送
2. HTTP 9101：Web/Collector POST 检测结果
3. Gateway 将 HTTP JSON 转成 *JSON#，广播给所有 TCP 客户端

推荐运行：
    python edge/robot_gateway/push_server.py \
      --tcp-host 0.0.0.0 \
      --tcp-port 9100 \
      --http-host 127.0.0.1 \
      --http-port 9101
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Dict, Set

import uvicorn
from fastapi import FastAPI, Request

from protocol import build_detection_frame, build_error_frame, encode_frame


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] visionops.gateway: %(message)s",
)

logger = logging.getLogger("visionops.gateway")


class ClientManager:
    """
    管理 TCP 上位机连接。
    """

    def __init__(self) -> None:
        self.clients: Set[asyncio.StreamWriter] = set()
        self._lock = asyncio.Lock()

    async def add(self, writer: asyncio.StreamWriter) -> None:
        async with self._lock:
            self.clients.add(writer)

    async def remove(self, writer: asyncio.StreamWriter) -> None:
        async with self._lock:
            self.clients.discard(writer)

    async def count(self) -> int:
        async with self._lock:
            return len(self.clients)

    async def broadcast(self, data: bytes) -> int:
        """
        向所有 TCP 客户端广播。
        返回成功推送的客户端数量。
        """
        async with self._lock:
            clients = list(self.clients)

        if not clients:
            return 0

        ok_count = 0
        disconnected = []

        for writer in clients:
            try:
                writer.write(data)
                await writer.drain()
                ok_count += 1
            except Exception:
                disconnected.append(writer)

        for writer in disconnected:
            await self.remove(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

        return ok_count


client_manager = ClientManager()
app = FastAPI(title="VisionOps Gateway Push API")


async def tcp_client_handler(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """
    上位机 TCP 连接处理。

    当前模式下：
    - 上位机不需要发送触发包
    - 只要保持连接，就会收到 Gateway 主动推送的 *JSON#
    - 如果上位机发送心跳或任意数据，这里会读取并忽略，避免缓冲区堆积
    """
    peer = writer.get_extra_info("peername")
    logger.info("[TCP] 上位机已连接: %s", peer)

    await client_manager.add(writer)

    try:
        while True:
            data = await reader.read(1024)
            if not data:
                logger.info("[TCP] 上位机断开: %s", peer)
                break

            logger.debug("[TCP] 收到上位机数据 peer=%s data=%r", peer, data[:200])

    except ConnectionResetError:
        logger.warning("[TCP] 上位机连接被重置: %s", peer)

    except Exception:
        logger.exception("[TCP] 上位机连接异常: %s", peer)

    finally:
        await client_manager.remove(writer)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "visionops-gateway",
        "tcp_clients": await client_manager.count(),
    }


@app.post("/push_result")
async def push_result(request: Request) -> Dict[str, Any]:
    """
    Web/Collector 调用该接口推送检测结果。

    输入可以直接是 engine.py /infer 返回的 JSON，例如：
        {
          "task": "detection",
          "latency_ms": 64.5,
          "predictions": [...]
        }
    """
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON object")

        frame_id = payload.get("frame_id")
        camera_id = int(payload.get("camera_id", 1))

        # v0.8.5: Collector may already send a complete custom gateway frame.
        # If function=result is present, preserve the custom JSON instead of
        # rebuilding a legacy detection-only frame and losing ROI/OBB/mask fields.
        if payload.get("function") == "result":
            payload.setdefault("camera_id", camera_id)
            if frame_id is not None:
                payload.setdefault("frame_id", frame_id)
            frame = encode_frame(payload)
        else:
            frame = build_detection_frame(
                inference_result=payload,
                camera_id=camera_id,
                frame_id=frame_id,
            )

    except Exception as e:
        logger.exception("[HTTP] 构造推送帧失败")
        frame = build_error_frame(str(e))

    pushed = await client_manager.broadcast(frame)

    logger.info(
        "[HTTP->TCP] pushed_clients=%s bytes=%s",
        pushed,
        len(frame),
    )

    return {
        "ok": True,
        "pushed_clients": pushed,
        "bytes": len(frame),
    }


async def main_async(args: argparse.Namespace) -> None:
    tcp_server = await asyncio.start_server(
        tcp_client_handler,
        host=args.tcp_host,
        port=args.tcp_port,
    )

    tcp_addrs = ", ".join(str(sock.getsockname()) for sock in tcp_server.sockets or [])
    logger.info("[TCP] 推送服务已启动: %s", tcp_addrs)
    logger.info("[HTTP] 接收服务启动: http://%s:%s", args.http_host, args.http_port)

    config = uvicorn.Config(
        app,
        host=args.http_host,
        port=args.http_port,
        log_level="info",
    )
    http_server = uvicorn.Server(config)

    async with tcp_server:
        await asyncio.gather(
            tcp_server.serve_forever(),
            http_server.serve(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp-host", default="0.0.0.0")
    parser.add_argument("--tcp-port", type=int, default=9100)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=9101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
