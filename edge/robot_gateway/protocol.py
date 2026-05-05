#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisionOps Robot Gateway 协议工具。

当前协议格式：
    *{JSON}#

注意：
1. 这里要求机械臂实际发送的是合法 JSON。
2. 对接文档里的 // 中文注释不能出现在真实 TCP 报文中。
3. 当前版本只做协议闭环，不调用真实模型。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple


FRAME_START = b"*"
FRAME_END = b"#"


def now_timestamp() -> List[int]:
    """
    暂时返回 [秒, 毫秒]。
    后续可以按对接方要求改成机械臂需要的时间格式。
    """
    t = time.time()
    sec = int(t)
    ms = int((t - sec) * 1000)
    return [sec, ms]


def extract_frames(buffer: bytes) -> Tuple[List[bytes], bytes]:
    """
    从 TCP 字节流中提取完整帧。

    输入可能是：
        b'*{"function":"camera"}#'
        b'xxx*{"function":"camera"}#yyy'
        b'*{"function":"camera"}#*{"function":"camera"}#'

    返回：
        frames: [b'{"function":"camera"}']
        remaining_buffer: 未处理完的残留字节
    """
    frames: List[bytes] = []

    while True:
        start = buffer.find(FRAME_START)
        if start < 0:
            # 没有找到起始符，丢弃无效数据
            return frames, b""

        end = buffer.find(FRAME_END, start + 1)
        if end < 0:
            # 找到了 *，但还没收到 #，保留剩余数据继续等
            return frames, buffer[start:]

        payload = buffer[start + 1:end].strip()
        if payload:
            frames.append(payload)

        buffer = buffer[end + 1:]


def decode_frame(payload: bytes) -> Dict[str, Any]:
    """
    将一帧 JSON payload 解析成 dict。
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"协议解码失败，不是 UTF-8: {e}") from e

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}; raw={text!r}") from e

    if not isinstance(data, dict):
        raise ValueError("JSON 根对象必须是 object")

    return data


def encode_frame(data: Dict[str, Any]) -> bytes:
    """
    将 dict 编码成协议帧：
        *{JSON}#
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return FRAME_START + payload.encode("utf-8") + FRAME_END


def normalize_camera_ids(camera: Any) -> List[int]:
    """
    对接文档中 camera 是字符串，例如 "1,2,3"。
    这里兼容：
        "1,2,3"
        "1"
        [1, 2, 3]
        1
    """
    if camera is None:
        return []

    if isinstance(camera, int):
        return [camera]

    if isinstance(camera, list):
        out = []
        for item in camera:
            try:
                out.append(int(item))
            except Exception:
                pass
        return out

    if isinstance(camera, str):
        out = []
        for part in camera.replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except Exception:
                pass
        return out

    return []


def validate_camera_request(req: Dict[str, Any]) -> None:
    """
    校验机械臂触发请求。
    """
    if req.get("function") != "camera":
        raise ValueError(f"不支持的 function: {req.get('function')}")

    if "triggerindex" not in req:
        raise ValueError("缺少 triggerindex")

    if "triggerpos" not in req:
        raise ValueError("缺少 triggerpos")

    camera_ids = normalize_camera_ids(req.get("camera"))
    if not camera_ids:
        raise ValueError("缺少 camera，或 camera 格式无效")


def build_mock_result(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    构造第一版模拟检测结果。

    后续这里会替换成：
        1. 触发相机
        2. 调用 RKNN 推理服务
        3. 将 bbox / mask / OBB 转成机械臂坐标
        4. 返回真实结果
    """
    camera_ids = normalize_camera_ids(req.get("camera"))
    camera_id = camera_ids[0] if camera_ids else 1

    return {
        "function": "result",
        "timestamp": req.get("timestamp", now_timestamp()),
        "triggerpos": req.get("triggerpos", 0),
        "triggerindex": req.get("triggerindex", 0),

        # 第一版约定：
        # 0 = 正常返回
        # 1 = 未检测到目标
        # -1 = 协议或内部错误
        # 9999 = 标定状态，按对接文档保留
        "result": 0,

        # 第一版先填 0，后续接深度/测距/标定后再更新
        "distance": 0.0,

        # 返回实际参与检测的相机 ID
        "camera_id": camera_id,

        # 第一版先空字符串，后续如有扫码器再接入
        "barcodes": "",

        # 按对接文档保留 types 字段
        # type: 1 左臂，2 右臂
        "types": [
            {
                "type": 1,
                "x": 100.01,
                "y": 200.02,
                "z": 300.03,
                "ox": 0.0,
                "oy": 0.0,
                "oz": 0.0,
                "ow": 1.0,
                "length": 60.06,
                "width": 70.07,
                "height": 80.08,
            }
        ],
    }


def build_error_result(req: Dict[str, Any] | None, message: str) -> Dict[str, Any]:
    """
    构造错误返回。
    """
    req = req or {}

    return {
        "function": "result",
        "timestamp": req.get("timestamp", now_timestamp()),
        "triggerpos": req.get("triggerpos", 0),
        "triggerindex": req.get("triggerindex", 0),
        "result": -1,
        "distance": 0.0,
        "camera_id": 0,
        "barcodes": "",
        "error": message,
        "types": [],
    }

def build_detection_frame(
    inference_result: Dict[str, Any],
    camera_id: int = 1
) -> bytes:
    """
    将推理结果封装为主动检测协议帧。
    
    协议格式：
    *{
      "function": "result",
      "timestamp": [秒, 毫秒],
      "camera_id": 1,
      "task": "检测任务类型",
      "predictions": [...]
    }#
    """
    predictions = inference_result.get("predictions", [])
    task = inference_result.get("task", "unknown")
    
    # 只保留关键字段，简化输出
    simplified_predictions = []
    for pred in predictions:
        simplified_predictions.append({
            "class_id": pred.get("class_id"),
            "class_name": pred.get("class_name"),
            "confidence": pred.get("confidence"),
            "bbox": pred.get("bbox"),
            "center": pred.get("center"),
        })
    
    frame = {
        "function": "result",
        "timestamp": now_timestamp(),
        "camera_id": camera_id,
        "task": task,
        "predictions": simplified_predictions,
    }
    
    return encode_frame(frame)
