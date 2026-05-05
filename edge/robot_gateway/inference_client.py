#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot Gateway 调用 VisionOps 推理服务的客户端。

当前版本：
1. 读取一张本地测试图片
2. multipart/form-data POST 到 http://127.0.0.1:8082/infer
3. 返回 engine.py 的原始 JSON 结果
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict


def post_image_to_infer(
    image_path: str,
    infer_url: str,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    path = Path(image_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"测试图片不存在: {path}")

    boundary = f"----VisionOpsRobotGateway{int(time.time() * 1000)}"
    content = path.read_bytes()

    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")

    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + content + footer

    req = urllib.request.Request(
        infer_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text)

    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"推理服务 HTTP 错误: {e.code}, detail={detail}") from e

    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接推理服务: {infer_url}, error={e}") from e
