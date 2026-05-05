#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot Gateway 测试客户端。

运行：
    python edge/robot_gateway/test_client.py --host 127.0.0.1 --port 9100
"""

from __future__ import annotations

import argparse
import json
import socket


def build_test_request() -> bytes:
    req = {
        "function": "camera",
        "timestamp": [1000, 2000],
        "triggerpos": 12345,
        "triggerindex": 1,
        "camera": "1,2,3",
        "left_arm": {
            "x": 100.01,
            "y": 200.02,
            "z": 300.03,
            "ox": 100.01,
            "oy": 200.02,
            "oz": 300.03,
            "ow": 30.03,
        },
        "right_arm": {
            "x": 100.01,
            "y": 200.02,
            "z": 300.03,
            "ox": 100.01,
            "oy": 200.02,
            "oz": 300.03,
            "ow": 30.03,
        },
    }

    payload = json.dumps(req, ensure_ascii=False, separators=(",", ":"))
    return b"*" + payload.encode("utf-8") + b"#"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()

    data = build_test_request()

    with socket.create_connection((args.host, args.port), timeout=5.0) as sock:
        print("[SEND]")
        print(data.decode("utf-8"))

        sock.sendall(data)

        resp = sock.recv(8192)
        print("\n[RECV]")
        print(resp.decode("utf-8"))


if __name__ == "__main__":
    main()
