#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模拟上位机 TCP 客户端。

运行：
    python edge/robot_gateway/active_push_client.py --host 127.0.0.1 --port 9100
"""

from __future__ import annotations

import argparse
import json
import socket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args()

    buffer = b""

    with socket.create_connection((args.host, args.port), timeout=5.0) as sock:
        print(f"[CONNECTED] {args.host}:{args.port}")

        while True:
            data = sock.recv(4096)
            if not data:
                print("[CLOSED]")
                break

            buffer += data

            while True:
                start = buffer.find(b"*")
                if start < 0:
                    buffer = b""
                    break

                end = buffer.find(b"#", start + 1)
                if end < 0:
                    buffer = buffer[start:]
                    break

                payload = buffer[start + 1:end]
                buffer = buffer[end + 1:]

                text = payload.decode("utf-8", errors="replace")
                print("\n[RECV RAW]")
                print(text)

                try:
                    obj = json.loads(text)
                    print("[RECV JSON]")
                    print(json.dumps(obj, ensure_ascii=False, indent=2))
                except Exception:
                    pass


if __name__ == "__main__":
    main()
