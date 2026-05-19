#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Modbus TCP 简单测试客户端 v1.2.1。

示例：
    python3 test_modbus_tcp_client.py --host 192.168.1.202 --port 1502 --unit-id 1 --address 0 --count 50
"""

import argparse
import sys

try:
    from pymodbus.client import ModbusTcpClient
except Exception:
    from pymodbus.client.sync import ModbusTcpClient


TASK_NAMES = {
    0: "unknown",
    1: "classification",
    2: "detection",
    3: "obb_detection",
    4: "segmentation",
    5: "roi_classification",
}

SCHEMA_NAMES = {
    0: "unknown",
    101: "classification_topk_v1",
    201: "detection_xyxy_v1",
    301: "obb_cxcywh_angle_points_v1",
    401: "segmentation_summary_v1",
    501: "roi_det_cls_v1",
}


def read_holding_registers_compat(client, address: int, count: int, unit_id: int):
    try:
        return client.read_holding_registers(address=address, count=count, slave=unit_id)
    except TypeError:
        return client.read_holding_registers(address=address, count=count, unit=unit_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.202")
    parser.add_argument("--port", type=int, default=1502)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--address", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    client = ModbusTcpClient(host=args.host, port=args.port, timeout=3)
    if not client.connect():
        print(f"[ERROR] connect failed: {args.host}:{args.port}")
        return 2

    try:
        rr = read_holding_registers_compat(client, args.address, args.count, args.unit_id)
        if rr.isError():
            print(f"[ERROR] modbus read error: {rr}")
            return 3

        regs = list(rr.registers)
        print(f"[OK] read {len(regs)} holding registers from {args.host}:{args.port}, unit_id={args.unit_id}, address={args.address}")
        for i, v in enumerate(regs):
            print(f"reg[{args.address + i:03d}] = {v}")

        if args.address == 0 and len(regs) >= 25:
            task_type = regs[5]
            schema = regs[6]
            print("\n[DECODE v2 COMMON]")
            print(f"magic               = {regs[0]} / 0x{regs[0]:04X}")
            print(f"protocol_version    = {regs[1]}")
            print(f"heartbeat           = {regs[2]}")
            print(f"service_status      = {regs[3]}")
            print(f"result_valid        = {regs[4]}")
            print(f"task_type           = {task_type} / {TASK_NAMES.get(task_type, 'unknown')}")
            print(f"result_schema       = {schema} / {SCHEMA_NAMES.get(schema, 'unknown')}")
            print(f"latency_ms          = {regs[12]}")
            print(f"ng_flag             = {regs[13]}")
            print(f"primary_class_id    = {regs[14]}")
            print(f"primary_conf        = {regs[15] / 10000.0:.4f}")
            print(f"result_count        = {regs[16]}")
            print(f"image_width         = {regs[17]}")
            print(f"image_height        = {regs[18]}")
            print(f"item_stride         = {regs[19]}")
            print(f"item_base           = {regs[20]}")
            print(f"max_items           = {regs[21]}")

        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
