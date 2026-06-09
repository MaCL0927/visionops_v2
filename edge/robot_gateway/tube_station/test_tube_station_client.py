#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 Tube Station Modbus TCP 服务。"""
from __future__ import annotations

import argparse
import sys
import time

try:
    from pymodbus.client import ModbusTcpClient
except Exception:
    from pymodbus.client.sync import ModbusTcpClient


def read_hr(client, address: int, count: int, unit_id: int):
    try:
        return client.read_holding_registers(address=address, count=count, slave=unit_id)
    except TypeError:
        return client.read_holding_registers(address=address, count=count, unit=unit_id)


def write_reg(client, address: int, value: int, unit_id: int):
    try:
        return client.write_register(address=address, value=value, slave=unit_id)
    except TypeError:
        return client.write_register(address=address, value=value, unit=unit_id)


def state_name(v: int) -> str:
    return {0: "unknown", 1: "stand", 2: "lying", 3: "missing/error"}.get(v, f"unknown({v})")


def status_name(v: int) -> str:
    return {0: "idle", 1: "busy", 2: "done", 3: "error"}.get(v, f"unknown({v})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1502)
    ap.add_argument("--unit-id", type=int, default=1)
    ap.add_argument("--address-base", type=int, default=0)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    client = ModbusTcpClient(host=args.host, port=args.port, timeout=3)
    if not client.connect():
        print(f"[ERROR] connect failed: {args.host}:{args.port}")
        return 2

    base = args.address_base
    seq = args.seq or int(time.time()) & 0xFFFF
    try:
        # reset then trigger
        write_reg(client, base + 0, 0, args.unit_id)
        time.sleep(0.1)
        write_reg(client, base + 1, seq, args.unit_id)
        write_reg(client, base + 0, 1, args.unit_id)
        print(f"[INFO] trigger sent: seq={seq}")

        deadline = time.time() + args.timeout
        regs = []
        while time.time() < deadline:
            rr = read_hr(client, base, 16, args.unit_id)
            if rr.isError():
                print(f"[ERROR] read error: {rr}")
                return 3
            regs = list(rr.registers)
            status = regs[2]
            print(f"[POLL] status={status}/{status_name(status)} result_seq={regs[3]} left={regs[4]} right={regs[5]} err={regs[6]} time={regs[7]}ms heartbeat={regs[8]}")
            if regs[3] == seq and status in (2, 3):
                break
            time.sleep(0.2)

        print("\n[RESULT]")
        print(f"status        = {regs[2]} / {status_name(regs[2])}")
        print(f"result_seq    = {regs[3]}")
        print(f"left_state    = {regs[4]} / {state_name(regs[4])}")
        print(f"right_state   = {regs[5]} / {state_name(regs[5])}")
        print(f"error_code    = {regs[6]}")
        print(f"process_ms    = {regs[7]}")
        print(f"det_count     = {regs[9]}")
        print(f"left_conf     = {regs[10] / 10000.0:.4f}")
        print(f"right_conf    = {regs[11] / 10000.0:.4f}")
        print(f"left_class_id = {regs[12] if regs[12] != 65535 else 'invalid'}")
        print(f"right_class_id= {regs[13] if regs[13] != 65535 else 'invalid'}")
        print(f"image_size    = {regs[14]}x{regs[15]}")

        # reset command
        write_reg(client, base + 0, 0, args.unit_id)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
