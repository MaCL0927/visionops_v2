#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test client for VisionOps unified Robot/PLC Modbus-TCP protocol."""
from __future__ import annotations

import argparse
import sys
import time
from typing import List

try:
    from pymodbus.client import ModbusTcpClient
except Exception:
    from pymodbus.client.sync import ModbusTcpClient

REG_RESULT = {
    "partition": 1,
    "tube": 2,
    "coord": 3,
}
REG_TRIGGER = {
    "partition": 101,
    "tube": 102,
    "coord": 103,
}
REG_COORD_BASE = 20


def _is_no_response(resp) -> bool:
    try:
        if not resp.isError():
            return False
    except Exception:
        return False
    msg = str(resp)
    return ("No Response" in msg) or ("Unable to decode" in msg)


def read_hr(client, address: int, count: int, unit_id: int):
    attempts = [
        lambda: client.read_holding_registers(address=address, count=count, device_id=unit_id),
        lambda: client.read_holding_registers(address=address, count=count, slave=unit_id),
        lambda: client.read_holding_registers(address=address, count=count, unit=unit_id),
        lambda: client.read_holding_registers(address=address, count=count),
    ]
    last_exc = None
    last_no_response = None
    for fn in attempts:
        try:
            resp = fn()
            if _is_no_response(resp):
                last_no_response = resp
                continue
            return resp
        except TypeError as exc:
            last_exc = exc
    if last_no_response is not None:
        return last_no_response
    raise last_exc


def write_reg(client, address: int, value: int, unit_id: int):
    attempts = [
        lambda: client.write_register(address=address, value=value, device_id=unit_id),
        lambda: client.write_register(address=address, value=value, slave=unit_id),
        lambda: client.write_register(address=address, value=value, unit=unit_id),
        lambda: client.write_register(address=address, value=value),
    ]
    last_exc = None
    last_no_response = None
    for fn in attempts:
        try:
            resp = fn()
            if _is_no_response(resp):
                last_no_response = resp
                continue
            return resp
        except TypeError as exc:
            last_exc = exc
    if last_no_response is not None:
        return last_no_response
    raise last_exc


def result_name(v: int) -> str:
    return {0: "NONE", 1: "OK", 2: "NG/ERROR"}.get(v, f"unknown({v})")


def print_coords(coords: List[int]) -> None:
    print("\n[COORDS] offset 20~99")
    for i in range(40):
        x = coords[2 * i] if 2 * i < len(coords) else 0
        y = coords[2 * i + 1] if 2 * i + 1 < len(coords) else 0
        print(f"slot{i+1:02d}: x={x:4d}, y={y:4d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5045)
    ap.add_argument("--unit-id", type=int, default=1)
    ap.add_argument("--address-base", type=int, default=0)
    ap.add_argument("--task", choices=["partition", "tube", "coord"], required=True)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--print-coords", action="store_true")
    args = ap.parse_args()

    client = ModbusTcpClient(host=args.host, port=args.port, timeout=3)
    if not client.connect():
        print(f"[ERROR] connect failed: {args.host}:{args.port}")
        return 2

    base = args.address_base
    result_reg = REG_RESULT[args.task]
    trigger_reg = REG_TRIGGER[args.task]

    try:
        # Clear trigger first, then set it to 1, same as robot behavior.
        for addr, val, name in [
            (base + trigger_reg, 0, "trigger=0"),
            (base + trigger_reg, 1, "trigger=1"),
        ]:
            wr = write_reg(client, addr, val, args.unit_id)
            if hasattr(wr, "isError") and wr.isError():
                print(f"[ERROR] write {name} at {addr} failed: {wr}")
                return 3
            time.sleep(0.1)

        print(f"[INFO] task={args.task} trigger register {trigger_reg} set to 1")

        deadline = time.time() + args.timeout
        result = 0
        while time.time() < deadline:
            rr = read_hr(client, base, 104, args.unit_id)
            if rr.isError():
                print(f"[ERROR] read error: {rr}")
                return 3
            regs = list(rr.registers)
            heartbeat = regs[0]
            result = regs[result_reg]
            trigger_val = regs[trigger_reg]
            print(f"[POLL] heartbeat={heartbeat} trigger[{trigger_reg}]={trigger_val} result[{result_reg}]={result}/{result_name(result)}")
            if result in (1, 2):
                break
            time.sleep(0.2)

        if result not in (1, 2):
            print("[ERROR] timeout waiting result")
            return 4

        print(f"\n[RESULT] task={args.task} result_reg={result_reg} value={result}/{result_name(result)}")

        if args.task == "coord" or args.print_coords:
            rr = read_hr(client, base + REG_COORD_BASE, 80, args.unit_id)
            if rr.isError():
                print(f"[ERROR] read coords failed: {rr}")
            else:
                print_coords(list(rr.registers))

        # Simulate robot reset trigger to 0. Service will clear result register; coordinates remain unchanged.
        write_reg(client, base + trigger_reg, 0, args.unit_id)
        print(f"[INFO] trigger register {trigger_reg} reset to 0")
        return 0 if result == 1 else 2
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
