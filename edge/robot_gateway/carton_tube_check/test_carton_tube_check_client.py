#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 VisionOps Carton Tube Check Modbus TCP 服务。"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List

try:
    from pymodbus.client import ModbusTcpClient
except Exception:
    from pymodbus.client.sync import ModbusTcpClient


def read_hr(client, address: int, count: int, unit_id: int):
    """Read holding registers with compatibility across pymodbus 2.x/3.x.

    pymodbus versions have used different keyword names for the Modbus unit id:
    unit=..., slave=..., and device_id=.... Some new versions also work without
    passing the id explicitly for TCP. Try the known forms in a safe order.
    """
    # PyModbus 3.x uses slave= for Modbus device id over TCP.
    # Some newer development versions accept device_id=, but trying it first can
    # be silently ignored by kwargs in some versions and result in unit id 0.
    # Therefore use slave -> unit -> device_id -> no-id.
    attempts = [
        lambda: client.read_holding_registers(address=address, count=count, slave=unit_id),
        lambda: client.read_holding_registers(address=address, count=count, unit=unit_id),
        lambda: client.read_holding_registers(address=address, count=count, device_id=unit_id),
        lambda: client.read_holding_registers(address=address, count=count),
    ]
    last_exc = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as exc:
            last_exc = exc
    raise last_exc


def write_reg(client, address: int, value: int, unit_id: int):
    """Write one holding register with compatibility across pymodbus 2.x/3.x."""
    # PyModbus 3.x uses slave= for Modbus device id over TCP.
    attempts = [
        lambda: client.write_register(address=address, value=value, slave=unit_id),
        lambda: client.write_register(address=address, value=value, unit=unit_id),
        lambda: client.write_register(address=address, value=value, device_id=unit_id),
        lambda: client.write_register(address=address, value=value),
    ]
    last_exc = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as exc:
            last_exc = exc
    raise last_exc


def status_name(v: int) -> str:
    return {0: "idle", 1: "busy", 2: "done", 3: "error"}.get(v, f"unknown({v})")


def final_name(v: int) -> str:
    return {0: "unknown", 1: "OK", 2: "NG", 3: "ERROR"}.get(v, f"unknown({v})")


def reason_name(v: int) -> str:
    return {
        0: "NONE",
        1: "LYING_DETECTED",
        2: "STAND_COUNT_LOW",
        3: "DEPTH_INVALID",
        4: "HEIGHT_HIGH",
        9: "INTERNAL_ERROR",
    }.get(v, f"unknown({v})")


def i16(v: int) -> int:
    v = int(v) & 0xFFFF
    return v - 65536 if v >= 32768 else v


def print_matrix(title: str, values: List[int], rows: int, cols: int, kind: str) -> None:
    print(f"\n[MATRIX] {title}")
    for r in range(rows):
        cells = []
        for c in range(cols):
            idx = r * cols + c
            v = values[idx] if idx < len(values) else 65535
            if kind == "u16_missing":
                s = "----" if v == 65535 else str(v)
            elif kind == "i16_missing":
                s = "----" if v == 32767 else str(i16(v))
            elif kind == "high":
                if v == 65535:
                    s = "----"
                elif v == 1:
                    s = "HIGH"
                else:
                    s = "ok"
            else:
                s = str(v)
            cells.append(s.rjust(7))
        print(f"row{r:02d}: " + " ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1503)
    ap.add_argument("--unit-id", type=int, default=1)
    ap.add_argument("--address-base", type=int, default=0)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=12.0)
    ap.add_argument("--print-matrix", action="store_true")
    args = ap.parse_args()

    client = ModbusTcpClient(host=args.host, port=args.port, timeout=3)
    if not client.connect():
        print(f"[ERROR] connect failed: {args.host}:{args.port}")
        return 2

    base = args.address_base
    seq = args.seq or int(time.time()) & 0xFFFF
    try:
        write_reg(client, base + 0, 0, args.unit_id)
        time.sleep(0.1)
        write_reg(client, base + 1, seq, args.unit_id)
        write_reg(client, base + 0, 1, args.unit_id)
        print(f"[INFO] trigger sent: seq={seq}")

        deadline = time.time() + args.timeout
        regs: List[int] = []
        while time.time() < deadline:
            rr = read_hr(client, base, 26, args.unit_id)
            if rr.isError():
                print(f"[ERROR] read error: {rr}")
                return 3
            regs = list(rr.registers)
            status = regs[2]
            print(
                f"[POLL] status={status}/{status_name(status)} result_seq={regs[3]} "
                f"final={regs[4]}/{final_name(regs[4])} reason={regs[5]}/{reason_name(regs[5])} "
                f"err={regs[6]} time={regs[7]}ms heartbeat={regs[8]}"
            )
            if regs[3] == seq and status in (2, 3):
                break
            time.sleep(0.2)

        if not regs:
            print("[ERROR] no registers read")
            return 4

        print("\n[RESULT]")
        print(f"status              = {regs[2]} / {status_name(regs[2])}")
        print(f"result_seq          = {regs[3]}")
        print(f"final_result        = {regs[4]} / {final_name(regs[4])}")
        print(f"ng_reason           = {regs[5]} / {reason_name(regs[5])}")
        print(f"error_code          = {regs[6]}")
        print(f"process_ms          = {regs[7]}")
        print(f"stand_count         = {regs[9]}")
        print(f"lying_count         = {regs[10]}")
        print(f"high_count          = {regs[11]}")
        print(f"max_height_diff_mm  = {regs[12]}")
        print(f"valid_pred_count    = {regs[13]}")
        print(f"raw_pred_count      = {regs[14]}")
        print(f"image_size          = {regs[15]}x{regs[16]}")
        print(f"depth_size          = {regs[17]}x{regs[18]}")
        print(f"baseline_mode_code  = {regs[19]}")
        print(f"grid                = {regs[20]}x{regs[21]}")
        print(f"detected_slots      = {regs[22]}")
        print(f"missing_slots       = {regs[23]}")

        rows = regs[20] or 5
        cols = regs[21] or 8
        n = rows * cols
        if args.print_matrix:
            rr_depth = read_hr(client, base + 30, n, args.unit_id)
            rr_diff = read_hr(client, base + 70, n, args.unit_id)
            rr_high = read_hr(client, base + 110, n, args.unit_id)
            rr_base = read_hr(client, base + 150, n, args.unit_id)
            if not rr_depth.isError():
                print_matrix("depth_mm", list(rr_depth.registers), rows, cols, "u16_missing")
            if not rr_base.isError():
                print_matrix("baseline_depth_mm", list(rr_base.registers), rows, cols, "u16_missing")
            if not rr_diff.isError():
                print_matrix("height_diff_mm = baseline - current_depth", list(rr_diff.registers), rows, cols, "i16_missing")
            if not rr_high.isError():
                print_matrix("height_high", list(rr_high.registers), rows, cols, "high")

        write_reg(client, base + 0, 0, args.unit_id)
        return 0 if regs[4] == 1 else 2
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
