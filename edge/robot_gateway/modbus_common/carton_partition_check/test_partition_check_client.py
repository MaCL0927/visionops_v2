#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 VisionOps Carton Partition Cell Check Modbus TCP 服务。"""
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
    attempts = [
        lambda: client.read_holding_registers(address=address, count=count, device_id=unit_id),
        lambda: client.read_holding_registers(address=address, count=count, slave=unit_id),
        lambda: client.read_holding_registers(address=address, count=count, unit=unit_id),
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
    attempts = [
        lambda: client.write_register(address=address, value=value, device_id=unit_id),
        lambda: client.write_register(address=address, value=value, slave=unit_id),
        lambda: client.write_register(address=address, value=value, unit=unit_id),
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
        1: "COUNT_MISMATCH",
        2: "GRID_ASSIGN_FAILED",
        3: "TEMPLATE_MISSING",
        4: "SLOT_MISSING",
        5: "MEAN_CENTER_ERROR",
        6: "P95_CENTER_ERROR",
        7: "GRID_CENTER_OFFSET",
        8: "ROW_ANGLE_DIFF",
        9: "COL_ANGLE_DIFF",
        10: "AFFINE_ROTATION",
        11: "AFFINE_SHEAR",
        12: "BOX_SIZE_ANOMALY",
        13: "MAX_CENTER_ERROR",
        14: "ROW_ANGLE_MAX_DIFF",
        15: "ROW_ANGLE_STD_DIFF",
        16: "EDGE_CELL_ERROR",
        90: "CALIBRATED",
        99: "INTERNAL_ERROR",
    }.get(v, f"unknown({v})")


def i16(v: int) -> int:
    v = int(v) & 0xFFFF
    return v - 65536 if v >= 32768 else v


def slot_status_name(v: int) -> str:
    return {0: "ok", 1: "missing", 2: "size_bad", 3: "other", 65535: "----"}.get(v, str(v))


def print_matrix(title: str, values: List[int], rows: int, cols: int, kind: str) -> None:
    print(f"\n[MATRIX] {title}")
    for r in range(rows):
        cells = []
        for c in range(cols):
            idx = r * cols + c
            v = values[idx] if idx < len(values) else 65535
            if kind == "status":
                s = slot_status_name(v)
            elif kind == "x10":
                s = "----" if v == 65535 else f"{v/10.0:.1f}"
            else:
                s = str(v)
            cells.append(s.rjust(9))
        print(f"row{r:02d}: " + " ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1504)
    ap.add_argument("--unit-id", type=int, default=1)
    ap.add_argument("--address-base", type=int, default=0)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=12.0)
    ap.add_argument("--print-slots", action="store_true")
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
            rr = read_hr(client, base, 40, args.unit_id)
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
        print(f"status                 = {regs[2]} / {status_name(regs[2])}")
        print(f"result_seq             = {regs[3]}")
        print(f"final_result           = {regs[4]} / {final_name(regs[4])}")
        print(f"ng_reason              = {regs[5]} / {reason_name(regs[5])}")
        print(f"error_code             = {regs[6]}")
        print(f"process_ms             = {regs[7]}")
        print(f"cell_count             = {regs[9]}")
        print(f"expected_count         = {regs[10]}")
        print(f"matched_count          = {regs[11]}")
        print(f"missing_count          = {regs[12]}")
        print(f"mean_center_err_px     = {regs[13] / 10.0:.1f}" if regs[13] != 65535 else "mean_center_err_px     = ----")
        print(f"p95_center_err_px      = {regs[14] / 10.0:.1f}" if regs[14] != 65535 else "p95_center_err_px      = ----")
        print(f"grid_center_offset_px  = {regs[15] / 10.0:.1f}" if regs[15] != 65535 else "grid_center_offset_px  = ----")
        print(f"row_angle_diff_deg     = {i16(regs[16]) / 100.0:.2f}" if regs[16] != 32767 else "row_angle_diff_deg     = ----")
        print(f"col_angle_diff_deg     = {i16(regs[17]) / 100.0:.2f}" if regs[17] != 32767 else "col_angle_diff_deg     = ----")
        print(f"affine_rot_deg         = {i16(regs[18]) / 100.0:.2f}" if regs[18] != 32767 else "affine_rot_deg         = ----")
        print(f"affine_shear           = {regs[19] / 10000.0:.4f}" if regs[19] != 65535 else "affine_shear           = ----")
        print(f"bad_size_count         = {regs[20]}")
        print(f"image_size             = {regs[21]}x{regs[22]}")
        print(f"template_loaded        = {regs[23]}")
        print(f"grid_assign_ok         = {regs[24]}")
        print(f"grid                   = {regs[25]}x{regs[26]}")
        print(f"raw_pred_count         = {regs[27]}")
        print(f"max_center_err_px      = {regs[28] / 10.0:.1f}" if regs[28] != 65535 else "max_center_err_px      = ----")
        print(f"edge_cell_max_err_px   = {regs[29] / 10.0:.1f}" if regs[29] != 65535 else "edge_cell_max_err_px   = ----")
        print(f"max_row_angle_diff_deg = {i16(regs[30]) / 100.0:.2f}" if regs[30] != 32767 else "max_row_angle_diff_deg = ----")
        print(f"row_angle_std_diff_deg = {i16(regs[31]) / 100.0:.2f}" if regs[31] != 32767 else "row_angle_std_diff_deg = ----")

        rows = regs[25] or 5
        cols = regs[26] or 8
        n = rows * cols
        if args.print_slots:
            rr_status = read_hr(client, base + 40, n, args.unit_id)
            rr_err = read_hr(client, base + 90, n, args.unit_id)
            if not rr_status.isError():
                print_matrix("slot_status", list(rr_status.registers), rows, cols, "status")
            if not rr_err.isError():
                print_matrix("center_error_px", list(rr_err.registers), rows, cols, "x10")

        write_reg(client, base + 0, 0, args.unit_id)
        return 0 if regs[4] == 1 else 2
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
