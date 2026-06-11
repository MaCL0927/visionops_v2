#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Carton Partition Cell Check Modbus TCP Service

PLC writes trigger registers -> HP60C RGB snapshot -> existing C++ YOLO infer ->
cell count + 5x8 template grid-pose check -> write OK/NG result.

This service is independent from tube_station and carton_tube_check.
Default port: 1504.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pymodbus.server import StartTcpServer
except Exception:  # pymodbus 2.x
    from pymodbus.server.sync import StartTcpServer

from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext

try:
    from pymodbus.device import ModbusDeviceIdentification
except Exception:
    ModbusDeviceIdentification = None

# Reuse the one-shot RGB + C++ infer + grid-template analysis.
import debug_partition_check_once as core

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ENV = THIS_DIR / "partition_check.env"

# -----------------------------
# Holding register map
# -----------------------------
REG_TRIGGER_CMD = 0          # PLC write: 1=trigger
REG_TRIGGER_SEQ = 1          # PLC write: increment for each trigger
REG_STATUS = 2               # 0=idle, 1=busy, 2=done, 3=error
REG_RESULT_SEQ = 3           # echo trigger seq when result is ready
REG_FINAL_RESULT = 4         # 0=unknown, 1=OK, 2=NG, 3=ERROR
REG_NG_REASON = 5            # see REASON_CODE
REG_ERROR_CODE = 6           # see ERR_*
REG_PROCESS_TIME_MS = 7
REG_HEARTBEAT = 8
REG_CELL_COUNT = 9
REG_EXPECTED_COUNT = 10
REG_MATCHED_COUNT = 11
REG_MISSING_COUNT = 12
REG_MEAN_CENTER_ERR_X10 = 13
REG_P95_CENTER_ERR_X10 = 14
REG_GRID_CENTER_OFFSET_X10 = 15
REG_ROW_ANGLE_DIFF_X100 = 16
REG_COL_ANGLE_DIFF_X100 = 17
REG_AFFINE_ROT_X100 = 18
REG_AFFINE_SHEAR_X10000 = 19
REG_BAD_SIZE_COUNT = 20
REG_IMAGE_WIDTH = 21
REG_IMAGE_HEIGHT = 22
REG_TEMPLATE_LOADED = 23
REG_GRID_ASSIGN_OK = 24
REG_ROWS = 25
REG_COLS = 26
REG_RAW_PRED_COUNT = 27
REG_MAX_CENTER_ERR_X10 = 28
REG_EDGE_CELL_MAX_ERR_X10 = 29
REG_ROW_ANGLE_MAX_DIFF_X100 = 30
REG_ROW_ANGLE_STD_DIFF_X100 = 31
REG_RESERVED_32 = 32
REG_RESERVED_33 = 33
REG_RESERVED_34 = 34
REG_RESERVED_35 = 35
REG_RESERVED_36 = 36
REG_RESERVED_37 = 37
REG_RESERVED_38 = 38
REG_RESERVED_39 = 39

# Per-slot output, 5x8 by default.
# status: 0=ok, 1=missing, 2=size_bad, 3=other, 65535=not_available
REG_SLOT_STATUS_BASE = 40
REG_SLOT_CENTER_ERR_X10_BASE = 90

STATUS_IDLE = 0
STATUS_BUSY = 1
STATUS_DONE = 2
STATUS_ERROR = 3

FINAL_UNKNOWN = 0
FINAL_OK = 1
FINAL_NG = 2
FINAL_ERROR = 3

REASON_CODE = {
    "NONE": 0,
    "COUNT_MISMATCH": 1,
    "GRID_ASSIGN_FAILED": 2,
    "TEMPLATE_MISSING": 3,
    "SLOT_MISSING": 4,
    "MEAN_CENTER_ERROR": 5,
    "P95_CENTER_ERROR": 6,
    "GRID_CENTER_OFFSET": 7,
    "ROW_ANGLE_DIFF": 8,
    "COL_ANGLE_DIFF": 9,
    "AFFINE_ROTATION": 10,
    "AFFINE_SHEAR": 11,
    "BOX_SIZE_ANOMALY": 12,
    "MAX_CENTER_ERROR": 13,
    "ROW_ANGLE_MAX_DIFF": 14,
    "ROW_ANGLE_STD_DIFF": 15,
    "EDGE_CELL_ERROR": 16,
    "CALIBRATED": 90,
    "INTERNAL_ERROR": 99,
}

ERR_NONE = 0
ERR_SNAPSHOT_FAILED = 201
ERR_INFER_FAILED = 202
ERR_ANALYZE_FAILED = 203
ERR_JSON_FAILED = 204
ERR_INTERNAL = 301
ERR_TEMPLATE_MISSING = 401

INVALID_U16 = 65535
INVALID_I16 = 32767

# Load service env.  core has also loaded it at import time; this keeps service-only vars available.
core.load_env_file(Path(os.environ.get("VISIONOPS_PARTITION_ENV", str(DEFAULT_ENV))))

ENABLE = core.getenv_int("VISIONOPS_PARTITION_ENABLE", 1)
HOST = core.getenv_str("VISIONOPS_PARTITION_MODBUS_HOST", "0.0.0.0")
PORT = core.getenv_int("VISIONOPS_PARTITION_MODBUS_PORT", 1504)
UNIT_ID = core.getenv_int("VISIONOPS_PARTITION_MODBUS_UNIT_ID", 1)
ADDRESS_BASE = core.getenv_int("VISIONOPS_PARTITION_ADDRESS_BASE", 0)
REGISTER_COUNT = core.getenv_int("VISIONOPS_PARTITION_REGISTER_COUNT", 128)
POLL_INTERVAL_S = max(0.02, core.getenv_int("VISIONOPS_PARTITION_POLL_INTERVAL_MS", 50) / 1000.0)
LOG_LEVEL = core.getenv_str("VISIONOPS_PARTITION_LOG_LEVEL", "INFO").upper()
SAVE_RESULT_DIR = core.getenv_str("VISIONOPS_PARTITION_SAVE_RESULT_DIR", "/tmp/carton_partition_check_latest")
SAVE_EVERY_TRIGGER = core.getenv_int("VISIONOPS_PARTITION_SAVE_EVERY_TRIGGER", 1)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def u16(v: int) -> int:
    return int(v) & 0xFFFF


def encode_x10(v: Any, missing: int = INVALID_U16) -> int:
    if v is None:
        return missing
    try:
        iv = int(round(float(v) * 10.0))
    except Exception:
        return missing
    if iv < 0:
        iv = 0
    if iv > 65534:
        iv = 65534
    return iv


def encode_x100_signed(v: Any, missing: int = INVALID_I16) -> int:
    if v is None:
        return missing
    try:
        iv = int(round(float(v) * 100.0))
    except Exception:
        return missing
    if iv > 32766:
        iv = 32766
    if iv < -32768:
        iv = -32768
    return iv & 0xFFFF


def encode_x10000(v: Any, missing: int = INVALID_U16) -> int:
    if v is None:
        return missing
    try:
        iv = int(round(float(v) * 10000.0))
    except Exception:
        return missing
    if iv < 0:
        iv = 0
    if iv > 65534:
        iv = 65534
    return iv


def final_result_code(text: str) -> int:
    text = (text or "").upper()
    if text == "OK":
        return FINAL_OK
    if text == "NG":
        return FINAL_NG
    if text == "ERROR":
        return FINAL_ERROR
    return FINAL_UNKNOWN


def reason_code(text: str) -> int:
    return REASON_CODE.get((text or "").upper(), REASON_CODE["INTERNAL_ERROR"])


def create_context() -> ModbusServerContext:
    n = max(1, core.EXPECTED_ROWS * core.EXPECTED_COLS)
    min_count = REG_SLOT_CENTER_ERR_X10_BASE + n + 16
    block_count = max(REGISTER_COUNT, ADDRESS_BASE + min_count + 10)
    hr_block = ModbusSequentialDataBlock(0, [0] * block_count)
    try:
        slave = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * block_count),
            co=ModbusSequentialDataBlock(0, [0] * block_count),
            hr=hr_block,
            ir=ModbusSequentialDataBlock(0, [0] * block_count),
            zero_mode=True,
        )
    except TypeError:
        slave = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * block_count),
            co=ModbusSequentialDataBlock(0, [0] * block_count),
            hr=hr_block,
            ir=ModbusSequentialDataBlock(0, [0] * block_count),
        )
    return ModbusServerContext(slaves={UNIT_ID: slave}, single=False)


def get_reg(context: ModbusServerContext, reg: int) -> int:
    try:
        values = context[UNIT_ID].getValues(3, ADDRESS_BASE + reg, count=1)
        return int(values[0]) if values else 0
    except TypeError:
        values = context[UNIT_ID].getValues(3, ADDRESS_BASE + reg, 1)
        return int(values[0]) if values else 0


def set_regs(context: ModbusServerContext, offset: int, values: List[int]) -> None:
    context[UNIT_ID].setValues(3, ADDRESS_BASE + offset, [u16(v) for v in values])


def set_reg(context: ModbusServerContext, reg: int, value: int) -> None:
    set_regs(context, reg, [value])


def slot_status_code(status: str) -> int:
    return {"ok": 0, "missing": 1, "size_bad": 2}.get(str(status or ""), 3)


def write_slot_registers(context: ModbusServerContext, result: Dict[str, Any]) -> None:
    rows = int(result.get("expected_rows") or core.EXPECTED_ROWS or 5)
    cols = int(result.get("expected_cols") or core.EXPECTED_COLS or 8)
    n = rows * cols
    status_values = [INVALID_U16] * n
    err_values = [INVALID_U16] * n
    slots = result.get("slots") if isinstance(result.get("slots"), list) else []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        try:
            sid = int(slot.get("slot_id"))
        except Exception:
            continue
        if 0 <= sid < n:
            status_values[sid] = slot_status_code(str(slot.get("status", "")))
            err_values[sid] = encode_x10(slot.get("center_error_px"), INVALID_U16)
    set_regs(context, REG_SLOT_STATUS_BASE, status_values)
    set_regs(context, REG_SLOT_CENTER_ERR_X10_BASE, err_values)


def save_debug_files(seq: int, rgb_bytes: bytes, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    if SAVE_EVERY_TRIGGER != 1:
        return
    try:
        out_dir = Path(SAVE_RESULT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "rgb.jpg").write_bytes(rgb_bytes)
        (out_dir / "infer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "last_seq.txt").write_text(str(seq), encoding="utf-8")
        try:
            core.draw_overlay(rgb_bytes, result, out_dir / "overlay.jpg")
        except Exception:
            logging.exception("failed to save overlay")
    except Exception:
        logging.exception("failed to save debug files")


def run_check_once() -> Dict[str, Any]:
    logging.info("fetch RGB snapshot: %s", core.SNAPSHOT_URL)
    rgb_bytes = core.http_get_bytes(core.SNAPSHOT_URL, core.HTTP_TIMEOUT_S)
    if not rgb_bytes:
        raise RuntimeError("snapshot is empty")

    logging.info("post C++ infer: %s bytes=%d", core.INFER_URL, len(rgb_bytes))
    payload = core.post_multipart_image(core.INFER_URL, rgb_bytes, timeout_s=core.HTTP_TIMEOUT_S)

    result = core.analyze(payload)
    result["_rgb_bytes"] = rgb_bytes
    result["_infer_payload"] = payload
    return result


class PartitionCheckService:
    def __init__(self, context: ModbusServerContext) -> None:
        self.context = context
        self.lock = threading.Lock()
        self.busy = False
        self.last_seq = -1
        self.heartbeat = 0

    def init_registers(self) -> None:
        values = [0] * REGISTER_COUNT
        values[REG_STATUS] = STATUS_IDLE
        values[REG_FINAL_RESULT] = FINAL_UNKNOWN
        values[REG_NG_REASON] = REASON_CODE["NONE"]
        values[REG_ERROR_CODE] = ERR_NONE
        values[REG_EXPECTED_COUNT] = core.EXPECTED_COUNT
        values[REG_ROWS] = core.EXPECTED_ROWS
        values[REG_COLS] = core.EXPECTED_COLS
        set_regs(self.context, 0, values)
        n = max(1, core.EXPECTED_ROWS * core.EXPECTED_COLS)
        set_regs(self.context, REG_SLOT_STATUS_BASE, [INVALID_U16] * n)
        set_regs(self.context, REG_SLOT_CENTER_ERR_X10_BASE, [INVALID_U16] * n)

    def heartbeat_loop(self) -> None:
        while True:
            self.heartbeat = (self.heartbeat + 1) & 0xFFFF
            set_reg(self.context, REG_HEARTBEAT, self.heartbeat)
            time.sleep(0.5)

    def poll_loop(self) -> None:
        logging.info("Carton partition check trigger poll started")
        while True:
            try:
                cmd = get_reg(self.context, REG_TRIGGER_CMD)
                seq = get_reg(self.context, REG_TRIGGER_SEQ)
                if cmd == 1 and seq != self.last_seq and not self.busy:
                    self.last_seq = seq
                    logging.info("trigger accepted: seq=%d", seq)
                    t = threading.Thread(target=self.handle_trigger, args=(seq,), daemon=True)
                    t.start()
            except Exception:
                logging.exception("trigger poll error")
            time.sleep(POLL_INTERVAL_S)

    def handle_trigger(self, seq: int) -> None:
        with self.lock:
            if self.busy:
                return
            self.busy = True

        start = time.time()
        set_regs(
            self.context,
            REG_STATUS,
            [STATUS_BUSY, seq, FINAL_UNKNOWN, REASON_CODE["NONE"], ERR_NONE, 0, self.heartbeat],
        )

        try:
            result = run_check_once()
            elapsed_ms = int(round((time.time() - start) * 1000))
            final_code = final_result_code(str(result.get("final_result", "")))
            reason = str(result.get("reason", "NONE"))
            status = STATUS_DONE if final_code in (FINAL_OK, FINAL_NG) else STATUS_ERROR
            err = int(result.get("error_code") or 0)
            if err == 0 and status == STATUS_ERROR:
                err = ERR_TEMPLATE_MISSING if reason == "TEMPLATE_MISSING" else ERR_ANALYZE_FAILED

            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            affine = metrics.get("affine") if isinstance(metrics.get("affine"), dict) else {}
            summary_values = [
                status,
                seq,
                final_code,
                reason_code(reason),
                err,
                elapsed_ms,
                self.heartbeat,
                int(result.get("valid_cell_count") or 0),
                int(result.get("expected_count") or core.EXPECTED_COUNT),
                int(metrics.get("matched_count") or 0),
                len(metrics.get("missing_slots") or []),
                encode_x10(metrics.get("mean_center_error_px")),
                encode_x10(metrics.get("p95_center_error_px")),
                encode_x10(metrics.get("grid_center_offset_px")),
                encode_x100_signed(metrics.get("row_angle_diff_deg")),
                encode_x100_signed(metrics.get("col_angle_diff_deg")),
                encode_x100_signed(affine.get("rotation_deg")),
                encode_x10000(affine.get("shear")),
                int(metrics.get("bad_size_count") or 0),
                int(result.get("image_width") or 0),
                int(result.get("image_height") or 0),
                1 if result.get("template_loaded") else 0,
                1 if result.get("grid_assign_ok") else 0,
                int(result.get("expected_rows") or core.EXPECTED_ROWS),
                int(result.get("expected_cols") or core.EXPECTED_COLS),
                int(result.get("raw_prediction_count") or 0),
                encode_x10(metrics.get("max_center_error_px")),
                encode_x10(metrics.get("edge_cell_max_error_px")),
                encode_x100_signed(metrics.get("max_row_angle_diff_deg")),
                encode_x100_signed(metrics.get("row_angle_std_diff_deg")),
            ]
            set_regs(self.context, REG_STATUS, summary_values)
            write_slot_registers(self.context, result)

            rgb_bytes = result.pop("_rgb_bytes", b"")
            payload = result.pop("_infer_payload", {})
            save_debug_files(seq, rgb_bytes, payload, result)

            logging.info(
                "trigger done: seq=%d status=%d final=%s reason=%s cells=%s/%s mean=%s p95=%s max=%s edge=%s row_mean=%s row_max=%s row_std_diff=%s col_diff=%s time=%dms",
                seq,
                status,
                result.get("final_result"),
                reason,
                result.get("valid_cell_count"),
                result.get("expected_count"),
                metrics.get("mean_center_error_px"),
                metrics.get("p95_center_error_px"),
                metrics.get("max_center_error_px"),
                metrics.get("edge_cell_max_error_px"),
                metrics.get("row_angle_diff_deg"),
                metrics.get("max_row_angle_diff_deg"),
                metrics.get("row_angle_std_diff_deg"),
                metrics.get("col_angle_diff_deg"),
                elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = int(round((time.time() - start) * 1000))
            msg = str(exc).lower()
            if "snapshot" in msg:
                err = ERR_SNAPSHOT_FAILED
            elif "infer" in msg or "post" in msg:
                err = ERR_INFER_FAILED
            elif "json" in msg:
                err = ERR_JSON_FAILED
            else:
                err = ERR_INTERNAL
            logging.error("trigger failed: seq=%d err=%d %s", seq, err, exc)
            logging.debug("%s", traceback.format_exc())
            set_regs(
                self.context,
                REG_STATUS,
                [
                    STATUS_ERROR,
                    seq,
                    FINAL_ERROR,
                    REASON_CODE["INTERNAL_ERROR"],
                    err,
                    elapsed_ms,
                    self.heartbeat,
                ],
            )
        finally:
            with self.lock:
                self.busy = False


def main() -> int:
    if ENABLE != 1:
        logging.warning("VISIONOPS_PARTITION_ENABLE != 1, exit")
        return 0

    context = create_context()
    service = PartitionCheckService(context)
    service.init_registers()

    threading.Thread(target=service.heartbeat_loop, daemon=True).start()
    threading.Thread(target=service.poll_loop, daemon=True).start()

    identity = None
    if ModbusDeviceIdentification is not None:
        identity = ModbusDeviceIdentification()
        identity.VendorName = "VisionOps"
        identity.ProductCode = "PARTITION_CELL"
        identity.ProductName = "VisionOps Carton Partition Cell Check Modbus TCP"
        identity.ModelName = "LB3576 HP60C Carton Partition Check"
        identity.MajorMinorRevision = "1.0"

    logging.info(
        "starting carton partition check: host=%s port=%d unit_id=%d address_base=%d rows=%d cols=%d expected=%d template=%s",
        HOST,
        PORT,
        UNIT_ID,
        ADDRESS_BASE,
        core.EXPECTED_ROWS,
        core.EXPECTED_COLS,
        core.EXPECTED_COUNT,
        core.TEMPLATE_PATH,
    )
    logging.info("snapshot_url=%s", core.SNAPSHOT_URL)
    logging.info("infer_url=%s", core.INFER_URL)
    StartTcpServer(context=context, identity=identity, address=(HOST, PORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
