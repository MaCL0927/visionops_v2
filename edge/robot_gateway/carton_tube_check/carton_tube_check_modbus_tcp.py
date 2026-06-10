#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Carton Tube Check Modbus TCP Service

PLC writes trigger registers -> HP60C RGB/depth snapshot -> C++ OBB infer ->
row-median depth height check -> write OK/NG result to holding registers.

This service is independent from robot_gateway/tube_station.
Default port: 1503, to avoid conflict with tube_station default 1502.
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

# Reuse the already verified one-shot RGB OBB + depth stage2 implementation.
import debug_depth_check_once as core

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ENV = THIS_DIR / "carton_tube_check.env"

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
REG_STAND_COUNT = 9
REG_LYING_COUNT = 10
REG_HIGH_COUNT = 11
REG_MAX_HEIGHT_DIFF_MM = 12
REG_VALID_PRED_COUNT = 13
REG_RAW_PRED_COUNT = 14
REG_IMAGE_WIDTH = 15
REG_IMAGE_HEIGHT = 16
REG_DEPTH_WIDTH = 17
REG_DEPTH_HEIGHT = 18
REG_BASELINE_MODE = 19       # 1=row_median, 2=current_frame_median, 3=fixed_env
REG_EXPECTED_ROWS = 20
REG_EXPECTED_COLS = 21
REG_DETECTED_SLOT_COUNT = 22
REG_MISSING_SLOT_COUNT = 23
REG_RESERVED_24 = 24
REG_RESERVED_25 = 25

# 5x8 matrix registers by default. Count is EXPECTED_ROWS * EXPECTED_COLS.
REG_MATRIX_DEPTH_BASE = 30       # uint16 depth mm, 65535=missing
REG_MATRIX_DIFF_BASE = 70        # int16 height diff mm, 32767=missing; positive means closer/higher
REG_MATRIX_HIGH_BASE = 110       # 0=ok, 1=high, 65535=missing
REG_MATRIX_BASELINE_BASE = 150   # uint16 baseline depth mm, 65535=missing

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
    "LYING_DETECTED": 1,
    "STAND_COUNT_LOW": 2,
    "DEPTH_INVALID": 3,
    "HEIGHT_HIGH": 4,
    "INTERNAL_ERROR": 9,
}

BASELINE_MODE_CODE = {
    "row_median": 1,
    "current_frame_median": 2,
    "fixed_env": 3,
    "invalid_fixed_env": 4,
    "invalid": 9,
}

ERR_NONE = 0
ERR_SNAPSHOT_FAILED = 201
ERR_INFER_FAILED = 202
ERR_DEPTH_FAILED = 203
ERR_ANALYZE_FAILED = 204
ERR_JSON_FAILED = 205
ERR_INTERNAL = 301

INVALID_U16 = 65535
INVALID_I16 = 32767

# Load env once here as well; core has already loaded it at import time. This is harmless and
# keeps service-only variables available when running manually.
core.load_env_file(Path(os.environ.get("VISIONOPS_CARTON_TUBE_ENV", str(DEFAULT_ENV))))

ENABLE = core.getenv_int("VISIONOPS_CARTON_TUBE_ENABLE", 1)
HOST = core.getenv_str("VISIONOPS_CARTON_TUBE_MODBUS_HOST", "0.0.0.0")
PORT = core.getenv_int("VISIONOPS_CARTON_TUBE_MODBUS_PORT", 1503)
UNIT_ID = core.getenv_int("VISIONOPS_CARTON_TUBE_MODBUS_UNIT_ID", 1)
ADDRESS_BASE = core.getenv_int("VISIONOPS_CARTON_TUBE_ADDRESS_BASE", 0)
REGISTER_COUNT = core.getenv_int("VISIONOPS_CARTON_TUBE_REGISTER_COUNT", 256)
POLL_INTERVAL_S = max(0.02, core.getenv_int("VISIONOPS_CARTON_TUBE_POLL_INTERVAL_MS", 50) / 1000.0)
LOG_LEVEL = core.getenv_str("VISIONOPS_CARTON_TUBE_LOG_LEVEL", "INFO").upper()
SAVE_RESULT_DIR = core.getenv_str("VISIONOPS_CARTON_TUBE_SAVE_RESULT_DIR", "/tmp/carton_tube_check_latest")
SAVE_EVERY_TRIGGER = core.getenv_int("VISIONOPS_CARTON_TUBE_SAVE_EVERY_TRIGGER", 1)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def u16(v: int) -> int:
    return int(v) & 0xFFFF


def encode_i16(v: Optional[float]) -> int:
    if v is None:
        return INVALID_I16
    try:
        iv = int(round(float(v)))
    except Exception:
        return INVALID_I16
    if iv > 32766:
        iv = 32766
    if iv < -32768:
        iv = -32768
    return iv & 0xFFFF


def encode_u16_mm(v: Any, missing: int = INVALID_U16) -> int:
    if v is None:
        return missing
    try:
        iv = int(round(float(v)))
    except Exception:
        return missing
    if iv < 0:
        return 0
    if iv > 65534:
        return 65534
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
    # Matrix output needs up to REG_MATRIX_BASELINE_BASE + 40 registers by default.
    min_count = REG_MATRIX_BASELINE_BASE + max(40, core.EXPECTED_ROWS * core.EXPECTED_COLS) + 16
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


def matrix_flat(mat: Any, rows: int, cols: int) -> List[Any]:
    out: List[Any] = []
    if not isinstance(mat, list):
        return [None] * (rows * cols)
    for r in range(rows):
        row = mat[r] if r < len(mat) and isinstance(mat[r], list) else []
        for c in range(cols):
            out.append(row[c] if c < len(row) else None)
    return out


def write_matrix_registers(context: ModbusServerContext, result: Dict[str, Any]) -> None:
    rows = int(result.get("expected_rows") or core.EXPECTED_ROWS or 5)
    cols = int(result.get("expected_cols") or core.EXPECTED_COLS or 8)
    n = rows * cols
    grid = result.get("grid") if isinstance(result.get("grid"), dict) else {}

    depth_values = [encode_u16_mm(v) for v in matrix_flat(grid.get("depth_mm"), rows, cols)]
    diff_values = [encode_i16(v) for v in matrix_flat(grid.get("height_diff_mm"), rows, cols)]
    high_values: List[int] = []
    for v in matrix_flat(grid.get("height_high"), rows, cols):
        if v is None:
            high_values.append(INVALID_U16)
        else:
            high_values.append(1 if bool(v) else 0)
    baseline_values = [encode_u16_mm(v) for v in matrix_flat(grid.get("baseline_depth_mm"), rows, cols)]

    set_regs(context, REG_MATRIX_DEPTH_BASE, depth_values[:n])
    set_regs(context, REG_MATRIX_DIFF_BASE, diff_values[:n])
    set_regs(context, REG_MATRIX_HIGH_BASE, high_values[:n])
    set_regs(context, REG_MATRIX_BASELINE_BASE, baseline_values[:n])


def count_missing_slots(result: Dict[str, Any]) -> int:
    rows = int(result.get("expected_rows") or core.EXPECTED_ROWS or 5)
    cols = int(result.get("expected_cols") or core.EXPECTED_COLS or 8)
    grid = result.get("grid") if isinstance(result.get("grid"), dict) else {}
    depth = matrix_flat(grid.get("depth_mm"), rows, cols)
    return sum(1 for x in depth if x is None)


def save_debug_files(seq: int, rgb_bytes: bytes, depth_bytes: bytes, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    if SAVE_EVERY_TRIGGER != 1:
        return
    try:
        out_dir = Path(SAVE_RESULT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "rgb.jpg").write_bytes(rgb_bytes)
        (out_dir / "depth.png").write_bytes(depth_bytes)
        (out_dir / "infer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "last_seq.txt").write_text(str(seq), encoding="utf-8")
    except Exception:
        logging.exception("failed to save debug files")


def run_check_once() -> Dict[str, Any]:
    logging.info("fetch RGB snapshot: %s", core.SNAPSHOT_URL)
    rgb_bytes = core.http_get_bytes(core.SNAPSHOT_URL, core.HTTP_TIMEOUT_S)
    if not rgb_bytes:
        raise RuntimeError("snapshot is empty")

    logging.info("post OBB infer: %s bytes=%d", core.INFER_URL, len(rgb_bytes))
    payload = core.post_multipart_image(core.INFER_URL, rgb_bytes, timeout_s=core.HTTP_TIMEOUT_S)

    logging.info("fetch depth PNG: %s", core.DEPTH_URL)
    depth_bytes = core.http_get_bytes(core.DEPTH_URL, core.HTTP_TIMEOUT_S)
    depth = core.decode_depth_png(depth_bytes)

    result = core.analyze(payload, depth)
    result["_rgb_bytes"] = rgb_bytes
    result["_depth_bytes"] = depth_bytes
    result["_infer_payload"] = payload
    return result


class CartonTubeCheckService:
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
        values[REG_EXPECTED_ROWS] = core.EXPECTED_ROWS
        values[REG_EXPECTED_COLS] = core.EXPECTED_COLS
        set_regs(self.context, 0, values)
        # Clear matrix area to missing values.
        n = max(1, core.EXPECTED_ROWS * core.EXPECTED_COLS)
        set_regs(self.context, REG_MATRIX_DEPTH_BASE, [INVALID_U16] * n)
        set_regs(self.context, REG_MATRIX_DIFF_BASE, [INVALID_I16] * n)
        set_regs(self.context, REG_MATRIX_HIGH_BASE, [INVALID_U16] * n)
        set_regs(self.context, REG_MATRIX_BASELINE_BASE, [INVALID_U16] * n)

    def heartbeat_loop(self) -> None:
        while True:
            self.heartbeat = (self.heartbeat + 1) & 0xFFFF
            set_reg(self.context, REG_HEARTBEAT, self.heartbeat)
            time.sleep(0.5)

    def poll_loop(self) -> None:
        logging.info("Carton tube check trigger poll started")
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
            err = ERR_NONE if status == STATUS_DONE else ERR_ANALYZE_FAILED

            missing = count_missing_slots(result)
            rows = int(result.get("expected_rows") or core.EXPECTED_ROWS)
            cols = int(result.get("expected_cols") or core.EXPECTED_COLS)
            detected_slots = rows * cols - missing

            summary_values = [
                status,
                seq,
                final_code,
                reason_code(reason),
                err,
                elapsed_ms,
                self.heartbeat,
                int(result.get("stand_count") or 0),
                int(result.get("lying_count") or 0),
                int(result.get("high_count") or 0),
                encode_u16_mm(result.get("max_height_diff_mm"), missing=0),
                int(result.get("valid_prediction_count") or 0),
                int(result.get("raw_prediction_count") or 0),
                int(result.get("image_width") or 0),
                int(result.get("image_height") or 0),
                int(result.get("depth_width") or 0),
                int(result.get("depth_height") or 0),
                BASELINE_MODE_CODE.get(str(result.get("baseline_mode") or ""), 0),
                rows,
                cols,
                detected_slots,
                missing,
            ]
            set_regs(self.context, REG_STATUS, summary_values)
            write_matrix_registers(self.context, result)

            rgb_bytes = result.pop("_rgb_bytes", b"")
            depth_bytes = result.pop("_depth_bytes", b"")
            payload = result.pop("_infer_payload", {})
            save_debug_files(seq, rgb_bytes, depth_bytes, payload, result)

            grid = result.get("grid") if isinstance(result.get("grid"), dict) else {}
            if grid:
                logging.info("depth_mm matrix:\n%s", core.format_matrix(grid.get("depth_mm") or []))
                logging.info("height_diff_mm matrix:\n%s", core.format_matrix(grid.get("height_diff_mm") or []))
                logging.info("height_high matrix:\n%s", core.format_matrix(grid.get("height_high") or [], width=7))
            logging.info(
                "trigger done: seq=%d status=%d final=%s reason=%s stand=%d lying=%d high=%d max_diff=%s time=%dms",
                seq,
                status,
                result.get("final_result"),
                reason,
                int(result.get("stand_count") or 0),
                int(result.get("lying_count") or 0),
                int(result.get("high_count") or 0),
                result.get("max_height_diff_mm"),
                elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = int(round((time.time() - start) * 1000))
            msg = str(exc).lower()
            if "snapshot" in msg:
                err = ERR_SNAPSHOT_FAILED
            elif "infer" in msg or "post" in msg:
                err = ERR_INFER_FAILED
            elif "depth" in msg:
                err = ERR_DEPTH_FAILED
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
        logging.warning("VISIONOPS_CARTON_TUBE_ENABLE != 1, exit")
        return 0

    context = create_context()
    service = CartonTubeCheckService(context)
    service.init_registers()

    threading.Thread(target=service.heartbeat_loop, daemon=True).start()
    threading.Thread(target=service.poll_loop, daemon=True).start()

    identity = None
    if ModbusDeviceIdentification is not None:
        identity = ModbusDeviceIdentification()
        identity.VendorName = "VisionOps"
        identity.ProductCode = "CARTON_TUBE"
        identity.ProductName = "VisionOps Carton Tube Check Modbus TCP"
        identity.ModelName = "LB3576 HP60C Carton Tube Check"
        identity.MajorMinorRevision = "1.0"

    logging.info(
        "starting carton tube check: host=%s port=%d unit_id=%d address_base=%d rows=%d cols=%d baseline_mode=%s",
        HOST,
        PORT,
        UNIT_ID,
        ADDRESS_BASE,
        core.EXPECTED_ROWS,
        core.EXPECTED_COLS,
        core.BASELINE_MODE,
    )
    logging.info("snapshot_url=%s", core.SNAPSHOT_URL)
    logging.info("depth_url=%s", core.DEPTH_URL)
    logging.info("infer_url=%s", core.INFER_URL)
    StartTcpServer(context=context, identity=identity, address=(HOST, PORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
