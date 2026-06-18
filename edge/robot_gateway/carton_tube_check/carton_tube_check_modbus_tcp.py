#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Carton Tube/Product Placement Check Modbus TCP Service

Standalone tube service using the robot protocol register map:
  0   Vision heartbeat
  2   product placement result: 0=no trigger/no result, 1=OK, 2=NG/ERROR
  102 product placement trigger: 0=idle, 1=trigger

When trigger 102 is 0, register 2 is cleared to 0.
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
from typing import Any, Dict, List

try:
    from pymodbus.server import StartTcpServer
except Exception:
    from pymodbus.server.sync import StartTcpServer

try:
    from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
except Exception:
    from pymodbus.datastore import ModbusSequentialDataBlock, ModbusDeviceContext as ModbusSlaveContext, ModbusServerContext

try:
    from pymodbus.device import ModbusDeviceIdentification
except Exception:
    ModbusDeviceIdentification = None

import debug_depth_check_once as core

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ENV = THIS_DIR / "carton_tube_check.env"
core.load_env_file(Path(os.environ.get("VISIONOPS_CARTON_TUBE_ENV", str(DEFAULT_ENV))))

REG_VISION_HEARTBEAT = 0
REG_PRODUCT_RESULT = 2
REG_TRIGGER_PRODUCT = 102

RESULT_NONE = 0
RESULT_OK = 1
RESULT_NG = 2

ENABLE = core.getenv_int("VISIONOPS_CARTON_TUBE_ENABLE", 1)
HOST = core.getenv_str("VISIONOPS_CARTON_TUBE_MODBUS_HOST", "0.0.0.0")
PORT = core.getenv_int("VISIONOPS_CARTON_TUBE_MODBUS_PORT", 1503)
UNIT_ID = core.getenv_int("VISIONOPS_CARTON_TUBE_MODBUS_UNIT_ID", 1)
SINGLE_SLAVE = core.getenv_int("VISIONOPS_CARTON_TUBE_MODBUS_SINGLE_SLAVE", 1)
ADDRESS_BASE = core.getenv_int("VISIONOPS_CARTON_TUBE_ADDRESS_BASE", 0)
REGISTER_COUNT = core.getenv_int("VISIONOPS_CARTON_TUBE_REGISTER_COUNT", 200)
POLL_INTERVAL_S = max(0.02, core.getenv_int("VISIONOPS_CARTON_TUBE_POLL_INTERVAL_MS", 50) / 1000.0)
LOG_LEVEL = core.getenv_str("VISIONOPS_CARTON_TUBE_LOG_LEVEL", "INFO").upper()
SAVE_RESULT_DIR = core.getenv_str("VISIONOPS_CARTON_TUBE_SAVE_RESULT_DIR", "/tmp/carton_tube_check_latest")
SAVE_EVERY_TRIGGER = core.getenv_int("VISIONOPS_CARTON_TUBE_SAVE_EVERY_TRIGGER", 1)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")


def u16(v: int) -> int:
    return int(v) & 0xFFFF


def result_code(result: Dict[str, Any]) -> int:
    return RESULT_OK if str(result.get("final_result", "")).upper() == "OK" else RESULT_NG


def create_context() -> ModbusServerContext:
    block_count = max(REGISTER_COUNT, ADDRESS_BASE + 200)
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
    if SINGLE_SLAVE == 1:
        return ModbusServerContext(slaves=slave, single=True)
    return ModbusServerContext(slaves={UNIT_ID: slave}, single=False)


def get_reg(context: ModbusServerContext, reg: int) -> int:
    try:
        values = context[UNIT_ID].getValues(3, ADDRESS_BASE + reg, count=1)
    except TypeError:
        values = context[UNIT_ID].getValues(3, ADDRESS_BASE + reg, 1)
    return int(values[0]) if values else 0


def set_regs(context: ModbusServerContext, offset: int, values: List[int]) -> None:
    context[UNIT_ID].setValues(3, ADDRESS_BASE + offset, [u16(v) for v in values])


def set_reg(context: ModbusServerContext, reg: int, value: int) -> None:
    set_regs(context, reg, [value])


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


def save_debug_files(rgb_bytes: bytes, depth_bytes: bytes, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    if SAVE_EVERY_TRIGGER != 1:
        return
    try:
        out_dir = Path(SAVE_RESULT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "rgb.jpg").write_bytes(rgb_bytes)
        (out_dir / "depth.png").write_bytes(depth_bytes)
        (out_dir / "infer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logging.exception("failed to save debug files")


class TubeProtocolService:
    def __init__(self, context: ModbusServerContext) -> None:
        self.context = context
        self.lock = threading.Lock()
        self.busy = False
        self.heartbeat = 0
        self.last_cmd = 0

    def init_registers(self) -> None:
        set_regs(self.context, 0, [0] * max(REGISTER_COUNT, 200))

    def heartbeat_loop(self) -> None:
        while True:
            self.heartbeat = 0 if self.heartbeat >= 1000 else self.heartbeat + 1
            set_reg(self.context, REG_VISION_HEARTBEAT, self.heartbeat)
            time.sleep(0.5)

    def poll_loop(self) -> None:
        logging.info("Carton tube robot-protocol trigger poll started")
        while True:
            try:
                cmd = get_reg(self.context, REG_TRIGGER_PRODUCT)
                if cmd == 0:
                    self.last_cmd = 0
                    set_reg(self.context, REG_PRODUCT_RESULT, RESULT_NONE)
                elif cmd == 1 and self.last_cmd == 0 and not self.busy:
                    self.last_cmd = 1
                    threading.Thread(target=self.handle_trigger, daemon=True).start()
            except Exception:
                logging.exception("trigger poll error")
            time.sleep(POLL_INTERVAL_S)

    def handle_trigger(self) -> None:
        with self.lock:
            if self.busy:
                return
            self.busy = True
        set_reg(self.context, REG_PRODUCT_RESULT, RESULT_NONE)
        start = time.time()
        try:
            result = run_check_once()
            code = result_code(result)
            set_reg(self.context, REG_PRODUCT_RESULT, code)
            rgb_bytes = result.pop("_rgb_bytes", b"")
            depth_bytes = result.pop("_depth_bytes", b"")
            payload = result.pop("_infer_payload", {})
            save_debug_files(rgb_bytes, depth_bytes, payload, result)
            logging.info("tube/product check done: final=%s code=%d reason=%s time=%dms", result.get("final_result"), code, result.get("reason"), int((time.time()-start)*1000))
        except Exception as exc:
            logging.error("tube/product check failed: %s", exc)
            logging.debug("%s", traceback.format_exc())
            set_reg(self.context, REG_PRODUCT_RESULT, RESULT_NG)
        finally:
            with self.lock:
                self.busy = False


def main() -> int:
    if ENABLE != 1:
        logging.warning("VISIONOPS_CARTON_TUBE_ENABLE != 1, exit")
        return 0
    context = create_context()
    service = TubeProtocolService(context)
    service.init_registers()
    threading.Thread(target=service.heartbeat_loop, daemon=True).start()
    threading.Thread(target=service.poll_loop, daemon=True).start()

    identity = None
    if ModbusDeviceIdentification is not None:
        identity = ModbusDeviceIdentification()
        identity.VendorName = "VisionOps"
        identity.ProductCode = "TUBE_PROTO"
        identity.ProductName = "VisionOps Carton Tube Robot Protocol"
        identity.ModelName = "LB3576 HP60C Carton Tube Check"
        identity.MajorMinorRevision = "1.0"

    logging.info("starting carton tube protocol service: host=%s port=%d unit_id=%d single_slave=%d address_base=%d register_count=%d", HOST, PORT, UNIT_ID, SINGLE_SLAVE, ADDRESS_BASE, max(REGISTER_COUNT, 200))
    logging.info("registers: result 2, trigger 102")
    StartTcpServer(context=context, identity=identity, address=(HOST, PORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
