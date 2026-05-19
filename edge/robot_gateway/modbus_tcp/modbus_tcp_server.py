#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Modbus TCP Server v1.2.1 - unified register map v2

功能：
1. 启动一个 Modbus TCP Server，默认监听 0.0.0.0:1502。
2. 周期性读取 VisionOps C++ 最新检测结果接口。
3. 调用 modbus_common.register_mapper 生成统一 register_map_v2。
4. 电脑 / 上位机 / PLC 通过 Modbus TCP 功能码 03 读取寄存器。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import threading
import urllib.request
from typing import Any, Dict, Optional, Set

try:
    from pymodbus.server import StartTcpServer
except Exception:
    from pymodbus.server.sync import StartTcpServer

from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext

try:
    from pymodbus.device import ModbusDeviceIdentification
except Exception:
    ModbusDeviceIdentification = None

# Allow importing sibling package: edge/robot_gateway/modbus_common
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_GATEWAY_DIR = os.path.dirname(THIS_DIR)
if ROBOT_GATEWAY_DIR not in sys.path:
    sys.path.insert(0, ROBOT_GATEWAY_DIR)

from modbus_common.register_mapper import (  # noqa: E402
    build_registers,
    DEFAULT_REGISTER_COUNT,
    DEFAULT_MAX_ITEMS,
)


def load_env_file(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


DEFAULT_ENV = os.path.join(THIS_DIR, "modbus_tcp.env")
load_env_file(os.environ.get("VISIONOPS_MODBUS_TCP_ENV", DEFAULT_ENV))


def getenv_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def getenv_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


ENABLE = getenv_int("VISIONOPS_MODBUS_TCP_ENABLE", 1)
HOST = getenv_str("VISIONOPS_MODBUS_TCP_HOST", "0.0.0.0")
PORT = getenv_int("VISIONOPS_MODBUS_TCP_PORT", 1502)
UNIT_ID = getenv_int("VISIONOPS_MODBUS_TCP_UNIT_ID", 1)
REGISTER_COUNT = getenv_int("VISIONOPS_MODBUS_TCP_REGISTER_COUNT", DEFAULT_REGISTER_COUNT)
MAX_OBJECTS = getenv_int("VISIONOPS_MODBUS_TCP_MAX_OBJECTS", DEFAULT_MAX_ITEMS)
RESULT_URL = getenv_str("VISIONOPS_RESULT_URL", "http://127.0.0.1:8090/api/cpp/stream/latest_result")
POLL_INTERVAL_MS = getenv_int("VISIONOPS_MODBUS_TCP_POLL_INTERVAL_MS", 100)
LOG_LEVEL = getenv_str("VISIONOPS_MODBUS_TCP_LOG_LEVEL", "INFO").upper()
NG_CLASS_IDS_RAW = getenv_str("VISIONOPS_MODBUS_TCP_NG_CLASS_IDS", "").strip()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

if NG_CLASS_IDS_RAW:
    NG_CLASS_IDS: Set[int] = {int(x.strip()) for x in NG_CLASS_IDS_RAW.split(",") if x.strip().isdigit()}
else:
    NG_CLASS_IDS = set()


def fetch_json(url: str, timeout_s: float = 0.5) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw)
    except Exception as e:
        logging.debug("fetch latest_result failed: %s", e)
        return None


def create_context() -> ModbusServerContext:
    block = ModbusSequentialDataBlock(0, [0] * REGISTER_COUNT)
    try:
        slave = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * REGISTER_COUNT),
            co=ModbusSequentialDataBlock(0, [0] * REGISTER_COUNT),
            hr=block,
            ir=ModbusSequentialDataBlock(0, [0] * REGISTER_COUNT),
            zero_mode=True,
        )
    except TypeError:
        slave = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * REGISTER_COUNT),
            co=ModbusSequentialDataBlock(0, [0] * REGISTER_COUNT),
            hr=block,
            ir=ModbusSequentialDataBlock(0, [0] * REGISTER_COUNT),
        )
    return ModbusServerContext(slaves={UNIT_ID: slave}, single=False)


def set_holding_registers(context: ModbusServerContext, values):
    context[UNIT_ID].setValues(3, 0, values)


def updater_loop(context: ModbusServerContext) -> None:
    heartbeat = 0
    logging.info("Result updater started, url=%s", RESULT_URL)

    while True:
        try:
            heartbeat = (heartbeat + 1) & 0xFFFF
            payload = fetch_json(RESULT_URL)
            regs = build_registers(
                payload=payload,
                heartbeat=heartbeat,
                register_count=REGISTER_COUNT,
                max_items=MAX_OBJECTS,
                ng_class_ids=NG_CLASS_IDS,
            )
            set_holding_registers(context, regs)

            if heartbeat % 50 == 0:
                logging.info(
                    "updated registers: heartbeat=%d valid=%d task_type=%d schema=%d count=%d status=%d",
                    regs[2], regs[4], regs[5], regs[6], regs[16], regs[3],
                )
        except Exception as e:
            logging.exception("updater loop error: %s", e)

        time.sleep(max(0.02, POLL_INTERVAL_MS / 1000.0))


def main() -> int:
    if ENABLE != 1:
        logging.warning("VISIONOPS_MODBUS_TCP_ENABLE != 1, exit.")
        return 0

    context = create_context()
    t = threading.Thread(target=updater_loop, args=(context,), daemon=True)
    t.start()

    identity = None
    if ModbusDeviceIdentification is not None:
        identity = ModbusDeviceIdentification()
        identity.VendorName = "VisionOps"
        identity.ProductCode = "VOPS"
        identity.VendorUrl = "https://github.com/MaCL0927/visionops_v2"
        identity.ProductName = "VisionOps Modbus TCP Server"
        identity.ModelName = "VisionOps Edge Box"
        identity.MajorMinorRevision = "1.2"

    logging.info("starting Modbus TCP server v1.2.1: host=%s port=%d unit_id=%d registers=%d", HOST, PORT, UNIT_ID, REGISTER_COUNT)
    StartTcpServer(context=context, identity=identity, address=(HOST, PORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
