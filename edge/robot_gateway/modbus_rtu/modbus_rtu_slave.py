#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Modbus RTU Slave v1.2.1 - GPIO-aware + unified register map v2

适用场景：
- Neardi LPR3576 / LB3576 RS485: /dev/ttyS5
- RS485 半双工方向控制 GPIO136
  - GPIO136=0: 接收
  - GPIO136=1: 发送

功能：
1. 周期性读取 VisionOps C++ 最新检测结果接口。
2. 调用 modbus_common.register_mapper 生成统一 register_map_v2。
3. 手写 Modbus RTU 功能码 03，从 /dev/ttyS5 读取请求并返回寄存器。
4. 回复前 GPIO136=1，回复完成后 GPIO136=0。

当前支持：
- Function 03: Read Holding Registers
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import threading
import urllib.request
from typing import Any, Dict, List, Optional, Set

try:
    import serial
except Exception:
    print("[ERROR] pyserial is required. Install with: pip install pyserial", file=sys.stderr)
    raise

# Allow importing sibling package: edge/robot_gateway/modbus_common
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_GATEWAY_DIR = os.path.dirname(THIS_DIR)
if ROBOT_GATEWAY_DIR not in sys.path:
    sys.path.insert(0, ROBOT_GATEWAY_DIR)

from modbus_common.register_mapper import (  # noqa: E402
    build_registers,
    clamp_u16,
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


DEFAULT_ENV = os.path.join(THIS_DIR, "modbus_rtu.env")
load_env_file(os.environ.get("VISIONOPS_MODBUS_ENV", DEFAULT_ENV))


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


def getenv_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


ENABLE = getenv_int("VISIONOPS_MODBUS_ENABLE", 1)
SERIAL_PORT = getenv_str("VISIONOPS_MODBUS_SERIAL", "/dev/ttyS5")
SLAVE_ID = getenv_int("VISIONOPS_MODBUS_SLAVE_ID", 1)
BAUDRATE = getenv_int("VISIONOPS_MODBUS_BAUDRATE", 9600)
PARITY = getenv_str("VISIONOPS_MODBUS_PARITY", "N").upper()
BYTESIZE = getenv_int("VISIONOPS_MODBUS_BYTESIZE", 8)
STOPBITS = getenv_int("VISIONOPS_MODBUS_STOPBITS", 1)
REGISTER_COUNT = getenv_int("VISIONOPS_MODBUS_REGISTER_COUNT", DEFAULT_REGISTER_COUNT)
MAX_OBJECTS = getenv_int("VISIONOPS_MODBUS_MAX_OBJECTS", DEFAULT_MAX_ITEMS)
RESULT_URL = getenv_str("VISIONOPS_RESULT_URL", "http://127.0.0.1:8090/api/cpp/stream/latest_result")
POLL_INTERVAL_MS = getenv_int("VISIONOPS_MODBUS_POLL_INTERVAL_MS", 100)
LOG_LEVEL = getenv_str("VISIONOPS_MODBUS_LOG_LEVEL", "INFO").upper()
NG_CLASS_IDS_RAW = getenv_str("VISIONOPS_MODBUS_NG_CLASS_IDS", "").strip()

# LPR3576 / LB3576 RS485 half-duplex direction control
GPIO_ENABLE = getenv_int("VISIONOPS_MODBUS_GPIO_ENABLE", 1)
GPIO_NUM = getenv_int("VISIONOPS_MODBUS_GPIO_NUM", 136)
GPIO_TX_VALUE = getenv_str("VISIONOPS_MODBUS_GPIO_TX_VALUE", "1")
GPIO_RX_VALUE = getenv_str("VISIONOPS_MODBUS_GPIO_RX_VALUE", "0")
TX_PRE_DELAY_MS = getenv_float("VISIONOPS_MODBUS_TX_PRE_DELAY_MS", 2.0)
TX_POST_DELAY_MS = getenv_float("VISIONOPS_MODBUS_TX_POST_DELAY_MS", 5.0)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

if NG_CLASS_IDS_RAW:
    NG_CLASS_IDS: Set[int] = {int(x.strip()) for x in NG_CLASS_IDS_RAW.split(",") if x.strip().isdigit()}
else:
    NG_CLASS_IDS = set()

registers_lock = threading.Lock()
holding_registers: List[int] = [0] * REGISTER_COUNT


def fetch_json(url: str, timeout_s: float = 0.5) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw)
    except Exception as e:
        logging.debug("fetch latest_result failed: %s", e)
        return None


# -----------------------------
# GPIO direction control
# -----------------------------

def gpio_base() -> str:
    return f"/sys/class/gpio/gpio{GPIO_NUM}"


def gpio_value_path() -> str:
    return os.path.join(gpio_base(), "value")


def gpio_write(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(value))


def setup_gpio() -> None:
    if GPIO_ENABLE != 1:
        logging.warning("GPIO direction control disabled")
        return

    if not os.path.exists(gpio_base()):
        try:
            gpio_write("/sys/class/gpio/export", str(GPIO_NUM))
        except OSError:
            pass

    if not os.path.exists(gpio_base()):
        raise RuntimeError(f"GPIO{GPIO_NUM} export failed: {gpio_base()} not found")

    gpio_write(os.path.join(gpio_base(), "direction"), "out")
    set_gpio_rx()
    logging.info("RS485 GPIO direction ready: gpio=%d rx=%s tx=%s", GPIO_NUM, GPIO_RX_VALUE, GPIO_TX_VALUE)


def set_gpio_tx() -> None:
    if GPIO_ENABLE == 1:
        gpio_write(gpio_value_path(), GPIO_TX_VALUE)
        if TX_PRE_DELAY_MS > 0:
            time.sleep(TX_PRE_DELAY_MS / 1000.0)


def set_gpio_rx() -> None:
    if GPIO_ENABLE == 1:
        gpio_write(gpio_value_path(), GPIO_RX_VALUE)


# -----------------------------
# Modbus RTU frame helpers
# -----------------------------

def modbus_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def append_crc(data: bytes) -> bytes:
    crc = modbus_crc16(data)
    return data + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def valid_crc(frame: bytes) -> bool:
    if len(frame) < 4:
        return False
    got = frame[-2] | (frame[-1] << 8)
    calc = modbus_crc16(frame[:-2])
    return got == calc


def make_exception_response(slave_id: int, function_code: int, exception_code: int) -> bytes:
    pdu = bytes([slave_id & 0xFF, (function_code | 0x80) & 0xFF, exception_code & 0xFF])
    return append_crc(pdu)


def make_read_holding_response(slave_id: int, start: int, qty: int) -> bytes:
    with registers_lock:
        regs = holding_registers[start:start + qty]

    payload = bytearray()
    payload.append(slave_id & 0xFF)
    payload.append(0x03)
    payload.append((qty * 2) & 0xFF)
    for v in regs:
        v = clamp_u16(v)
        payload.append((v >> 8) & 0xFF)
        payload.append(v & 0xFF)
    return append_crc(bytes(payload))


def handle_request(frame: bytes) -> Optional[bytes]:
    if len(frame) < 8:
        logging.debug("ignore short frame: %s", frame.hex(" "))
        return None

    if not valid_crc(frame):
        logging.warning("bad crc frame: %s", frame.hex(" "))
        return None

    slave = frame[0]
    func = frame[1]

    if slave == 0:
        # Broadcast request has no response.
        return None
    if slave != SLAVE_ID:
        logging.debug("ignore frame for slave %d", slave)
        return None

    if func != 0x03:
        logging.warning("unsupported function code: %d", func)
        return make_exception_response(SLAVE_ID, func, 0x01)

    if len(frame) != 8:
        logging.warning("invalid fc03 frame length=%d: %s", len(frame), frame.hex(" "))
        return make_exception_response(SLAVE_ID, func, 0x03)

    start = (frame[2] << 8) | frame[3]
    qty = (frame[4] << 8) | frame[5]

    if qty < 1 or qty > 125:
        logging.warning("invalid quantity: start=%d qty=%d", start, qty)
        return make_exception_response(SLAVE_ID, func, 0x03)

    if start < 0 or start + qty > REGISTER_COUNT:
        logging.warning("invalid address range: start=%d qty=%d count=%d", start, qty, REGISTER_COUNT)
        return make_exception_response(SLAVE_ID, func, 0x02)

    return make_read_holding_response(SLAVE_ID, start, qty)


# -----------------------------
# Background register updater
# -----------------------------

def updater_loop() -> None:
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
            with registers_lock:
                holding_registers[:] = regs

            if heartbeat % 50 == 0:
                logging.info(
                    "updated registers: heartbeat=%d valid=%d task_type=%d schema=%d count=%d status=%d",
                    regs[2], regs[4], regs[5], regs[6], regs[16], regs[3],
                )
        except Exception as e:
            logging.exception("updater loop error: %s", e)

        time.sleep(max(0.02, POLL_INTERVAL_MS / 1000.0))


# -----------------------------
# Serial RTU slave loop
# -----------------------------

def serial_parity_value(parity: str) -> str:
    if parity == "E":
        return serial.PARITY_EVEN
    if parity == "O":
        return serial.PARITY_ODD
    return serial.PARITY_NONE


def read_exact_request(ser: serial.Serial) -> Optional[bytes]:
    """Read one FC03 Modbus RTU request, exactly 8 bytes."""
    first = ser.read(1)
    if not first:
        return None
    rest = ser.read(7)
    frame = first + rest
    if len(frame) < 8:
        time.sleep(0.02)
        more = ser.read(256)
        frame += more
        logging.warning("incomplete frame length=%d: %s", len(frame), frame.hex(" "))
        return None
    return frame


def send_response(ser: serial.Serial, response: bytes) -> None:
    set_gpio_tx()
    try:
        ser.write(response)
        ser.flush()
        char_time_s = 11.0 / float(BAUDRATE) if BAUDRATE > 0 else 0.002
        tx_time_s = len(response) * char_time_s
        time.sleep(tx_time_s + TX_POST_DELAY_MS / 1000.0)
    finally:
        set_gpio_rx()


def rtu_slave_loop() -> None:
    bytesize_map = {5: serial.FIVEBITS, 6: serial.SIXBITS, 7: serial.SEVENBITS, 8: serial.EIGHTBITS}
    stopbits_map = {1: serial.STOPBITS_ONE, 2: serial.STOPBITS_TWO}

    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        bytesize=bytesize_map.get(BYTESIZE, serial.EIGHTBITS),
        parity=serial_parity_value(PARITY),
        stopbits=stopbits_map.get(STOPBITS, serial.STOPBITS_ONE),
        timeout=0.2,
        write_timeout=1.0,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )

    set_gpio_rx()
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    logging.info(
        "starting GPIO-aware Modbus RTU slave v1.2.1: port=%s slave_id=%d baudrate=%d %d%s%d registers=%d gpio=%d",
        SERIAL_PORT, SLAVE_ID, BAUDRATE, BYTESIZE, PARITY, STOPBITS, REGISTER_COUNT, GPIO_NUM,
    )
    logging.info("Server listening.")

    req_count = 0
    resp_count = 0

    try:
        while True:
            frame = read_exact_request(ser)
            if not frame:
                continue
            req_count += 1
            if req_count <= 5 or req_count % 100 == 0:
                logging.info("request #%d: %s", req_count, frame.hex(" "))

            response = handle_request(frame)
            if response:
                send_response(ser, response)
                resp_count += 1
                if resp_count <= 5 or resp_count % 100 == 0:
                    logging.info("response #%d: %s", resp_count, response.hex(" "))
    finally:
        try:
            set_gpio_rx()
        finally:
            ser.close()


def main() -> int:
    if ENABLE != 1:
        logging.warning("VISIONOPS_MODBUS_ENABLE != 1, exit.")
        return 0
    if not os.path.exists(SERIAL_PORT):
        logging.error("serial port not found: %s", SERIAL_PORT)
        logging.error("please check: ls -l /dev/ttyS* /dev/ttyAMA* /dev/ttyUSB*")
        return 2

    setup_gpio()

    t = threading.Thread(target=updater_loop, daemon=True)
    t.start()

    rtu_slave_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
