#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Modbus RTU Master Push v1.3.2

适用场景：
- PLC 作为 Modbus RTU Slave。
- 3576 作为 Modbus RTU Master。
- 3576 周期性读取 /api/cpp/stream/latest_result，
  调用 modbus_common.register_mapper 生成 register_map_v2，
  然后使用功能码 06/16 主动写入 PLC。
- v1.3.2 增强响应帧扫描和自动重试，降低偶发 CRC / 短帧导致的误判失败。

注意：
- 该服务会主动占用 /dev/ttyS5。
- 不要和 visionops-modbus-rtu.service（从站服务）同时使用同一个 /dev/ttyS5。
- LPR3576 / LB3576 RS485 半双工使用 GPIO136 控制方向。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import termios
import time
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import serial
except Exception:
    print("[ERROR] pyserial is required. Install with: pip install pyserial", file=sys.stderr)
    raise

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_GATEWAY_DIR = os.path.dirname(THIS_DIR)
if ROBOT_GATEWAY_DIR not in sys.path:
    sys.path.insert(0, ROBOT_GATEWAY_DIR)

from modbus_common.register_mapper import (  # noqa: E402
    build_registers,
    clamp_u16,
    describe_registers,
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


DEFAULT_ENV = os.path.join(THIS_DIR, "modbus_rtu_master.env")
load_env_file(os.environ.get("VISIONOPS_MODBUS_MASTER_ENV", DEFAULT_ENV))


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


ENABLE = getenv_int("VISIONOPS_MODBUS_MASTER_ENABLE", 1)
SERIAL_PORT = getenv_str("VISIONOPS_MODBUS_SERIAL", "/dev/ttyS5")
TARGET_SLAVE_ID = getenv_int("VISIONOPS_MODBUS_TARGET_SLAVE_ID", 1)
BAUDRATE = getenv_int("VISIONOPS_MODBUS_BAUDRATE", 9600)
PARITY = getenv_str("VISIONOPS_MODBUS_PARITY", "N").upper()
BYTESIZE = getenv_int("VISIONOPS_MODBUS_BYTESIZE", 8)
STOPBITS = getenv_int("VISIONOPS_MODBUS_STOPBITS", 1)

TARGET_ADDRESS = getenv_int("VISIONOPS_MODBUS_TARGET_ADDRESS", 4096)
SOURCE_START = getenv_int("VISIONOPS_MODBUS_SOURCE_START", 0)
WRITE_COUNT = getenv_int("VISIONOPS_MODBUS_WRITE_COUNT", 120)
WRITE_CHUNK_SIZE = getenv_int("VISIONOPS_MODBUS_WRITE_CHUNK_SIZE", 120)
PUSH_INTERVAL_MS = getenv_int("VISIONOPS_MODBUS_PUSH_INTERVAL_MS", 100)
WRITE_ONLY_ON_CHANGE = getenv_int("VISIONOPS_MODBUS_WRITE_ONLY_ON_CHANGE", 0)

# Write function:
# - "16": always use FC16 Write Multiple Registers
# - "06": use FC06 Write Single Register; requires each chunk qty=1
# - "auto": use FC06 when qty=1, otherwise FC16
WRITE_FUNCTION = getenv_str("VISIONOPS_MODBUS_WRITE_FUNCTION", "auto").lower()
VERIFY_AFTER_WRITE = getenv_int("VISIONOPS_MODBUS_VERIFY_AFTER_WRITE", 0)
REQUIRE_RESPONSE = getenv_int("VISIONOPS_MODBUS_REQUIRE_RESPONSE", 1)

REGISTER_COUNT = getenv_int("VISIONOPS_MODBUS_REGISTER_COUNT", DEFAULT_REGISTER_COUNT)
MAX_OBJECTS = getenv_int("VISIONOPS_MODBUS_MAX_OBJECTS", DEFAULT_MAX_ITEMS)
RESULT_URL = getenv_str("VISIONOPS_RESULT_URL", "http://127.0.0.1:8090/api/cpp/stream/latest_result")
NG_CLASS_IDS_RAW = getenv_str("VISIONOPS_MODBUS_NG_CLASS_IDS", "").strip()

GPIO_ENABLE = getenv_int("VISIONOPS_MODBUS_GPIO_ENABLE", 1)
GPIO_NUM = getenv_int("VISIONOPS_MODBUS_GPIO_NUM", 136)
GPIO_TX_VALUE = getenv_str("VISIONOPS_MODBUS_GPIO_TX_VALUE", "1")
GPIO_RX_VALUE = getenv_str("VISIONOPS_MODBUS_GPIO_RX_VALUE", "0")
TX_PRE_DELAY_MS = getenv_float("VISIONOPS_MODBUS_TX_PRE_DELAY_MS", 1.0)
# Kept for backward compatibility. In v1.3.1 this delay is applied after switching back to RX,
# not while GPIO is still in TX mode.
TX_POST_DELAY_MS = getenv_float("VISIONOPS_MODBUS_TX_POST_DELAY_MS", 0.0)
RX_SETTLE_DELAY_MS = getenv_float("VISIONOPS_MODBUS_RX_SETTLE_DELAY_MS", TX_POST_DELAY_MS)
RESPONSE_TIMEOUT_MS = getenv_float("VISIONOPS_MODBUS_RESPONSE_TIMEOUT_MS", 1000.0)

# v1.3.2 reliability options:
# - response scanner ignores stale/noisy bytes and searches for a valid RTU frame.
# - retry repeats the whole write/read transaction after transient CRC/short-frame/timeout errors.
RESPONSE_SCAN_MAX_BYTES = getenv_int("VISIONOPS_MODBUS_RESPONSE_SCAN_MAX_BYTES", 256)
RETRY_COUNT = getenv_int("VISIONOPS_MODBUS_RETRY_COUNT", 2)
RETRY_DELAY_MS = getenv_float("VISIONOPS_MODBUS_RETRY_DELAY_MS", 50.0)

LOG_LEVEL = getenv_str("VISIONOPS_MODBUS_LOG_LEVEL", "INFO").upper()

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
# Modbus RTU helpers
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


def serial_parity_value(parity: str) -> str:
    if parity == "E":
        return serial.PARITY_EVEN
    if parity == "O":
        return serial.PARITY_ODD
    return serial.PARITY_NONE


def make_write_multiple_request(slave_id: int, start_addr: int, values: List[int]) -> bytes:
    qty = len(values)
    if qty < 1 or qty > 123:
        raise ValueError(f"FC16 write quantity must be 1..123, got {qty}")
    byte_count = qty * 2
    payload = bytearray()
    payload.append(slave_id & 0xFF)
    payload.append(0x10)
    payload.append((start_addr >> 8) & 0xFF)
    payload.append(start_addr & 0xFF)
    payload.append((qty >> 8) & 0xFF)
    payload.append(qty & 0xFF)
    payload.append(byte_count & 0xFF)
    for v in values:
        v = clamp_u16(v)
        payload.append((v >> 8) & 0xFF)
        payload.append(v & 0xFF)
    return append_crc(bytes(payload))




def make_write_single_request(slave_id: int, start_addr: int, value: int) -> bytes:
    """Function code 06: Write Single Register."""
    v = clamp_u16(value)
    payload = bytes([
        slave_id & 0xFF,
        0x06,
        (start_addr >> 8) & 0xFF,
        start_addr & 0xFF,
        (v >> 8) & 0xFF,
        v & 0xFF,
    ])
    return append_crc(payload)


def make_read_holding_request(slave_id: int, start_addr: int, qty: int) -> bytes:
    """Function code 03: Read Holding Registers."""
    if qty < 1 or qty > 125:
        raise ValueError(f"FC03 read quantity must be 1..125, got {qty}")
    payload = bytes([
        slave_id & 0xFF,
        0x03,
        (start_addr >> 8) & 0xFF,
        start_addr & 0xFF,
        (qty >> 8) & 0xFF,
        qty & 0xFF,
    ])
    return append_crc(payload)


def expected_write_response(slave_id: int, start_addr: int, qty: int) -> bytes:
    payload = bytes([
        slave_id & 0xFF,
        0x10,
        (start_addr >> 8) & 0xFF,
        start_addr & 0xFF,
        (qty >> 8) & 0xFF,
        qty & 0xFF,
    ])
    return append_crc(payload)


def send_rtu_frame(ser: serial.Serial, frame: bytes) -> None:
    """Send one RTU frame on half-duplex RS485.

    v1.3.1 critical change:
    - wait until the UART has physically transmitted all bytes using tcdrain()
    - immediately switch GPIO back to RX
    - only then apply RX settle delay

    The old implementation kept GPIO in TX mode for an estimated frame time plus
    TX_POST_DELAY_MS, which can miss a fast PLC acknowledgement.
    """
    # Drop stale bytes before a new transaction.
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    set_gpio_tx()
    try:
        ser.write(frame)
        ser.flush()
        try:
            termios.tcdrain(ser.fileno())
        except Exception:
            # Fallback: estimate physical transmit time.
            char_time_s = 11.0 / float(BAUDRATE) if BAUDRATE > 0 else 0.002
            time.sleep(len(frame) * char_time_s)
    finally:
        # Switch back to receive as soon as the request has left the UART.
        set_gpio_rx()

    if RX_SETTLE_DELAY_MS > 0:
        time.sleep(RX_SETTLE_DELAY_MS / 1000.0)


def extract_valid_response(
    raw: bytes,
    slave_id: int,
    function_code: int,
    expected_len: int,
) -> Optional[bytes]:
    """Find a valid Modbus RTU response frame inside a raw byte stream.

    Why this is needed:
    Some RS485/USB/PLC combinations occasionally leave a stale byte, echo, or
    partial frame in the UART buffer. A naive reader that takes the first 8 bytes
    can report "bad crc" even though a valid response appears a few bytes later.

    Supported frames:
    - Normal FC06/FC16 response: 8 bytes
    - Normal FC03 response: slave, 0x03, byte_count, payload..., crc_lo, crc_hi
    - Exception response: 5 bytes
    """
    if not raw:
        return None

    max_i = max(0, len(raw) - 4)
    expected_slave = slave_id & 0xFF
    expected_func = function_code & 0xFF
    exception_func = expected_func | 0x80

    for i in range(0, max_i + 1):
        if raw[i] != expected_slave:
            continue
        if i + 2 > len(raw):
            break

        func = raw[i + 1]

        # Exception response: slave, func|0x80, exception_code, crc_lo, crc_hi
        if func == exception_func and i + 5 <= len(raw):
            candidate = raw[i:i + 5]
            if valid_crc(candidate):
                return bytes(candidate)

        # Normal response.
        if func != expected_func:
            continue

        if expected_func == 0x03:
            if i + 3 > len(raw):
                continue
            byte_count = raw[i + 2]
            frame_len = 3 + byte_count + 2
            if frame_len < 5:
                continue
            if i + frame_len <= len(raw):
                candidate = raw[i:i + frame_len]
                if valid_crc(candidate):
                    return bytes(candidate)
        else:
            frame_len = expected_len
            if i + frame_len <= len(raw):
                candidate = raw[i:i + frame_len]
                if valid_crc(candidate):
                    return bytes(candidate)

    return None


def read_response(
    ser: serial.Serial,
    expected_len: int = 8,
    slave_id: int = TARGET_SLAVE_ID,
    function_code: int = 0x10,
) -> bytes:
    """Read a response and return the first valid RTU frame found.

    On timeout, returns the raw bytes captured so validation/logging can show
    what was actually seen on the bus.
    """
    deadline = time.time() + RESPONSE_TIMEOUT_MS / 1000.0
    raw = bytearray()

    while time.time() < deadline:
        chunk = ser.read(256)
        if chunk:
            raw.extend(chunk)

            frame = extract_valid_response(
                bytes(raw),
                slave_id=slave_id,
                function_code=function_code,
                expected_len=expected_len,
            )
            if frame is not None:
                return frame

            # Avoid unbounded growth if the line is noisy.
            if len(raw) > RESPONSE_SCAN_MAX_BYTES:
                del raw[:-RESPONSE_SCAN_MAX_BYTES]
        else:
            time.sleep(0.001)

    return bytes(raw)


def validate_write_response(resp: bytes, slave_id: int, function_code: int, start_addr: int, qty: int, value: int = 0) -> Tuple[bool, str]:
    if not resp:
        return False, "timeout/no response"
    if len(resp) < 5:
        return False, f"short response len={len(resp)}: {resp.hex(' ')}"
    if not valid_crc(resp):
        return False, f"bad crc: {resp.hex(' ')}"
    if resp[0] != (slave_id & 0xFF):
        return False, f"unexpected slave id {resp[0]}, expected {slave_id}"
    if resp[1] & 0x80:
        exc = resp[2] if len(resp) >= 3 else -1
        return False, f"exception response func=0x{resp[1]:02x} code=0x{exc:02x} frame={resp.hex(' ')}"
    if resp[1] != function_code:
        return False, f"unexpected function code 0x{resp[1]:02x}, expected 0x{function_code:02x}"
    if len(resp) < 8:
        return False, f"short normal response len={len(resp)}"

    got_start = (resp[2] << 8) | resp[3]
    if got_start != start_addr:
        return False, f"unexpected ack start={got_start}, expected start={start_addr}"

    if function_code == 0x10:
        got_qty = (resp[4] << 8) | resp[5]
        if got_qty != qty:
            return False, f"unexpected ack qty={got_qty}, expected qty={qty}"
    elif function_code == 0x06:
        got_value = (resp[4] << 8) | resp[5]
        if got_value != clamp_u16(value):
            return False, f"unexpected ack value={got_value}, expected value={clamp_u16(value)}"

    return True, "ok"


def validate_read_response(resp: bytes, slave_id: int, values: List[int]) -> Tuple[bool, str]:
    if not resp:
        return False, "timeout/no response"
    if len(resp) < 5:
        return False, f"short response len={len(resp)}: {resp.hex(' ')}"
    if not valid_crc(resp):
        return False, f"bad crc: {resp.hex(' ')}"
    if resp[0] != (slave_id & 0xFF):
        return False, f"unexpected slave id {resp[0]}, expected {slave_id}"
    if resp[1] & 0x80:
        exc = resp[2] if len(resp) >= 3 else -1
        return False, f"exception response func=0x{resp[1]:02x} code=0x{exc:02x} frame={resp.hex(' ')}"
    if resp[1] != 0x03:
        return False, f"unexpected function code 0x{resp[1]:02x}, expected 0x03"
    byte_count = resp[2]
    if byte_count != len(values) * 2:
        return False, f"unexpected byte_count={byte_count}, expected={len(values) * 2}"
    got = []
    pos = 3
    for _ in values:
        got.append((resp[pos] << 8) | resp[pos + 1])
        pos += 2
    expected = [clamp_u16(v) for v in values]
    if got != expected:
        return False, f"readback mismatch got={got[:8]} expected={expected[:8]}"
    return True, "ok"


def open_serial() -> serial.Serial:
    bytesize_map = {5: serial.FIVEBITS, 6: serial.SIXBITS, 7: serial.SEVENBITS, 8: serial.EIGHTBITS}
    stopbits_map = {1: serial.STOPBITS_ONE, 2: serial.STOPBITS_TWO}
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        bytesize=bytesize_map.get(BYTESIZE, serial.EIGHTBITS),
        parity=serial_parity_value(PARITY),
        stopbits=stopbits_map.get(STOPBITS, serial.STOPBITS_ONE),
        timeout=0.02,
        write_timeout=1.0,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )
    set_gpio_rx()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser




def choose_write_function(qty: int) -> int:
    """Return 0x06 or 0x10."""
    mode = WRITE_FUNCTION.strip().lower()
    if mode in {"6", "06", "fc06", "single"}:
        if qty != 1:
            raise ValueError("VISIONOPS_MODBUS_WRITE_FUNCTION=06 requires WRITE_CHUNK_SIZE=1 or WRITE_COUNT=1")
        return 0x06
    if mode in {"16", "10", "0x10", "fc16", "multiple"}:
        return 0x10
    # auto
    return 0x06 if qty == 1 else 0x10


def write_chunk_transaction(ser: serial.Serial, start_addr: int, chunk: List[int]) -> Tuple[bool, str, bytes, bytes, int]:
    qty = len(chunk)
    function_code = choose_write_function(qty)

    if function_code == 0x06:
        req = make_write_single_request(TARGET_SLAVE_ID, start_addr, chunk[0])
        expected_len = 8
    else:
        req = make_write_multiple_request(TARGET_SLAVE_ID, start_addr, chunk)
        expected_len = 8

    attempts = max(1, RETRY_COUNT + 1)
    last_msg = ""
    last_resp = b""

    for attempt in range(1, attempts + 1):
        if attempt > 1:
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
            if RETRY_DELAY_MS > 0:
                time.sleep(RETRY_DELAY_MS / 1000.0)

        send_rtu_frame(ser, req)
        resp = read_response(
            ser,
            expected_len=expected_len,
            slave_id=TARGET_SLAVE_ID,
            function_code=function_code,
        )
        ok, msg = validate_write_response(
            resp=resp,
            slave_id=TARGET_SLAVE_ID,
            function_code=function_code,
            start_addr=start_addr,
            qty=qty,
            value=chunk[0] if chunk else 0,
        )

        if ok:
            suffix = "" if attempt == 1 else f" after retry {attempt - 1}/{RETRY_COUNT}"
            return True, f"{msg}{suffix}", req, resp, function_code

        last_msg = msg
        last_resp = resp

        if (not ok) and VERIFY_AFTER_WRITE == 1:
            # Some PLCs execute the write but the local UART misses or corrupts the acknowledgement.
            # Read the same address back and use the read-back as a stronger success criterion.
            read_req = make_read_holding_request(TARGET_SLAVE_ID, start_addr, qty)
            send_rtu_frame(ser, read_req)
            read_resp = read_response(
                ser,
                expected_len=5 + qty * 2,
                slave_id=TARGET_SLAVE_ID,
                function_code=0x03,
            )
            read_ok, read_msg = validate_read_response(read_resp, TARGET_SLAVE_ID, chunk)
            if read_ok:
                suffix = "" if attempt == 1 else f" after retry {attempt - 1}/{RETRY_COUNT}"
                return (
                    True,
                    f"write ack missing/invalid ({msg}), but readback ok{suffix}",
                    req,
                    read_resp,
                    function_code,
                )
            last_msg = f"{msg}; readback failed: {read_msg}"
            last_resp = resp or read_resp

        if REQUIRE_RESPONSE != 1 and not ok:
            return True, f"sent without required ack: {msg}", req, resp, function_code

        if attempt < attempts:
            logging.debug(
                "write retry %d/%d for slave=%d func=0x%02x start=%d qty=%d after: %s raw=%s",
                attempt,
                RETRY_COUNT,
                TARGET_SLAVE_ID,
                function_code,
                start_addr,
                qty,
                last_msg,
                last_resp.hex(" ") if last_resp else "",
            )

    return False, f"{last_msg}; attempts={attempts}", req, last_resp, function_code


def values_digest(values: List[int]) -> str:
    data = bytearray()
    for v in values:
        v = clamp_u16(v)
        data.append((v >> 8) & 0xFF)
        data.append(v & 0xFF)
    return hashlib.sha256(bytes(data)).hexdigest()


def push_values_once(ser: serial.Serial, values: List[int], push_count: int) -> bool:
    ok_all = True
    total = len(values)
    offset = 0
    chunk_index = 0

    while offset < total:
        chunk = values[offset:offset + WRITE_CHUNK_SIZE]
        start_addr = TARGET_ADDRESS + offset
        qty = len(chunk)
        chunk_index += 1

        try:
            ok, msg, req, resp, function_code = write_chunk_transaction(ser, start_addr, chunk)
        except Exception as e:
            logging.exception("push #%d chunk #%d transaction error: %s", push_count, chunk_index, e)
            ok_all = False
            offset += qty
            continue

        if push_count <= 5 or push_count % 100 == 0:
            logging.info(
                "push #%d chunk #%d request: slave=%d func=0x%02x start=%d qty=%d bytes=%d first=%s",
                push_count, chunk_index, TARGET_SLAVE_ID, function_code, start_addr, qty, len(req), req[:20].hex(" "),
            )
        elif LOG_LEVEL == "DEBUG":
            logging.debug("push #%d chunk #%d request: %s", push_count, chunk_index, req.hex(" "))

        if ok:
            if push_count <= 5 or push_count % 100 == 0 or LOG_LEVEL == "DEBUG":
                logging.info(
                    "push #%d chunk #%d response ok: %s response=%s",
                    push_count, chunk_index, msg, resp.hex(" ") if resp else "",
                )
        else:
            logging.warning(
                "push #%d chunk #%d failed: %s response=%s",
                push_count, chunk_index, msg, resp.hex(" ") if resp else "",
            )
            ok_all = False

        offset += qty

    return ok_all


def main() -> int:
    if ENABLE != 1:
        logging.warning("VISIONOPS_MODBUS_MASTER_ENABLE != 1, exit.")
        return 0
    if not os.path.exists(SERIAL_PORT):
        logging.error("serial port not found: %s", SERIAL_PORT)
        logging.error("please check: ls -l /dev/ttyS* /dev/ttyAMA* /dev/ttyUSB*")
        return 2
    if WRITE_COUNT < 1:
        logging.error("VISIONOPS_MODBUS_WRITE_COUNT must be >= 1")
        return 2
    if WRITE_CHUNK_SIZE < 1 or WRITE_CHUNK_SIZE > 123:
        logging.error("VISIONOPS_MODBUS_WRITE_CHUNK_SIZE must be 1..123 for FC16")
        return 2

    setup_gpio()

    logging.info(
        "starting Modbus RTU Master Push v1.3.2: port=%s target_slave=%d baudrate=%d %d%s%d "
        "target_address=%d source_start=%d write_count=%d chunk_size=%d interval_ms=%d "
        "write_function=%s require_response=%d verify_after_write=%d tx_pre_ms=%.3f rx_settle_ms=%.3f "
        "timeout_ms=%.1f retry_count=%d retry_delay_ms=%.1f scan_max_bytes=%d",
        SERIAL_PORT, TARGET_SLAVE_ID, BAUDRATE, BYTESIZE, PARITY, STOPBITS,
        TARGET_ADDRESS, SOURCE_START, WRITE_COUNT, WRITE_CHUNK_SIZE, PUSH_INTERVAL_MS,
        WRITE_FUNCTION, REQUIRE_RESPONSE, VERIFY_AFTER_WRITE, TX_PRE_DELAY_MS, RX_SETTLE_DELAY_MS,
        RESPONSE_TIMEOUT_MS, RETRY_COUNT, RETRY_DELAY_MS, RESPONSE_SCAN_MAX_BYTES,
    )

    ser = open_serial()
    heartbeat = 0
    push_count = 0
    ok_count = 0
    fail_count = 0
    last_digest = ""

    try:
        while True:
            t0 = time.time()
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

                if SOURCE_START < 0 or SOURCE_START + WRITE_COUNT > len(regs):
                    logging.error(
                        "invalid source range: source_start=%d write_count=%d regs_len=%d",
                        SOURCE_START, WRITE_COUNT, len(regs),
                    )
                    time.sleep(max(0.02, PUSH_INTERVAL_MS / 1000.0))
                    continue

                values = regs[SOURCE_START:SOURCE_START + WRITE_COUNT]
                digest = values_digest(values)
                if WRITE_ONLY_ON_CHANGE == 1 and digest == last_digest:
                    time.sleep(max(0.02, PUSH_INTERVAL_MS / 1000.0))
                    continue
                last_digest = digest

                push_count += 1
                ok = push_values_once(ser, values, push_count)
                desc = describe_registers(regs)
                if ok:
                    ok_count += 1
                    if push_count <= 5 or push_count % 50 == 0:
                        logging.info(
                            "push #%d ok: ok_count=%d fail_count=%d target_start=%d count=%d "
                            "magic=%s version=%s heartbeat=%s task_type=%s schema=%s result_count=%s ng=%s",
                            push_count, ok_count, fail_count, TARGET_ADDRESS, WRITE_COUNT,
                            desc.get("magic"), desc.get("protocol_version"), desc.get("heartbeat"),
                            desc.get("task_type"), desc.get("result_schema"), desc.get("result_count"),
                            desc.get("ng_flag"),
                        )
                else:
                    fail_count += 1
                    logging.warning(
                        "push #%d failed: ok_count=%d fail_count=%d target_start=%d count=%d",
                        push_count, ok_count, fail_count, TARGET_ADDRESS, WRITE_COUNT,
                    )

            except Exception as e:
                fail_count += 1
                logging.exception("push loop error: %s", e)

            elapsed = time.time() - t0
            sleep_s = max(0.02, PUSH_INTERVAL_MS / 1000.0 - elapsed)
            time.sleep(sleep_s)

    finally:
        try:
            set_gpio_rx()
        finally:
            ser.close()


if __name__ == "__main__":
    sys.exit(main())
