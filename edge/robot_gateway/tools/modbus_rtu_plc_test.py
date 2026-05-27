#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Modbus RTU PLC Test Tool for LPR3576 / LB3576

功能：
- 3576 作为 Modbus RTU Master
- PLC 作为 Modbus RTU Slave
- 支持：
  1) 读 Holding Registers，功能码 03
  2) 写单个 Holding Register，功能码 06
  3) 写后读回验证

注意：
- 使用 /dev/ttyS5 时，请先停止其他占用该串口的服务。
- LPR3576 / LB3576 RS485 半双工方向由 GPIO136 控制：
  GPIO136=1 发送
  GPIO136=0 接收
"""

import argparse
import os
import sys
import time
import termios

try:
    import serial
except Exception:
    print("[ERROR] pyserial not found. Try: /opt/visionops/venv/bin/python -m pip install pyserial")
    raise


def crc16_modbus(data: bytes) -> int:
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
    crc = crc16_modbus(data)
    return data + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def valid_crc(frame: bytes) -> bool:
    if len(frame) < 4:
        return False
    got = frame[-2] | (frame[-1] << 8)
    calc = crc16_modbus(frame[:-2])
    return got == calc


class GpioDir:
    def __init__(self, gpio_num: int = 136, tx_value: str = "1", rx_value: str = "0", enable: bool = True):
        self.gpio_num = gpio_num
        self.tx_value = tx_value
        self.rx_value = rx_value
        self.enable = enable
        self.base = f"/sys/class/gpio/gpio{gpio_num}"
        self.value_path = os.path.join(self.base, "value")

    def setup(self):
        if not self.enable:
            return
        if not os.path.exists(self.base):
            try:
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(self.gpio_num))
            except OSError:
                pass

        if not os.path.exists(self.base):
            raise RuntimeError(f"GPIO{self.gpio_num} export failed")

        with open(os.path.join(self.base, "direction"), "w") as f:
            f.write("out")
        self.rx()

    def write_value(self, value: str):
        if not self.enable:
            return
        with open(self.value_path, "w") as f:
            f.write(str(value))

    def tx(self):
        self.write_value(self.tx_value)

    def rx(self):
        self.write_value(self.rx_value)


def make_read_request(slave: int, addr: int, count: int) -> bytes:
    if count < 1 or count > 125:
        raise ValueError("read count must be 1..125")
    p = bytes([
        slave & 0xFF,
        0x03,
        (addr >> 8) & 0xFF,
        addr & 0xFF,
        (count >> 8) & 0xFF,
        count & 0xFF,
    ])
    return append_crc(p)


def make_write_single_request(slave: int, addr: int, value: int) -> bytes:
    value &= 0xFFFF
    p = bytes([
        slave & 0xFF,
        0x06,
        (addr >> 8) & 0xFF,
        addr & 0xFF,
        (value >> 8) & 0xFF,
        value & 0xFF,
    ])
    return append_crc(p)


def scan_response(buf: bytes, slave: int, func: int, expected_len: int = None):
    """
    从字节流中扫描合法 RTU 响应帧。
    允许前面有噪声/残留字节。
    """
    for i in range(len(buf)):
        if buf[i] != (slave & 0xFF):
            continue
        if i + 2 > len(buf):
            continue

        f = buf[i + 1]

        # 异常响应：slave, func|0x80, exception, crc_lo, crc_hi
        if f == (func | 0x80):
            if i + 5 <= len(buf):
                frame = buf[i:i + 5]
                if valid_crc(frame):
                    return frame

        # 正常响应
        if f != func:
            continue

        if func == 0x03:
            if i + 3 <= len(buf):
                byte_count = buf[i + 2]
                total_len = 3 + byte_count + 2
                if i + total_len <= len(buf):
                    frame = buf[i:i + total_len]
                    if valid_crc(frame):
                        return frame

        elif func == 0x06:
            total_len = 8
            if i + total_len <= len(buf):
                frame = buf[i:i + total_len]
                if valid_crc(frame):
                    return frame

        elif expected_len:
            if i + expected_len <= len(buf):
                frame = buf[i:i + expected_len]
                if valid_crc(frame):
                    return frame

    return None


def send_and_recv(ser, gpio: GpioDir, req: bytes, slave: int, func: int, timeout_ms: int, rx_settle_ms: float = 0):
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    gpio.tx()
    time.sleep(0.001)

    ser.write(req)
    ser.flush()

    try:
        termios.tcdrain(ser.fileno())
    except Exception:
        # 兜底等待发送时间
        char_time = 11.0 / float(ser.baudrate)
        time.sleep(len(req) * char_time)

    gpio.rx()

    if rx_settle_ms > 0:
        time.sleep(rx_settle_ms / 1000.0)

    deadline = time.time() + timeout_ms / 1000.0
    raw = bytearray()

    while time.time() < deadline:
        chunk = ser.read(256)
        if chunk:
            raw.extend(chunk)
            frame = scan_response(bytes(raw), slave, func)
            if frame:
                return frame, bytes(raw)
        else:
            time.sleep(0.001)

    return None, bytes(raw)


def parse_read_response(frame: bytes, count: int):
    if len(frame) < 5:
        raise ValueError("short read response")
    if frame[1] & 0x80:
        raise RuntimeError(f"exception response: func=0x{frame[1]:02x}, code=0x{frame[2]:02x}")
    if frame[1] != 0x03:
        raise ValueError(f"unexpected function: 0x{frame[1]:02x}")
    byte_count = frame[2]
    if byte_count != count * 2:
        raise ValueError(f"unexpected byte_count={byte_count}, expected={count * 2}")
    values = []
    pos = 3
    for _ in range(count):
        values.append((frame[pos] << 8) | frame[pos + 1])
        pos += 2
    return values


def main():
    ap = argparse.ArgumentParser(description="Simple Modbus RTU PLC read/write test tool for LPR3576")
    ap.add_argument("--port", default="/dev/ttyS5")
    ap.add_argument("--slave", type=int, default=1)
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--parity", default="N", choices=["N", "E", "O"])
    ap.add_argument("--bytesize", type=int, default=8)
    ap.add_argument("--stopbits", type=int, default=1)
    ap.add_argument("--gpio", type=int, default=136)
    ap.add_argument("--timeout-ms", type=int, default=1500)

    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("read", help="read holding registers, FC03")
    r.add_argument("--addr", type=int, required=True, help="protocol address, e.g. 4096")
    r.add_argument("--count", type=int, default=1)

    w = sub.add_parser("write", help="write single register, FC06")
    w.add_argument("--addr", type=int, required=True, help="protocol address, e.g. 4096")
    w.add_argument("--value", type=int, required=True, help="uint16 value")

    wr = sub.add_parser("write-read", help="write single register then read back")
    wr.add_argument("--addr", type=int, required=True)
    wr.add_argument("--value", type=int, required=True)

    args = ap.parse_args()

    gpio = GpioDir(args.gpio)
    gpio.setup()

    parity_map = {
        "N": serial.PARITY_NONE,
        "E": serial.PARITY_EVEN,
        "O": serial.PARITY_ODD,
    }

    ser = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        bytesize=args.bytesize,
        parity=parity_map[args.parity],
        stopbits=args.stopbits,
        timeout=0.02,
        write_timeout=1.0,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )

    try:
        gpio.rx()
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        if args.cmd == "read":
            req = make_read_request(args.slave, args.addr, args.count)
            print(f"[TX] {req.hex(' ')}")
            frame, raw = send_and_recv(ser, gpio, req, args.slave, 0x03, args.timeout_ms)
            print(f"[RAW RX] {raw.hex(' ') if raw else '<empty>'}")
            if not frame:
                print("[ERROR] no valid response")
                return 2
            print(f"[RX] {frame.hex(' ')}")
            values = parse_read_response(frame, args.count)
            for i, v in enumerate(values):
                print(f"[{args.addr + i}] = {v}")

        elif args.cmd == "write":
            req = make_write_single_request(args.slave, args.addr, args.value)
            print(f"[TX] {req.hex(' ')}")
            frame, raw = send_and_recv(ser, gpio, req, args.slave, 0x06, args.timeout_ms)
            print(f"[RAW RX] {raw.hex(' ') if raw else '<empty>'}")
            if not frame:
                print("[ERROR] no valid response")
                return 2
            print(f"[RX] {frame.hex(' ')}")
            print("[OK] write ack received")

        elif args.cmd == "write-read":
            req = make_write_single_request(args.slave, args.addr, args.value)
            print(f"[WRITE TX] {req.hex(' ')}")
            frame, raw = send_and_recv(ser, gpio, req, args.slave, 0x06, args.timeout_ms)
            print(f"[WRITE RAW RX] {raw.hex(' ') if raw else '<empty>'}")
            if frame:
                print(f"[WRITE RX] {frame.hex(' ')}")
                print("[OK] write ack received")
            else:
                print("[WARN] write ack missing/invalid, continue readback")

            time.sleep(0.05)

            req2 = make_read_request(args.slave, args.addr, 1)
            print(f"[READ TX] {req2.hex(' ')}")
            frame2, raw2 = send_and_recv(ser, gpio, req2, args.slave, 0x03, args.timeout_ms)
            print(f"[READ RAW RX] {raw2.hex(' ') if raw2 else '<empty>'}")
            if not frame2:
                print("[ERROR] readback no valid response")
                return 2
            print(f"[READ RX] {frame2.hex(' ')}")
            values = parse_read_response(frame2, 1)
            print(f"[{args.addr}] = {values[0]}")
            if values[0] == (args.value & 0xFFFF):
                print("[OK] readback matched")
            else:
                print(f"[ERROR] readback mismatch: expected {args.value & 0xFFFF}, got {values[0]}")
                return 3

        return 0

    finally:
        gpio.rx()
        ser.close()


if __name__ == "__main__":
    sys.exit(main())
