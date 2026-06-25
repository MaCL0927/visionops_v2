#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from pymodbus.client import ModbusTcpClient
except Exception:
    from pymodbus.client.sync import ModbusTcpClient


DEFAULT_OUT_DIR = "/tmp/vision_robot_protocol_latest/coordinate_check"
_running = True


def handle_signal(signum, frame):
    global _running
    _running = False


def now_str():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def s16(v):
    v = int(v) & 0xFFFF
    return v - 65536 if v >= 32768 else v


def read_holding_registers_compat(client, address, count, unit_id):
    last_exc = None
    for kwargs in (
        {"address": address, "count": count, "slave": unit_id},
        {"address": address, "count": count, "unit": unit_id},
        {"address": address, "count": count},
    ):
        try:
            rr = client.read_holding_registers(**kwargs)
            if rr is None:
                continue
            if hasattr(rr, "isError") and rr.isError():
                raise RuntimeError(str(rr))
            return [int(x) & 0xFFFF for x in rr.registers]
        except Exception as exc:
            last_exc = exc
    raise RuntimeError("read_holding_registers failed: %r" % (last_exc,))


def invert_2x2(a00, a01, a10, a11):
    det = a00 * a11 - a01 * a10
    if abs(det) < 1e-12:
        raise RuntimeError("affine matrix A is singular, cannot invert")
    return (
        a11 / det,
        -a01 / det,
        -a10 / det,
        a00 / det,
    )


def robot_to_image(x_robot, y_robot, args):
    """
    正向:
      x_robot = A00*x_camera + A01*y_camera + B0
      y_robot = A10*x_camera + A11*y_camera + B1

    反向:
      [x_camera, y_camera] = inv(A) * ([x_robot, y_robot] - b)
    """
    ia00, ia01, ia10, ia11 = invert_2x2(args.a00, args.a01, args.a10, args.a11)

    dx = float(x_robot) - args.b0
    dy = float(y_robot) - args.b1

    x_camera = ia00 * dx + ia01 * dy
    y_camera = ia10 * dx + ia11 * dy

    return x_camera, y_camera


def build_points(regs_raw, base_reg, args):
    regs_signed = [s16(v) for v in regs_raw]
    points = []

    for idx in range(40):
        reg_x = base_reg + 2 * idx
        reg_y = base_reg + 2 * idx + 1

        x_robot = regs_signed[2 * idx]
        y_robot = regs_signed[2 * idx + 1]

        x_img, y_img = robot_to_image(x_robot, y_robot, args)

        if args.round_image:
            x_img_out = int(round(x_img))
            y_img_out = int(round(y_img))
        else:
            x_img_out = round(float(x_img), 3)
            y_img_out = round(float(y_img), 3)

        points.append({
            "index": idx + 1,
            "reg_x": reg_x,
            "reg_y": reg_y,
            "robot_x": x_robot,
            "robot_y": y_robot,
            "image_x": x_img_out,
            "image_y": y_img_out,
            "image_x_float": float(x_img),
            "image_y_float": float(y_img),
            "raw_x_uint16": int(regs_raw[2 * idx]) & 0xFFFF,
            "raw_y_uint16": int(regs_raw[2 * idx + 1]) & 0xFFFF,
        })

    return regs_signed, points


def build_matrix(points, matrix_rows, matrix_cols):
    matrix = []
    idx = 0
    for r in range(matrix_rows):
        row = []
        for c in range(matrix_cols):
            row.append(points[idx])
            idx += 1
        matrix.append(row)
    return matrix


def format_text_snapshot(save_count, timestamp, regs_raw, args):
    regs_signed, points = build_points(regs_raw, args.base_reg, args)
    matrix = build_matrix(points, args.matrix_rows, args.matrix_cols)

    lines = []
    lines.append("=" * 100)
    lines.append("save_count: %06d" % save_count)
    lines.append("timestamp: %s" % timestamp)
    lines.append("source_register_range: %d-%d" % (args.base_reg, args.base_reg + len(regs_raw) - 1))
    lines.append("source_register_value_type: signed int16 robot coordinate")
    lines.append("saved_coordinate_frame: robot + image")
    lines.append("matrix_shape: %d x %d points" % (args.matrix_rows, args.matrix_cols))
    lines.append("")
    lines.append("robot_to_image: inverse of")
    lines.append("  x_robot = %.8f*x_camera + %.8f*y_camera + %.8f" % (args.a00, args.a01, args.b0))
    lines.append("  y_robot = %.8f*x_camera + %.8f*y_camera + %.8f" % (args.a10, args.a11, args.b1))
    lines.append("")

    lines.append("[机器人坐标矩阵 x,y]")
    for r, row in enumerate(matrix, start=1):
        pairs = []
        for p in row:
            pairs.append("(%6d,%6d)" % (p["robot_x"], p["robot_y"]))
        lines.append("row_%02d: %s" % (r, "  ".join(pairs)))
    lines.append("")

    lines.append("[图像坐标矩阵 x,y]")
    for r, row in enumerate(matrix, start=1):
        pairs = []
        for p in row:
            if args.round_image:
                pairs.append("(%5d,%5d)" % (p["image_x"], p["image_y"]))
            else:
                pairs.append("(%9.3f,%9.3f)" % (p["image_x"], p["image_y"]))
        lines.append("row_%02d: %s" % (r, "  ".join(pairs)))
    lines.append("")

    lines.append("[机器人坐标 -> 图像坐标 对照矩阵]")
    for r, row in enumerate(matrix, start=1):
        pairs = []
        for p in row:
            if args.round_image:
                pairs.append("R(%6d,%6d)->I(%5d,%5d)" % (
                    p["robot_x"],
                    p["robot_y"],
                    p["image_x"],
                    p["image_y"],
                ))
            else:
                pairs.append("R(%6d,%6d)->I(%9.3f,%9.3f)" % (
                    p["robot_x"],
                    p["robot_y"],
                    p["image_x"],
                    p["image_y"],
                ))
        lines.append("row_%02d: %s" % (r, "  ".join(pairs)))
    lines.append("")

    return "\n".join(lines)

def append_text_log(path, text):
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def load_state(state_path):
    if not state_path.exists():
        return {"save_count": 0, "last_regs_raw": None}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"save_count": 0, "last_regs_raw": None}


def save_state(state_path, save_count, regs_raw):
    state = {
        "save_count": int(save_count),
        "last_regs_raw": [int(x) & 0xFFFF for x in regs_raw],
        "updated_at": now_str(),
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Monitor coordinate registers 20-99 and save image-coordinate update logs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5046)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--base-reg", type=int, default=20)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)

    # 你当前贴出的寄存器排列：每行 5 个点，共 8 行。
    parser.add_argument("--matrix-rows", type=int, default=8)
    parser.add_argument("--matrix-cols", type=int, default=5)

    # Robot coordinate affine transform.
    parser.add_argument("--a00", type=float, default=0.02143055)
    parser.add_argument("--a01", type=float, default=-1.49495102)
    parser.add_argument("--a10", type=float, default=-1.47967273)
    parser.add_argument("--a11", type=float, default=-0.00292085)
    parser.add_argument("--b0", type=float, default=946.29821487)
    parser.add_argument("--b1", type=float, default=994.16131507)

    parser.add_argument("--round-image", action="store_true", default=True, help="图像坐标四舍五入为整数")
    parser.add_argument("--float-image", dest="round_image", action="store_false", help="图像坐标保存为小数")
    parser.add_argument("--no-save-on-start", action="store_true")
    args = parser.parse_args()

    if args.count != 80:
        raise SystemExit("This logger expects 80 registers for 40 points.")
    if args.matrix_rows * args.matrix_cols != 40:
        raise SystemExit("matrix_rows * matrix_cols must be 40")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text_log_path = out_dir / "coord_image_update_log.txt"
    jsonl_log_path = out_dir / "coord_image_update_log.jsonl"
    latest_json_path = out_dir / "coord_image_latest.json"
    state_path = out_dir / "coord_image_log_state.json"

    state = load_state(state_path)
    save_count = int(state.get("save_count") or 0)
    last_regs_raw = state.get("last_regs_raw")
    if isinstance(last_regs_raw, list):
        last_regs_raw = [int(x) & 0xFFFF for x in last_regs_raw]
    else:
        last_regs_raw = None

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("[INFO] monitor Modbus registers %d-%d" % (args.base_reg, args.base_reg + args.count - 1))
    print("[INFO] source frame: robot coordinates in signed int16 registers")
    print("[INFO] saved frame: image coordinates after inverse affine transform")
    print("[INFO] target: %s:%d unit_id=%d" % (args.host, args.port, args.unit_id))
    print("[INFO] out_dir:", out_dir)
    print("[INFO] text_log:", text_log_path)
    print("[INFO] latest_json:", latest_json_path)
    print("[INFO] matrix_shape: %dx%d points" % (args.matrix_rows, args.matrix_cols))

    first_loop = True
    client = None

    while _running:
        try:
            if client is None:
                client = ModbusTcpClient(args.host, port=args.port)
                ok = client.connect()
                if ok is False:
                    raise RuntimeError("connect failed")
                print("[INFO] connected")

            regs_raw = read_holding_registers_compat(
                client,
                address=args.base_reg,
                count=args.count,
                unit_id=args.unit_id,
            )

            changed = regs_raw != last_regs_raw
            should_save = changed

            if first_loop and args.no_save_on_start and last_regs_raw is None:
                should_save = False
                last_regs_raw = regs_raw

            if should_save:
                save_count += 1
                timestamp = now_str()

                regs_signed, points = build_points(regs_raw, args.base_reg, args)
                matrix = build_matrix(points, args.matrix_rows, args.matrix_cols)

                record = {
                    "save_count": save_count,
                    "timestamp": timestamp,
                    "source_register_range": [args.base_reg, args.base_reg + args.count - 1],
                    "source_register_value_type": "signed int16 robot coordinate",
                    "saved_coordinate_frame": "image",
                    "matrix_shape": [args.matrix_rows, args.matrix_cols],
                    "affine_robot_from_image": {
                        "a00": args.a00,
                        "a01": args.a01,
                        "a10": args.a10,
                        "a11": args.a11,
                        "b0": args.b0,
                        "b1": args.b1,
                    },
                    "registers_raw_uint16": regs_raw,
                    "registers_robot_signed_int16": regs_signed,
                    "points": points,
                    "image_matrix": [
                        [
                            {
                                "index": p["index"],
                                "image_x": p["image_x"],
                                "image_y": p["image_y"],
                                "robot_x": p["robot_x"],
                                "robot_y": p["robot_y"],
                                "reg_x": p["reg_x"],
                                "reg_y": p["reg_y"],
                            }
                            for p in row
                        ]
                        for row in matrix
                    ],
                }

                text = format_text_snapshot(save_count, timestamp, regs_raw, args)

                append_text_log(text_log_path, text)
                append_jsonl(jsonl_log_path, record)
                latest_json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                save_state(state_path, save_count, regs_raw)

                last_regs_raw = regs_raw
                print("[SAVE] count=%06d timestamp=%s" % (save_count, timestamp))

            first_loop = False
            time.sleep(args.interval)

        except Exception as exc:
            print("[WARN] %s" % exc)
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass
            client = None
            time.sleep(1.0)

    try:
        if client is not None:
            client.close()
    except Exception:
        pass

    print("[INFO] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
