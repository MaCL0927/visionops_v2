#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Tube Station Modbus TCP Service

上位机写触发寄存器 -> HP60C 抓一张图 -> C++ 单图推理 -> 写回左右纸筒状态。

默认端口: 1502
默认类别映射: stand class_id=0, lying class_id=1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from pymodbus.server import StartTcpServer
except Exception:  # pymodbus 2.x
    from pymodbus.server.sync import StartTcpServer

from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext

try:
    from pymodbus.device import ModbusDeviceIdentification
except Exception:
    ModbusDeviceIdentification = None

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ENV = THIS_DIR / "tube_station.env"

# Register constants
REG_TRIGGER_CMD = 0
REG_TRIGGER_SEQ = 1
REG_STATUS = 2
REG_RESULT_SEQ = 3
REG_LEFT_STATE = 4
REG_RIGHT_STATE = 5
REG_ERROR_CODE = 6
REG_PROCESS_TIME_MS = 7
REG_HEARTBEAT = 8
REG_DETECTION_COUNT = 9
REG_LEFT_CONF = 10
REG_RIGHT_CONF = 11
REG_LEFT_CLASS_ID = 12
REG_RIGHT_CLASS_ID = 13
REG_IMAGE_WIDTH = 14
REG_IMAGE_HEIGHT = 15

STATUS_IDLE = 0
STATUS_BUSY = 1
STATUS_DONE = 2
STATUS_ERROR = 3

STATE_UNKNOWN = 0
STATE_STAND = 1
STATE_LYING = 2
STATE_MISSING_OR_ERROR = 3

INVALID_U16 = 65535

ERR_NONE = 0
ERR_NO_OBJECT = 101
ERR_ONE_OBJECT = 102
ERR_NO_VALID_CLASS = 103
ERR_SNAPSHOT_FAILED = 201
ERR_INFER_FAILED = 202
ERR_JSON_FAILED = 203
ERR_INTERNAL = 301


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


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


def parse_int_set(raw: str, default: Iterable[int]) -> set[int]:
    if not raw:
        return set(default)
    out: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if item == "":
            continue
        try:
            out.add(int(item))
        except ValueError:
            pass
    return out or set(default)


load_env_file(Path(os.environ.get("VISIONOPS_TUBE_ENV", str(DEFAULT_ENV))))

ENABLE = getenv_int("VISIONOPS_TUBE_ENABLE", 1)
HOST = getenv_str("VISIONOPS_TUBE_MODBUS_HOST", "0.0.0.0")
PORT = getenv_int("VISIONOPS_TUBE_MODBUS_PORT", 1502)
UNIT_ID = getenv_int("VISIONOPS_TUBE_MODBUS_UNIT_ID", 1)
ADDRESS_BASE = getenv_int("VISIONOPS_TUBE_ADDRESS_BASE", 0)
REGISTER_COUNT = getenv_int("VISIONOPS_TUBE_REGISTER_COUNT", 64)
RESULT_MODE = getenv_str("VISIONOPS_TUBE_RESULT_MODE", "infer_once").strip().lower()
SNAPSHOT_URL = getenv_str("VISIONOPS_TUBE_SNAPSHOT_URL", "http://127.0.0.1:18181/stream/snapshot.jpg")
INFER_URL = getenv_str("VISIONOPS_TUBE_INFER_URL", "http://127.0.0.1:8090/api/cpp/infer")
LATEST_RESULT_URL = getenv_str("VISIONOPS_TUBE_LATEST_RESULT_URL", "http://127.0.0.1:8090/api/cpp/stream/latest_result")
MIN_CONF = getenv_float("VISIONOPS_TUBE_MIN_CONF", 0.30)
STAND_CLASS_IDS = parse_int_set(getenv_str("VISIONOPS_TUBE_STAND_CLASS_IDS", "0"), [0])
LYING_CLASS_IDS = parse_int_set(getenv_str("VISIONOPS_TUBE_LYING_CLASS_IDS", "1"), [1])
SNAPSHOT_TIMEOUT_S = getenv_int("VISIONOPS_TUBE_SNAPSHOT_TIMEOUT_MS", 1500) / 1000.0
INFER_TIMEOUT_S = getenv_int("VISIONOPS_TUBE_INFER_TIMEOUT_MS", 5000) / 1000.0
TRIGGER_TIMEOUT_S = getenv_int("VISIONOPS_TUBE_TRIGGER_TIMEOUT_MS", 8000) / 1000.0
POLL_INTERVAL_S = max(0.02, getenv_int("VISIONOPS_TUBE_POLL_INTERVAL_MS", 50) / 1000.0)
LOG_LEVEL = getenv_str("VISIONOPS_TUBE_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def u16(v: int) -> int:
    return int(v) & 0xFFFF


def now_ms() -> int:
    return int(time.time() * 1000)


def create_context() -> ModbusServerContext:
    block_count = max(REGISTER_COUNT, ADDRESS_BASE + REGISTER_COUNT + 10)
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


def http_get_bytes(url: str, timeout_s: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "VisionOps-TubeStation/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        code = getattr(resp, "status", 200)
        if code < 200 or code >= 300:
            raise RuntimeError(f"GET {url} HTTP {code}")
        return resp.read()


def http_get_json(url: str, timeout_s: float) -> Dict[str, Any]:
    raw = http_get_bytes(url, timeout_s).decode("utf-8", errors="replace")
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON from {url}: {exc}: {raw[:200]}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON from {url} is not object")
    return data


def post_multipart_image(url: str, image_bytes: bytes, filename: str = "hp60c_trigger.jpg", timeout_s: float = 5.0) -> Dict[str, Any]:
    boundary = "----VisionOpsTubeStationBoundary" + str(now_ms())
    head = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = head + image_bytes + tail
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "Accept": "application/json",
            "User-Agent": "VisionOps-TubeStation/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc.reason)
        raise RuntimeError(f"POST {url} HTTP {exc.code}: {detail[:500]}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"infer returned non JSON: {exc}") from exc


def find_predictions(obj: Any) -> List[Dict[str, Any]]:
    """Recursively find the first meaningful predictions list."""
    if isinstance(obj, dict):
        preds = obj.get("predictions")
        if isinstance(preds, list):
            return [p for p in preds if isinstance(p, dict)]
        for key in ("raw", "data", "result", "detection"):
            if key in obj:
                found = find_predictions(obj[key])
                if found:
                    return found
        # fallback: search all values
        for value in obj.values():
            found = find_predictions(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_predictions(value)
            if found:
                return found
    return []


def image_size_from_payload(payload: Dict[str, Any]) -> Tuple[int, int]:
    for obj in (payload, payload.get("raw") if isinstance(payload.get("raw"), dict) else None):
        if not isinstance(obj, dict):
            continue
        w = obj.get("image_width") or obj.get("width")
        h = obj.get("image_height") or obj.get("height")
        try:
            if w and h:
                return int(w), int(h)
        except Exception:
            pass
    return 0, 0


def pred_class_id(pred: Dict[str, Any]) -> Optional[int]:
    raw = pred.get("class_id", pred.get("cls", pred.get("label_id")))
    try:
        return int(raw)
    except Exception:
        return None


def pred_conf(pred: Dict[str, Any]) -> float:
    raw = pred.get("confidence", pred.get("score", pred.get("conf", 0.0)))
    try:
        return float(raw)
    except Exception:
        return 0.0


def pred_center_x(pred: Dict[str, Any]) -> float:
    if isinstance(pred.get("center"), list) and pred["center"]:
        try:
            return float(pred["center"][0])
        except Exception:
            pass
    for key in ("center_x", "cx"):
        if key in pred:
            try:
                return float(pred[key])
            except Exception:
                pass
    bbox = pred.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            x1, _, x2, _ = [float(x) for x in bbox[:4]]
            return (x1 + x2) / 2.0
        except Exception:
            pass
    return 0.0


def class_to_state(class_id: Optional[int]) -> int:
    if class_id in STAND_CLASS_IDS:
        return STATE_STAND
    if class_id in LYING_CLASS_IDS:
        return STATE_LYING
    return STATE_UNKNOWN


def select_left_right(payload: Dict[str, Any]) -> Dict[str, Any]:
    preds = find_predictions(payload)
    valid: List[Dict[str, Any]] = []
    for p in preds:
        cid = pred_class_id(p)
        state = class_to_state(cid)
        conf = pred_conf(p)
        if state == STATE_UNKNOWN:
            continue
        if conf < MIN_CONF:
            continue
        q = dict(p)
        q["_class_id"] = cid
        q["_state"] = state
        q["_conf"] = conf
        q["_cx"] = pred_center_x(p)
        valid.append(q)

    # 多于两个时先取置信度最高的两个，再按 x 排序区分左右。
    selected = sorted(valid, key=lambda x: x.get("_conf", 0.0), reverse=True)[:2]
    selected = sorted(selected, key=lambda x: x.get("_cx", 0.0))

    width, height = image_size_from_payload(payload)
    result = {
        "left_state": STATE_MISSING_OR_ERROR,
        "right_state": STATE_MISSING_OR_ERROR,
        "left_conf": 0,
        "right_conf": 0,
        "left_class_id": INVALID_U16,
        "right_class_id": INVALID_U16,
        "count": len(valid),
        "image_width": width,
        "image_height": height,
        "error_code": ERR_NONE,
        "raw_predictions_count": len(preds),
    }

    if len(valid) <= 0:
        result["error_code"] = ERR_NO_OBJECT
        return result

    if len(selected) == 1:
        only = selected[0]
        # 如果知道图像宽度，可以按半幅判断左右；否则保守写左侧。
        is_left = True
        if width > 0:
            is_left = float(only.get("_cx", 0.0)) < width / 2.0
        side = "left" if is_left else "right"
        result[f"{side}_state"] = only["_state"]
        result[f"{side}_conf"] = int(round(only["_conf"] * 10000))
        result[f"{side}_class_id"] = int(only["_class_id"])
        result["error_code"] = ERR_ONE_OBJECT
        return result

    left, right = selected[0], selected[1]
    result.update({
        "left_state": int(left["_state"]),
        "right_state": int(right["_state"]),
        "left_conf": int(round(float(left["_conf"]) * 10000)),
        "right_conf": int(round(float(right["_conf"]) * 10000)),
        "left_class_id": int(left["_class_id"]),
        "right_class_id": int(right["_class_id"]),
    })
    return result


def run_infer_once() -> Dict[str, Any]:
    if RESULT_MODE == "latest_result":
        logging.info("fetch latest result: %s", LATEST_RESULT_URL)
        return http_get_json(LATEST_RESULT_URL, timeout_s=INFER_TIMEOUT_S)

    logging.info("fetch snapshot: %s", SNAPSHOT_URL)
    try:
        image_bytes = http_get_bytes(SNAPSHOT_URL, timeout_s=SNAPSHOT_TIMEOUT_S)
    except Exception as exc:
        raise RuntimeError(f"snapshot failed: {exc}") from exc
    if not image_bytes:
        raise RuntimeError("snapshot is empty")

    logging.info("post infer: %s bytes=%d", INFER_URL, len(image_bytes))
    return post_multipart_image(INFER_URL, image_bytes, timeout_s=INFER_TIMEOUT_S)


class TubeStationService:
    def __init__(self, context: ModbusServerContext) -> None:
        self.context = context
        self.lock = threading.Lock()
        self.busy = False
        self.last_seq = -1
        self.heartbeat = 0

    def init_registers(self) -> None:
        values = [0] * REGISTER_COUNT
        values[REG_STATUS] = STATUS_IDLE
        values[REG_LEFT_STATE] = STATE_UNKNOWN
        values[REG_RIGHT_STATE] = STATE_UNKNOWN
        values[REG_ERROR_CODE] = ERR_NONE
        values[REG_LEFT_CLASS_ID] = INVALID_U16
        values[REG_RIGHT_CLASS_ID] = INVALID_U16
        set_regs(self.context, 0, values)

    def heartbeat_loop(self) -> None:
        while True:
            self.heartbeat = (self.heartbeat + 1) & 0xFFFF
            set_reg(self.context, REG_HEARTBEAT, self.heartbeat)
            time.sleep(0.5)

    def poll_loop(self) -> None:
        logging.info("Tube station trigger poll started")
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
        set_regs(self.context, REG_STATUS, [STATUS_BUSY])
        set_regs(self.context, REG_ERROR_CODE, [ERR_NONE])
        set_regs(self.context, REG_LEFT_STATE, [STATE_UNKNOWN, STATE_UNKNOWN])

        try:
            payload = run_infer_once()
            decision = select_left_right(payload)
            elapsed_ms = int(round((time.time() - start) * 1000))
            status = STATUS_DONE if decision.get("error_code", 0) in (ERR_NONE, ERR_ONE_OBJECT) else STATUS_ERROR
            # 如果只检测到一个纸筒，状态可以标 error，也可以标 done+error_code。这里采用 error，便于上位机处理。
            if decision.get("error_code") == ERR_ONE_OBJECT:
                status = STATUS_ERROR

            set_regs(
                self.context,
                REG_STATUS,
                [
                    status,
                    seq,
                    decision["left_state"],
                    decision["right_state"],
                    decision.get("error_code", ERR_NONE),
                    elapsed_ms,
                    self.heartbeat,
                    decision.get("count", 0),
                    decision.get("left_conf", 0),
                    decision.get("right_conf", 0),
                    decision.get("left_class_id", INVALID_U16),
                    decision.get("right_class_id", INVALID_U16),
                    decision.get("image_width", 0),
                    decision.get("image_height", 0),
                ],
            )
            logging.info(
                "trigger done: seq=%d status=%d left=%d right=%d err=%d time=%dms count=%d",
                seq,
                status,
                decision["left_state"],
                decision["right_state"],
                decision.get("error_code", 0),
                elapsed_ms,
                decision.get("count", 0),
            )
        except Exception as exc:
            elapsed_ms = int(round((time.time() - start) * 1000))
            msg = str(exc)
            if "snapshot" in msg.lower():
                err = ERR_SNAPSHOT_FAILED
            elif "json" in msg.lower():
                err = ERR_JSON_FAILED
            elif "infer" in msg.lower() or "post" in msg.lower():
                err = ERR_INFER_FAILED
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
                    STATE_MISSING_OR_ERROR,
                    STATE_MISSING_OR_ERROR,
                    err,
                    elapsed_ms,
                ],
            )
        finally:
            with self.lock:
                self.busy = False


def main() -> int:
    if ENABLE != 1:
        logging.warning("VISIONOPS_TUBE_ENABLE != 1, exit")
        return 0

    context = create_context()
    service = TubeStationService(context)
    service.init_registers()

    threading.Thread(target=service.heartbeat_loop, daemon=True).start()
    threading.Thread(target=service.poll_loop, daemon=True).start()

    identity = None
    if ModbusDeviceIdentification is not None:
        identity = ModbusDeviceIdentification()
        identity.VendorName = "VisionOps"
        identity.ProductCode = "TUBE"
        identity.VendorUrl = "https://github.com/MaCL0927/visionops_v2"
        identity.ProductName = "VisionOps Tube Station Modbus TCP"
        identity.ModelName = "LB3576 HP60C Tube Station"
        identity.MajorMinorRevision = "1.0"

    logging.info(
        "starting tube station: host=%s port=%d unit_id=%d address_base=%d mode=%s stand=%s lying=%s",
        HOST,
        PORT,
        UNIT_ID,
        ADDRESS_BASE,
        RESULT_MODE,
        sorted(STAND_CLASS_IDS),
        sorted(LYING_CLASS_IDS),
    )
    logging.info("snapshot_url=%s", SNAPSHOT_URL)
    logging.info("infer_url=%s", INFER_URL)
    StartTcpServer(context=context, identity=identity, address=(HOST, PORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
