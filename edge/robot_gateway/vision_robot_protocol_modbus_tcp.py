#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps unified Robot/PLC Modbus-TCP protocol service.

Protocol from 与视觉通讯内容定义表.xlsx:
  Vision -> Robot/PLC holding registers:
    0   heartbeat, +1 every 0.5s, reset to 0 after 1000
    1   carton partition check result, 0=no trigger/no result, 1=OK, 2=NG/ERROR
    2   product placement / carton tube check result, 0=no trigger/no result, 1=OK, 2=NG/ERROR
    3   coordinate recognition result, 0=no trigger/no result, 1=OK, 2=NG/ERROR
    20~99 partition cell center coordinates: slot1_x, slot1_y ... slot40_x, slot40_y

  Robot/PLC -> Vision holding registers:
    100 heartbeat from Robot/PLC (read-only for Vision)
    101 carton partition check trigger: 0=idle, 1=trigger
    102 product placement / carton tube check trigger: 0=idle, 1=left 4 cols, 2=right 4 cols, 3=all cols
    103 coordinate recognition trigger: 0=idle, 1=trigger

The robot side holds trigger=1 until it receives the result, then writes trigger=0.
When trigger is 0, result registers 1~3 are cleared to 0. Coordinates 20~99 are not cleared.
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

try:
    from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
except Exception:  # pymodbus newer versions
    from pymodbus.datastore import ModbusSequentialDataBlock, ModbusDeviceContext as ModbusSlaveContext, ModbusServerContext

try:
    from pymodbus.device import ModbusDeviceIdentification
except Exception:
    ModbusDeviceIdentification = None

THIS_DIR = Path(__file__).resolve().parent
PARTITION_DIR = THIS_DIR / "carton_partition_check"
TUBE_DIR = THIS_DIR / "carton_tube_check"

# Import task cores from their existing folders.
sys.path.insert(0, str(PARTITION_DIR))
import debug_partition_check_once as partition_core  # type: ignore  # noqa: E402
sys.path.insert(0, str(TUBE_DIR))
import debug_depth_check_once as tube_core  # type: ignore  # noqa: E402

DEFAULT_ENV = THIS_DIR / "vision_robot_protocol.env"
DEFAULT_PARTITION_ENV = PARTITION_DIR / "partition_check.env"
DEFAULT_TUBE_ENV = TUBE_DIR / "carton_tube_check.env"

# Load env files.
partition_core.load_env_file(Path(os.environ.get("VISIONOPS_PARTITION_ENV", str(DEFAULT_PARTITION_ENV))))
tube_core.load_env_file(Path(os.environ.get("VISIONOPS_CARTON_TUBE_ENV", str(DEFAULT_TUBE_ENV))))
partition_core.load_env_file(Path(os.environ.get("VISIONOPS_ROBOT_PROTOCOL_ENV", str(DEFAULT_ENV))))

# -----------------------------
# Protocol holding register map
# -----------------------------
REG_VISION_HEARTBEAT = 0
REG_PARTITION_RESULT = 1
REG_PRODUCT_RESULT = 2
REG_COORD_RESULT = 3
REG_COORD_BASE = 20              # slot1_x, slot1_y ... slot40_x, slot40_y
REG_ROBOT_HEARTBEAT = 100
REG_TRIGGER_PARTITION = 101
REG_TRIGGER_PRODUCT = 102
REG_TRIGGER_COORD = 103

RESULT_NONE = 0
RESULT_OK = 1
RESULT_NG = 2

TUBE_TRIGGER_LEFT = 1
TUBE_TRIGGER_RIGHT = 2
TUBE_TRIGGER_ALL = 3
TUBE_TRIGGER_TO_REGION = {
    TUBE_TRIGGER_LEFT: "left",
    TUBE_TRIGGER_RIGHT: "right",
    TUBE_TRIGGER_ALL: "all",
}

ENABLE = partition_core.getenv_int("VISIONOPS_ROBOT_PROTOCOL_ENABLE", 1)
HOST = partition_core.getenv_str("VISIONOPS_ROBOT_PROTOCOL_MODBUS_HOST", "0.0.0.0")
PORT = partition_core.getenv_int("VISIONOPS_ROBOT_PROTOCOL_MODBUS_PORT", 5045)
UNIT_ID = partition_core.getenv_int("VISIONOPS_ROBOT_PROTOCOL_MODBUS_UNIT_ID", 1)
SINGLE_SLAVE = partition_core.getenv_int("VISIONOPS_ROBOT_PROTOCOL_MODBUS_SINGLE_SLAVE", 1)
ADDRESS_BASE = partition_core.getenv_int("VISIONOPS_ROBOT_PROTOCOL_ADDRESS_BASE", 0)
REGISTER_COUNT = partition_core.getenv_int("VISIONOPS_ROBOT_PROTOCOL_REGISTER_COUNT", 200)
POLL_INTERVAL_S = max(0.02, partition_core.getenv_int("VISIONOPS_ROBOT_PROTOCOL_POLL_INTERVAL_MS", 50) / 1000.0)
LOG_LEVEL = partition_core.getenv_str("VISIONOPS_ROBOT_PROTOCOL_LOG_LEVEL", "INFO").upper()
SAVE_RESULT_ROOT = partition_core.getenv_str("VISIONOPS_ROBOT_PROTOCOL_SAVE_RESULT_ROOT", "/tmp/vision_robot_protocol_latest")
SAVE_EVERY_TRIGGER = partition_core.getenv_int("VISIONOPS_ROBOT_PROTOCOL_SAVE_EVERY_TRIGGER", 1)

COORD_OUTPUT_FRAME = partition_core.getenv_str("VISIONOPS_COORD_OUTPUT_FRAME", "image").strip().lower()
COORD_REGISTER_ORDER = partition_core.getenv_str("VISIONOPS_COORD_REGISTER_ORDER", "column").strip().lower()

# Coordinate affine transform.
# Legacy VISIONOPS_COORD_A00...B1 is kept as the default/single-arm transform and
# also as the default left-hand transform for backward compatibility.
COORD_A00 = float(partition_core.getenv_str("VISIONOPS_COORD_A00", "1.0"))
COORD_A01 = float(partition_core.getenv_str("VISIONOPS_COORD_A01", "0.0"))
COORD_A10 = float(partition_core.getenv_str("VISIONOPS_COORD_A10", "0.0"))
COORD_A11 = float(partition_core.getenv_str("VISIONOPS_COORD_A11", "1.0"))
COORD_B0 = float(partition_core.getenv_str("VISIONOPS_COORD_B0", "0.0"))
COORD_B1 = float(partition_core.getenv_str("VISIONOPS_COORD_B1", "0.0"))

# 103 coordinate recognition can use different robot coordinate transforms for
# the left-hand and right-hand workspaces. A cell is assigned to left/right by
# its vision column index: default left cols 0~3, right cols 4~7.
COORD_DUAL_ARM_ENABLE = partition_core.getenv_int("VISIONOPS_COORD_DUAL_ARM_ENABLE", 0)
COORD_LEFT_COL_START = partition_core.getenv_int("VISIONOPS_COORD_LEFT_COL_START", 0)
COORD_LEFT_COL_END = partition_core.getenv_int("VISIONOPS_COORD_LEFT_COL_END", 3)
COORD_RIGHT_COL_START = partition_core.getenv_int("VISIONOPS_COORD_RIGHT_COL_START", 4)
COORD_RIGHT_COL_END = partition_core.getenv_int("VISIONOPS_COORD_RIGHT_COL_END", 7)

COORD_LEFT_A00 = float(partition_core.getenv_str("VISIONOPS_COORD_LEFT_A00", str(COORD_A00)))
COORD_LEFT_A01 = float(partition_core.getenv_str("VISIONOPS_COORD_LEFT_A01", str(COORD_A01)))
COORD_LEFT_A10 = float(partition_core.getenv_str("VISIONOPS_COORD_LEFT_A10", str(COORD_A10)))
COORD_LEFT_A11 = float(partition_core.getenv_str("VISIONOPS_COORD_LEFT_A11", str(COORD_A11)))
COORD_LEFT_B0 = float(partition_core.getenv_str("VISIONOPS_COORD_LEFT_B0", str(COORD_B0)))
COORD_LEFT_B1 = float(partition_core.getenv_str("VISIONOPS_COORD_LEFT_B1", str(COORD_B1)))

# If right-hand parameters are not configured, fall back to the legacy transform
# so old deployments keep working even when COORD_DUAL_ARM_ENABLE is enabled by mistake.
COORD_RIGHT_A00 = float(partition_core.getenv_str("VISIONOPS_COORD_RIGHT_A00", str(COORD_A00)))
COORD_RIGHT_A01 = float(partition_core.getenv_str("VISIONOPS_COORD_RIGHT_A01", str(COORD_A01)))
COORD_RIGHT_A10 = float(partition_core.getenv_str("VISIONOPS_COORD_RIGHT_A10", str(COORD_A10)))
COORD_RIGHT_A11 = float(partition_core.getenv_str("VISIONOPS_COORD_RIGHT_A11", str(COORD_A11)))
COORD_RIGHT_B0 = float(partition_core.getenv_str("VISIONOPS_COORD_RIGHT_B0", str(COORD_B0)))
COORD_RIGHT_B1 = float(partition_core.getenv_str("VISIONOPS_COORD_RIGHT_B1", str(COORD_B1)))

# 103 coordinate recognition policy.
# During product placement, some cells are covered and the visible cell count can be < 40.
# For trigger 103, keep previous registers for invisible slots and update only detected/matched slots.
COORD_ALWAYS_OK = partition_core.getenv_int("VISIONOPS_COORD_ALWAYS_OK", 1)
COORD_PARTIAL_UPDATE_ENABLE = partition_core.getenv_int("VISIONOPS_COORD_PARTIAL_UPDATE_ENABLE", 1)
COORD_PARTIAL_SLOT_MATCH_MAX_DIST_PX = float(partition_core.getenv_str("VISIONOPS_COORD_PARTIAL_SLOT_MATCH_MAX_DIST_PX", "22.0"))
COORD_PARTIAL_MIN_CONF = float(partition_core.getenv_str(
    "VISIONOPS_COORD_PARTIAL_MIN_CONF",
    partition_core.getenv_str("VISIONOPS_PARTITION_MIN_CONF", "0.10"),
))
COORD_TEMPLATE_PATH = Path(partition_core.getenv_str(
    "VISIONOPS_COORD_TEMPLATE_PATH",
    partition_core.getenv_str("VISIONOPS_PARTITION_TEMPLATE_PATH", str(PARTITION_DIR / "partition_template.json")),
))

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")


def u16(v: int) -> int:
    return int(v) & 0xFFFF


def result_code_from_task(result: Dict[str, Any]) -> int:
    return RESULT_OK if str(result.get("final_result", "")).upper() == "OK" else RESULT_NG


def create_context() -> ModbusServerContext:
    # Need at least 0~199 because the table defines Robot->Vision 100~199.
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


def safe_i16_coord(v: Any) -> Optional[int]:
    """Return signed int16 coordinate value.

    set_regs() will convert it to 16-bit two's-complement by u16().
    Robot/PLC should read coordinate registers as signed INT16.
    """
    try:
        iv = int(round(float(v)))
    except Exception:
        return None
    if iv < -32768:
        iv = -32768
    if iv > 32767:
        iv = 32767
    return iv


def coord_arm_from_slot_id(slot_id: int, rows: int, cols: int) -> str:
    """Return coordinate transform arm name for a vision slot.

    Vision slot_id is 0-based row-major, so col = slot_id % cols.
    With the default 5x8 grid, cols 0~3 are left-hand and cols 4~7 are
    right-hand. If a configured column range misses the slot, fall back to the
    image/grid midpoint to avoid silently dropping coordinates.
    """
    if COORD_DUAL_ARM_ENABLE != 1:
        return "single"
    try:
        col = int(slot_id) % max(int(cols), 1)
    except Exception:
        return "single"

    if COORD_LEFT_COL_START <= col <= COORD_LEFT_COL_END:
        return "left"
    if COORD_RIGHT_COL_START <= col <= COORD_RIGHT_COL_END:
        return "right"
    return "left" if col < max(int(cols), 1) / 2.0 else "right"


def coord_affine_params_for_arm(arm: str) -> tuple[float, float, float, float, float, float]:
    if COORD_DUAL_ARM_ENABLE == 1 and arm == "left":
        return COORD_LEFT_A00, COORD_LEFT_A01, COORD_LEFT_A10, COORD_LEFT_A11, COORD_LEFT_B0, COORD_LEFT_B1
    if COORD_DUAL_ARM_ENABLE == 1 and arm == "right":
        return COORD_RIGHT_A00, COORD_RIGHT_A01, COORD_RIGHT_A10, COORD_RIGHT_A11, COORD_RIGHT_B0, COORD_RIGHT_B1
    return COORD_A00, COORD_A01, COORD_A10, COORD_A11, COORD_B0, COORD_B1


def image_to_robot_coord(x_camera: Any, y_camera: Any, arm: str = "single") -> tuple[Optional[int], Optional[int]]:
    try:
        x = float(x_camera)
        y = float(y_camera)
    except Exception:
        return None, None

    if COORD_OUTPUT_FRAME in {"robot", "robot_mm", "base", "robot_base"}:
        a00, a01, a10, a11, b0, b1 = coord_affine_params_for_arm(arm)
        xr = a00 * x + a01 * y + b0
        yr = a10 * x + a11 * y + b1
    else:
        xr = x
        yr = y

    return safe_i16_coord(xr), safe_i16_coord(yr)


def coord_register_index_from_slot_id(slot_id: int, rows: int, cols: int) -> Optional[int]:
    """Map vision slot_id to register slot index.

    Vision slot_id is assumed to be 0-based row-major:
      old order: 0,1,...,7, 8,9,...,15, ...
    Register slot index can be:
      row-major:    left->right, top->bottom
      column-major: top->bottom, left->right
    """
    n = rows * cols
    if slot_id < 0 or slot_id >= n:
        return None

    order = COORD_REGISTER_ORDER.replace("-", "_").replace(" ", "_")
    if order in {"column", "col", "column_major", "col_major", "down_then_right", "top_down_left_right"}:
        row = slot_id // cols
        col = slot_id % cols
        return col * rows + row

    return slot_id


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _pred_conf(pred: Dict[str, Any]) -> float:
    try:
        if hasattr(partition_core, "pred_conf"):
            return float(partition_core.pred_conf(pred))
    except Exception:
        pass
    for key in ("confidence", "conf", "score", "prob"):
        if key in pred:
            v = _safe_float(pred.get(key))
            if v is not None:
                return v
    return 0.0


def _pred_bbox(pred: Dict[str, Any]) -> Optional[List[float]]:
    try:
        if hasattr(partition_core, "pred_bbox"):
            bbox = partition_core.pred_bbox(pred)
            if bbox is not None and len(bbox) >= 4:
                return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    except Exception:
        pass
    for key in ("bbox", "box", "xyxy"):
        bbox = pred.get(key)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            except Exception:
                return None
    keys = ("x1", "y1", "x2", "y2")
    if all(k in pred for k in keys):
        vals = [_safe_float(pred.get(k)) for k in keys]
        if all(v is not None for v in vals):
            return [float(v) for v in vals]  # type: ignore[arg-type]
    return None


def _find_predictions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        if hasattr(partition_core, "find_predictions"):
            preds = partition_core.find_predictions(payload)
            return [p for p in preds if isinstance(p, dict)]
    except Exception:
        pass

    def walk(obj: Any) -> List[Dict[str, Any]]:
        if isinstance(obj, list):
            if all(isinstance(x, dict) for x in obj):
                # Prefer lists that look like detections.
                if any(("bbox" in x or "box" in x or "xyxy" in x or "confidence" in x or "conf" in x or "score" in x) for x in obj):
                    return [x for x in obj if isinstance(x, dict)]
            out: List[Dict[str, Any]] = []
            for x in obj:
                out.extend(walk(x))
            return out
        if isinstance(obj, dict):
            for key in ("predictions", "detections", "objects", "results", "data"):
                if key in obj:
                    found = walk(obj.get(key))
                    if found:
                        return found
        return []

    return walk(payload)


def _center_from_template_cell(cell: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    for xk, yk in (("cx", "cy"), ("x", "y"), ("center_x", "center_y"), ("image_cx", "image_cy")):
        if xk in cell and yk in cell:
            return _safe_float(cell.get(xk)), _safe_float(cell.get(yk))
    center = cell.get("center") or cell.get("point")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return _safe_float(center[0]), _safe_float(center[1])
    return None, None


def _iter_template_cell_dicts(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("cells", "template_cells", "slots", "points", "centers"):
            val = obj.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
            if isinstance(val, dict):
                out: List[Dict[str, Any]] = []
                for k, v in val.items():
                    if isinstance(v, dict):
                        item = dict(v)
                        item.setdefault("slot_id", k)
                        out.append(item)
                    elif isinstance(v, (list, tuple)) and len(v) >= 2:
                        out.append({"slot_id": k, "cx": v[0], "cy": v[1]})
                return out
        # A template may itself be a slot-id keyed dict.
        out = []
        for k, v in obj.items():
            if isinstance(v, dict):
                item = dict(v)
                item.setdefault("slot_id", k)
                out.append(item)
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                out.append({"slot_id": k, "cx": v[0], "cy": v[1]})
        return out
    return []


def _parse_slot_id(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        pass
    try:
        import re
        m = re.search(r"(\d+)", str(raw))
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def load_template_slot_centers(rows: int, cols: int) -> List[Dict[str, Any]]:
    """Load template slot centers from partition_template.json for partial 103 matching."""
    n = min(max(rows * cols, 1), 40)
    path = COORD_TEMPLATE_PATH
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("coordinate partial update: failed to load template %s: %s", path, exc)
        return []

    raw_cells = _iter_template_cell_dicts(obj)
    parsed: List[Dict[str, Any]] = []
    raw_sids: List[int] = []
    for idx, cell in enumerate(raw_cells):
        x, y = _center_from_template_cell(cell)
        if x is None or y is None:
            continue
        sid = _parse_slot_id(cell.get("slot_id", cell.get("id", cell.get("slot"))))
        if sid is None:
            sid = idx
        raw_sids.append(sid)
        parsed.append({"slot_id": sid, "cx": float(x), "cy": float(y)})

    if not parsed:
        return []

    # Normalize common 1-based template slot IDs to 0-based IDs used by this service.
    if raw_sids and min(raw_sids) >= 1 and max(raw_sids) <= n and 0 not in raw_sids:
        for cell in parsed:
            cell["slot_id"] = int(cell["slot_id"]) - 1

    out = []
    used = set()
    for cell in parsed:
        sid = int(cell["slot_id"])
        if 0 <= sid < n and sid not in used:
            used.add(sid)
            out.append(cell)
    return out


def build_partial_cells_from_raw_predictions(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assign visible detections to nearest template slots, allowing fewer than 40 cells.

    This is used only for trigger 103. It solves the case where partition_core.analyze()
    fails grid assignment when count != 40 and therefore returns cells=[].
    """
    payload = result.get("_infer_payload") if isinstance(result.get("_infer_payload"), dict) else {}
    preds = _find_predictions(payload)

    rows = int(result.get("expected_rows") or partition_core.EXPECTED_ROWS or 5)
    cols = int(result.get("expected_cols") or partition_core.EXPECTED_COLS or 8)
    rows = min(max(rows, 1), 40)
    cols = min(max(cols, 1), 40)
    template = load_template_slot_centers(rows, cols)
    if not template:
        result["coord_partial_update_error"] = "template_not_available"
        return []

    best_by_slot: Dict[int, Dict[str, Any]] = {}
    unmatched = 0
    for pred in preds:
        if not isinstance(pred, dict):
            continue
        conf = _pred_conf(pred)
        if conf < COORD_PARTIAL_MIN_CONF:
            continue
        bbox = _pred_bbox(pred)
        if bbox is None:
            unmatched += 1
            continue
        x1, y1, x2, y2 = bbox[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        best_slot = None
        best_dist2 = None
        for slot in template:
            dx = cx - float(slot["cx"])
            dy = cy - float(slot["cy"])
            dist2 = dx * dx + dy * dy
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_slot = int(slot["slot_id"])

        if best_slot is None or best_dist2 is None:
            unmatched += 1
            continue
        dist = best_dist2 ** 0.5
        if dist > COORD_PARTIAL_SLOT_MATCH_MAX_DIST_PX:
            unmatched += 1
            continue

        cell = {
            "slot_id": best_slot,
            "cx": cx,
            "cy": cy,
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "partial_slot_match_dist_px": round(dist, 3),
            "partial_update_source": "raw_prediction_nearest_template",
        }
        prev = best_by_slot.get(best_slot)
        if prev is None:
            best_by_slot[best_slot] = cell
        else:
            # For duplicate detections assigned to the same slot, keep the closer one;
            # if distance is equal, keep higher confidence.
            prev_dist = float(prev.get("partial_slot_match_dist_px", 1e9))
            prev_conf = float(prev.get("confidence", 0.0))
            if dist < prev_dist or (abs(dist - prev_dist) < 1e-6 and conf > prev_conf):
                best_by_slot[best_slot] = cell

    cells = [best_by_slot[k] for k in sorted(best_by_slot.keys())]
    result["coord_partial_update_debug"] = {
        "enabled": True,
        "source": "raw_predictions_nearest_template",
        "template_path": str(COORD_TEMPLATE_PATH),
        "raw_prediction_count": len(preds),
        "matched_cell_count": len(cells),
        "unmatched_or_filtered_count": unmatched,
        "min_conf": COORD_PARTIAL_MIN_CONF,
        "max_match_dist_px": COORD_PARTIAL_SLOT_MATCH_MAX_DIST_PX,
    }
    return cells


def ensure_coordinate_cells_for_partial_update(result: Dict[str, Any]) -> None:
    if COORD_PARTIAL_UPDATE_ENABLE != 1:
        return

    cells = result.get("cells") if isinstance(result.get("cells"), list) else []
    valid_existing = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if cell.get("slot_id") is None:
            continue
        if cell.get("cx") is None or cell.get("cy") is None:
            continue
        valid_existing.append(cell)

    if valid_existing:
        result["coord_partial_update_debug"] = {
            "enabled": True,
            "source": "existing_analyze_cells",
            "matched_cell_count": len(valid_existing),
        }
        return

    partial_cells = build_partial_cells_from_raw_predictions(result)
    if partial_cells:
        result["cells"] = partial_cells
        result["valid_cell_count"] = len(partial_cells)
        result["coord_cells_filled_from_raw_predictions"] = True


def write_partition_coordinates(context: ModbusServerContext, result: Dict[str, Any]) -> None:
    """Write 40 cell center coordinates into registers 20~99.

    Coordinates are not cleared when there is no coordinate trigger. Missing slots keep their previous value.
    """
    rows = int(result.get("expected_rows") or partition_core.EXPECTED_ROWS or 5)
    cols = int(result.get("expected_cols") or partition_core.EXPECTED_COLS or 8)
    rows = min(max(rows, 1), 40)
    cols = min(max(cols, 1), 40)
    n = min(max(rows * cols, 1), 40)

    # Read current values so missing slots are preserved.
    try:
        current = context[UNIT_ID].getValues(3, ADDRESS_BASE + REG_COORD_BASE, count=80)
    except TypeError:
        current = context[UNIT_ID].getValues(3, ADDRESS_BASE + REG_COORD_BASE, 80)
    coords = [int(x) for x in current[:80]] if current else [0] * 80
    if len(coords) < 80:
        coords += [0] * (80 - len(coords))

    cells = result.get("cells") if isinstance(result.get("cells"), list) else []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        try:
            sid = int(cell.get("slot_id"))
        except Exception:
            continue
        out_idx = coord_register_index_from_slot_id(sid, rows, cols)
        if out_idx is not None and 0 <= out_idx < n:
            x_img = cell.get("cx")
            y_img = cell.get("cy")
            arm = coord_arm_from_slot_id(sid, rows, cols)
            x_out, y_out = image_to_robot_coord(x_img, y_img, arm=arm)
            row = sid // cols if cols > 0 else 0
            col = sid % cols if cols > 0 else 0

            # Keep debug information in result.json.
            cell["output_frame"] = COORD_OUTPUT_FRAME
            cell["coord_dual_arm_enable"] = COORD_DUAL_ARM_ENABLE
            cell["coord_arm"] = arm
            cell["vision_row"] = row
            cell["vision_col"] = col
            cell["register_order"] = COORD_REGISTER_ORDER
            cell["vision_slot_id"] = sid
            cell["register_slot_id"] = out_idx
            cell["register_x"] = REG_COORD_BASE + 2 * out_idx
            cell["register_y"] = REG_COORD_BASE + 2 * out_idx + 1
            cell["image_cx"] = x_img
            cell["image_cy"] = y_img
            if x_out is not None:
                cell["robot_cx"] = x_out
                coords[2 * out_idx] = x_out
            if y_out is not None:
                cell["robot_cy"] = y_out
                coords[2 * out_idx + 1] = y_out

    set_regs(context, REG_COORD_BASE, coords)


def save_partition_debug(task_name: str, rgb_bytes: bytes, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    if SAVE_EVERY_TRIGGER != 1:
        return
    try:
        out_dir = Path(SAVE_RESULT_ROOT) / task_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "rgb.jpg").write_bytes(rgb_bytes)
        (out_dir / "infer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            partition_core.draw_overlay(rgb_bytes, result, out_dir / "overlay.jpg")
        except Exception:
            logging.exception("failed to save partition overlay")
    except Exception:
        logging.exception("failed to save partition debug files")


def save_tube_debug(rgb_bytes: bytes, depth_bytes: bytes, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    if SAVE_EVERY_TRIGGER != 1:
        return
    try:
        out_dir = Path(SAVE_RESULT_ROOT) / "product_tube_check"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "rgb.jpg").write_bytes(rgb_bytes)
        (out_dir / "depth.png").write_bytes(depth_bytes)
        (out_dir / "infer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            tube_core.draw_tube_overlay(rgb_bytes, payload, result, out_dir / "overlay.jpg")
        except Exception:
            logging.exception("failed to save tube overlay")
    except Exception:
        logging.exception("failed to save tube debug files")


def run_partition_once() -> Dict[str, Any]:
    logging.info("partition: fetch RGB snapshot: %s", partition_core.SNAPSHOT_URL)
    rgb_bytes = partition_core.http_get_bytes(partition_core.SNAPSHOT_URL, partition_core.HTTP_TIMEOUT_S)
    if not rgb_bytes:
        raise RuntimeError("partition snapshot is empty")
    logging.info("partition: post C++ infer: %s bytes=%d", partition_core.INFER_URL, len(rgb_bytes))
    payload = partition_core.post_multipart_image(partition_core.INFER_URL, rgb_bytes, timeout_s=partition_core.HTTP_TIMEOUT_S)
    result = partition_core.analyze(payload)
    result["_rgb_bytes"] = rgb_bytes
    result["_infer_payload"] = payload
    return result


def tube_region_from_trigger_cmd(cmd: int) -> str:
    return TUBE_TRIGGER_TO_REGION.get(int(cmd), "none")


def run_tube_once(region: str = "all", trigger_cmd: int = TUBE_TRIGGER_ALL) -> Dict[str, Any]:
    logging.info("tube: fetch RGB snapshot: %s region=%s trigger_cmd=%s", tube_core.SNAPSHOT_URL, region, trigger_cmd)
    rgb_bytes = tube_core.http_get_bytes(tube_core.SNAPSHOT_URL, tube_core.HTTP_TIMEOUT_S)
    if not rgb_bytes:
        raise RuntimeError("tube snapshot is empty")
    logging.info("tube: post OBB infer: %s bytes=%d", tube_core.INFER_URL, len(rgb_bytes))
    payload = tube_core.post_multipart_image(tube_core.INFER_URL, rgb_bytes, timeout_s=tube_core.HTTP_TIMEOUT_S)
    logging.info("tube: fetch depth PNG: %s", tube_core.DEPTH_URL)
    depth_bytes = tube_core.http_get_bytes(tube_core.DEPTH_URL, tube_core.HTTP_TIMEOUT_S)
    depth = tube_core.decode_depth_png(depth_bytes)
    result = tube_core.analyze(payload, depth, region=region)
    result["trigger_cmd"] = int(trigger_cmd)
    result["trigger_region"] = region
    result["_rgb_bytes"] = rgb_bytes
    result["_depth_bytes"] = depth_bytes
    result["_infer_payload"] = payload
    return result


class VisionRobotProtocolService:
    def __init__(self, context: ModbusServerContext) -> None:
        self.context = context
        self.lock = threading.Lock()
        self.busy = False
        self.heartbeat = 0
        self.last_cmd = {
            REG_TRIGGER_PARTITION: 0,
            REG_TRIGGER_PRODUCT: 0,
            REG_TRIGGER_COORD: 0,
        }

    def init_registers(self) -> None:
        set_regs(self.context, 0, [0] * max(REGISTER_COUNT, 200))

    def heartbeat_loop(self) -> None:
        while True:
            self.heartbeat = 0 if self.heartbeat >= 1000 else self.heartbeat + 1
            set_reg(self.context, REG_VISION_HEARTBEAT, self.heartbeat)
            time.sleep(0.5)

    def maybe_clear_result(self, trigger_reg: int, result_reg: int) -> None:
        cmd = get_reg(self.context, trigger_reg)
        if cmd == 0:
            self.last_cmd[trigger_reg] = 0
            set_reg(self.context, result_reg, RESULT_NONE)

    def poll_loop(self) -> None:
        logging.info("Vision robot protocol trigger poll started")
        while True:
            try:
                # Clear result registers while their trigger signals are 0. Coordinates are intentionally kept.
                self.maybe_clear_result(REG_TRIGGER_PARTITION, REG_PARTITION_RESULT)
                self.maybe_clear_result(REG_TRIGGER_PRODUCT, REG_PRODUCT_RESULT)
                self.maybe_clear_result(REG_TRIGGER_COORD, REG_COORD_RESULT)

                # Start at most one task at a time to avoid competing for camera/infer service.
                if not self.busy:
                    part_cmd = get_reg(self.context, REG_TRIGGER_PARTITION)
                    tube_cmd = get_reg(self.context, REG_TRIGGER_PRODUCT)
                    coord_cmd = get_reg(self.context, REG_TRIGGER_COORD)

                    if part_cmd == 1 and self.last_cmd[REG_TRIGGER_PARTITION] == 0:
                        self.last_cmd[REG_TRIGGER_PARTITION] = 1
                        threading.Thread(target=self.handle_partition_check, daemon=True).start()
                    elif tube_cmd in TUBE_TRIGGER_TO_REGION and self.last_cmd[REG_TRIGGER_PRODUCT] == 0:
                        self.last_cmd[REG_TRIGGER_PRODUCT] = int(tube_cmd)
                        threading.Thread(target=self.handle_tube_check, args=(int(tube_cmd),), daemon=True).start()
                    elif coord_cmd == 1 and self.last_cmd[REG_TRIGGER_COORD] == 0:
                        self.last_cmd[REG_TRIGGER_COORD] = 1
                        threading.Thread(target=self.handle_coordinate_check, daemon=True).start()
            except Exception:
                logging.exception("trigger poll error")
            time.sleep(POLL_INTERVAL_S)

    def _enter_busy(self) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            return True

    def _leave_busy(self) -> None:
        with self.lock:
            self.busy = False

    def handle_partition_check(self) -> None:
        if not self._enter_busy():
            return
        set_reg(self.context, REG_PARTITION_RESULT, RESULT_NONE)
        start = time.time()
        try:
            result = run_partition_once()
            code = result_code_from_task(result)
            set_reg(self.context, REG_PARTITION_RESULT, code)
            rgb_bytes = result.pop("_rgb_bytes", b"")
            payload = result.pop("_infer_payload", {})
            save_partition_debug("partition_check", rgb_bytes, payload, result)
            logging.info("partition check done: result=%s code=%d reason=%s time=%dms", result.get("final_result"), code, result.get("reason"), int((time.time()-start)*1000))
        except Exception as exc:
            logging.error("partition check failed: %s", exc)
            logging.debug("%s", traceback.format_exc())
            set_reg(self.context, REG_PARTITION_RESULT, RESULT_NG)
        finally:
            self._leave_busy()

    def handle_tube_check(self, trigger_cmd: int = TUBE_TRIGGER_ALL) -> None:
        if not self._enter_busy():
            return
        region = tube_region_from_trigger_cmd(trigger_cmd)
        set_reg(self.context, REG_PRODUCT_RESULT, RESULT_NONE)
        start = time.time()
        try:
            if region == "none":
                raise RuntimeError(f"invalid tube trigger command: {trigger_cmd}")
            result = run_tube_once(region=region, trigger_cmd=trigger_cmd)
            code = result_code_from_task(result)
            set_reg(self.context, REG_PRODUCT_RESULT, code)
            rgb_bytes = result.pop("_rgb_bytes", b"")
            depth_bytes = result.pop("_depth_bytes", b"")
            payload = result.pop("_infer_payload", {})
            save_tube_debug(rgb_bytes, depth_bytes, payload, result)
            logging.info(
                "tube/product check done: trigger_cmd=%s region=%s result=%s code=%d reason=%s selected_stand=%s selected_lying=%s high=%s time=%dms",
                trigger_cmd,
                region,
                result.get("final_result"),
                code,
                result.get("reason"),
                result.get("selected_stand_count"),
                result.get("selected_lying_count"),
                result.get("high_count"),
                int((time.time()-start)*1000),
            )
        except Exception as exc:
            logging.error("tube/product check failed: trigger_cmd=%s region=%s error=%s", trigger_cmd, region, exc)
            logging.debug("%s", traceback.format_exc())
            set_reg(self.context, REG_PRODUCT_RESULT, RESULT_NG)
        finally:
            self._leave_busy()

    def handle_coordinate_check(self) -> None:
        if not self._enter_busy():
            return
        set_reg(self.context, REG_COORD_RESULT, RESULT_NONE)
        start = time.time()
        try:
            result = run_partition_once()
            original_status = {
                "final_result": result.get("final_result"),
                "reason": result.get("reason"),
                "error": result.get("error"),
                "raw_prediction_count": result.get("raw_prediction_count"),
                "valid_cell_count": result.get("valid_cell_count"),
                "expected_count": result.get("expected_count"),
            }

            # Trigger 103 is used while products are being placed into cells, so count != 40 is normal.
            # If analyze() cannot assign a full grid and returns cells=[], recover the visible cells
            # from raw predictions by matching them to the saved template. Missing/covered slots keep
            # their previous register values because write_partition_coordinates() preserves them.
            ensure_coordinate_cells_for_partial_update(result)
            write_partition_coordinates(self.context, result)

            if COORD_ALWAYS_OK == 1:
                result["coord_original_status"] = original_status
                result["coord_result_override"] = {
                    "enabled": True,
                    "policy": "always_ok_for_coordinate_task_partial_update",
                    "reason": "trigger 103 ignores count mismatch; only visible matched cells update registers",
                    "register_result": "OK",
                }
                result["final_result"] = "OK"
                result["reason"] = "NONE"
                code = RESULT_OK
            else:
                code = result_code_from_task(result)

            set_reg(self.context, REG_COORD_RESULT, code)
            rgb_bytes = result.pop("_rgb_bytes", b"")
            payload = result.pop("_infer_payload", {})
            save_partition_debug("coordinate_check", rgb_bytes, payload, result)
            logging.info(
                "coordinate check done: result=%s code=%d reason=%s cells=%s partial=%s time=%dms",
                result.get("final_result"),
                code,
                result.get("reason"),
                result.get("valid_cell_count"),
                result.get("coord_partial_update_debug"),
                int((time.time()-start)*1000),
            )
        except Exception as exc:
            logging.error("coordinate check failed: %s", exc)
            logging.debug("%s", traceback.format_exc())
            set_reg(self.context, REG_COORD_RESULT, RESULT_NG)
        finally:
            self._leave_busy()


def main() -> int:
    if ENABLE != 1:
        logging.warning("VISIONOPS_ROBOT_PROTOCOL_ENABLE != 1, exit")
        return 0

    context = create_context()
    service = VisionRobotProtocolService(context)
    service.init_registers()

    threading.Thread(target=service.heartbeat_loop, daemon=True).start()
    threading.Thread(target=service.poll_loop, daemon=True).start()

    identity = None
    if ModbusDeviceIdentification is not None:
        identity = ModbusDeviceIdentification()
        identity.VendorName = "VisionOps"
        identity.ProductCode = "ROBOT_PROTO"
        identity.ProductName = "VisionOps Robot Protocol Modbus TCP"
        identity.ModelName = "LB3576 HP60C Partition+Tube Robot Gateway"
        identity.MajorMinorRevision = "1.0"

    logging.info(
        "starting VisionOps robot protocol: host=%s port=%d unit_id=%d single_slave=%d address_base=%d register_count=%d",
        HOST, PORT, UNIT_ID, SINGLE_SLAVE, ADDRESS_BASE, max(REGISTER_COUNT, 200),
    )
    logging.info("registers: result 1/2/3, coords 20~99, triggers 101/102/103, tube trigger 102: 1=left,2=right,3=all, coord_register_order=%s", COORD_REGISTER_ORDER)
    logging.info(
        "coordinate output frame=%s, single/legacy affine A=[[%.6f, %.6f], [%.6f, %.6f]], b=[%.6f, %.6f]",
        COORD_OUTPUT_FRAME,
        COORD_A00,
        COORD_A01,
        COORD_A10,
        COORD_A11,
        COORD_B0,
        COORD_B1,
    )
    logging.info(
        "coordinate dual-arm=%d, left_cols=%d~%d affine A=[[%.6f, %.6f], [%.6f, %.6f]], b=[%.6f, %.6f], right_cols=%d~%d affine A=[[%.6f, %.6f], [%.6f, %.6f]], b=[%.6f, %.6f]",
        COORD_DUAL_ARM_ENABLE,
        COORD_LEFT_COL_START,
        COORD_LEFT_COL_END,
        COORD_LEFT_A00,
        COORD_LEFT_A01,
        COORD_LEFT_A10,
        COORD_LEFT_A11,
        COORD_LEFT_B0,
        COORD_LEFT_B1,
        COORD_RIGHT_COL_START,
        COORD_RIGHT_COL_END,
        COORD_RIGHT_A00,
        COORD_RIGHT_A01,
        COORD_RIGHT_A10,
        COORD_RIGHT_A11,
        COORD_RIGHT_B0,
        COORD_RIGHT_B1,
    )
    StartTcpServer(context=context, identity=identity, address=(HOST, PORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())