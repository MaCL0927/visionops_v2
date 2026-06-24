#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps carton partition cell-structure one-shot debug script.

Purpose:
  1. Fetch one HP60C RGB snapshot from the existing bridge.
  2. POST it to the existing C++ /api/cpp/infer endpoint.
  3. Parse YOLO cell detections.
  4. Check whether the detected 5x8 cell grid matches a calibrated normal template.

This script does NOT change the camera or inference interface.  It is intended as
an edge/robot_gateway test module before enabling the Modbus-TCP service.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[ERROR] This script requires python3-opencv and numpy: {exc}", file=sys.stderr)
    print("        Try: sudo apt install -y python3-opencv python3-numpy", file=sys.stderr)
    raise

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ENV = THIS_DIR / "partition_check.env"


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
    except Exception:
        return default


def getenv_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def parse_int_set(raw: str, default: Iterable[int]) -> set[int]:
    out: set[int] = set()
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.add(int(item))
        except Exception:
            pass
    return out or set(default)


def parse_name_set(raw: str, default: Iterable[str]) -> set[str]:
    out = {x.strip().lower() for x in str(raw or "").split(",") if x.strip()}
    return out or {x.strip().lower() for x in default}


load_env_file(Path(os.environ.get("VISIONOPS_PARTITION_ENV", str(DEFAULT_ENV))))

SNAPSHOT_URL = getenv_str("VISIONOPS_PARTITION_SNAPSHOT_URL", "http://127.0.0.1:18182/stream/snapshot.jpg")
INFER_URL = getenv_str("VISIONOPS_PARTITION_INFER_URL", "http://127.0.0.1:8090/api/cpp/infer")
HTTP_TIMEOUT_S = getenv_int("VISIONOPS_PARTITION_HTTP_TIMEOUT_MS", 5000) / 1000.0
MIN_CONF = getenv_float("VISIONOPS_PARTITION_MIN_CONF", 0.25)
CELL_CLASS_IDS = parse_int_set(getenv_str("VISIONOPS_PARTITION_CELL_CLASS_IDS", "0"), [0])
CELL_CLASS_NAMES = parse_name_set(getenv_str("VISIONOPS_PARTITION_CELL_CLASS_NAMES", "cell,paper_cell"), ["cell", "paper_cell"])
EXPECTED_ROWS = getenv_int("VISIONOPS_PARTITION_EXPECTED_ROWS", 5)
EXPECTED_COLS = getenv_int("VISIONOPS_PARTITION_EXPECTED_COLS", 8)
EXPECTED_COUNT = getenv_int("VISIONOPS_PARTITION_EXPECTED_COUNT", EXPECTED_ROWS * EXPECTED_COLS)
STRICT_COUNT = getenv_int("VISIONOPS_PARTITION_STRICT_COUNT", 1)
NMS_IOU = getenv_float("VISIONOPS_PARTITION_NMS_IOU", 0.60)
TEMPLATE_PATH = getenv_str("VISIONOPS_PARTITION_TEMPLATE_PATH", str(THIS_DIR / "partition_template.json"))

# Thresholds.  These are intentionally conservative defaults; tune them from normal samples.
MAX_MEAN_CENTER_ERR_PX = getenv_float("VISIONOPS_PARTITION_MAX_MEAN_CENTER_ERR_PX", 22.0)
MAX_P95_CENTER_ERR_PX = getenv_float("VISIONOPS_PARTITION_MAX_P95_CENTER_ERR_PX", 38.0)
# Added in v0.2: keep global mean/P95 thresholds loose, and use local/row/edge metrics
# to catch tilted partition cases where the detector still returns all 40 cells.
# Set any of the following thresholds to <= 0 to disable that check.
MAX_CENTER_ERR_PX = getenv_float("VISIONOPS_PARTITION_MAX_CENTER_ERR_PX", 24.0)
MAX_EDGE_CELL_ERR_PX = getenv_float("VISIONOPS_PARTITION_MAX_EDGE_CELL_ERR_PX", 20.0)
MAX_ROW_ANGLE_DIFF_MAX_DEG = getenv_float("VISIONOPS_PARTITION_MAX_ROW_ANGLE_DIFF_MAX_DEG", 3.5)
MAX_ROW_ANGLE_STD_DIFF_DEG = getenv_float("VISIONOPS_PARTITION_MAX_ROW_ANGLE_STD_DIFF_DEG", 0.70)
MAX_GRID_CENTER_OFFSET_PX = getenv_float("VISIONOPS_PARTITION_MAX_GRID_CENTER_OFFSET_PX", 35.0)
MAX_ROW_ANGLE_DIFF_DEG = getenv_float("VISIONOPS_PARTITION_MAX_ROW_ANGLE_DIFF_DEG", 5.0)
MAX_COL_ANGLE_DIFF_DEG = getenv_float("VISIONOPS_PARTITION_MAX_COL_ANGLE_DIFF_DEG", 5.0)
MAX_AFFINE_ROT_DEG = getenv_float("VISIONOPS_PARTITION_MAX_AFFINE_ROT_DEG", 5.0)
MAX_AFFINE_SHEAR = getenv_float("VISIONOPS_PARTITION_MAX_AFFINE_SHEAR", 0.18)
ENABLE_SIZE_CHECK = getenv_int("VISIONOPS_PARTITION_ENABLE_SIZE_CHECK", 0)
MIN_BOX_SIZE_RATIO = getenv_float("VISIONOPS_PARTITION_MIN_BOX_SIZE_RATIO", 0.55)
MAX_BOX_SIZE_RATIO = getenv_float("VISIONOPS_PARTITION_MAX_BOX_SIZE_RATIO", 1.80)
MAX_BAD_SIZE_COUNT = getenv_int("VISIONOPS_PARTITION_MAX_BAD_SIZE_COUNT", 6)


def now_ms() -> int:
    return int(time.time() * 1000)


def http_get_bytes(url: str, timeout_s: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "VisionOps-PartitionCheck/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        code = getattr(resp, "status", 200)
        if code < 200 or code >= 300:
            raise RuntimeError(f"GET {url} HTTP {code}")
        return resp.read()


def post_multipart_image(url: str, image_bytes: bytes, filename: str = "hp60c_partition_trigger.jpg", timeout_s: float = 5.0) -> Dict[str, Any]:
    boundary = "----VisionOpsPartitionBoundary" + str(now_ms())
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
            "User-Agent": "VisionOps-PartitionCheck/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc.reason)
        raise RuntimeError(f"POST {url} HTTP {exc.code}: {detail[:500]}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"infer returned non JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError("infer JSON is not an object")
    return obj


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


def pred_name(pred: Dict[str, Any]) -> str:
    for key in ("class_name", "class", "label", "name"):
        v = pred.get(key)
        if v is not None:
            return str(v).strip().lower()
    return ""


def pred_conf(pred: Dict[str, Any]) -> float:
    raw = pred.get("confidence", pred.get("score", pred.get("conf", 0.0)))
    try:
        return float(raw)
    except Exception:
        return 0.0


def pred_bbox(pred: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bbox = pred.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [float(x) for x in bbox[:4]]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return x1, y1, x2, y2
        except Exception:
            pass

    # Some C++ payloads may expose center/width/height instead of bbox.
    cx = pred.get("center_x", pred.get("cx"))
    cy = pred.get("center_y", pred.get("cy"))
    w = pred.get("width", pred.get("w"))
    h = pred.get("height", pred.get("h"))
    if cx is not None and cy is not None and w is not None and h is not None:
        try:
            cx_f, cy_f, w_f, h_f = float(cx), float(cy), float(w), float(h)
            return cx_f - w_f / 2, cy_f - h_f / 2, cx_f + w_f / 2, cy_f + h_f / 2
        except Exception:
            pass

    center = pred.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2 and w is not None and h is not None:
        try:
            cx_f, cy_f, w_f, h_f = float(center[0]), float(center[1]), float(w), float(h)
            return cx_f - w_f / 2, cy_f - h_f / 2, cx_f + w_f / 2, cy_f + h_f / 2
        except Exception:
            pass
    return None


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def nms_items(items: List[Dict[str, Any]], iou_thres: float) -> List[Dict[str, Any]]:
    if iou_thres <= 0:
        return items
    ordered = sorted(items, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    keep: List[Dict[str, Any]] = []
    for item in ordered:
        bbox = tuple(item.get("bbox", []))
        if len(bbox) != 4:
            continue
        if all(iou_xyxy(bbox, tuple(k.get("bbox", []))) <= iou_thres for k in keep):
            keep.append(item)
    return keep


def parse_cell_items(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, int, int]:
    preds = find_predictions(payload)
    width, height = image_size_from_payload(payload)
    items: List[Dict[str, Any]] = []
    for idx, pred in enumerate(preds):
        cid = pred_class_id(pred)
        name = pred_name(pred)
        conf = pred_conf(pred)
        if cid not in CELL_CLASS_IDS and name not in CELL_CLASS_NAMES:
            continue
        if conf < MIN_CONF:
            continue
        bbox = pred_bbox(pred)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 1 or h <= 1:
            continue
        items.append({
            "idx": idx,
            "class_id": cid,
            "class_name": name,
            "confidence": round(conf, 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "cx": round((x1 + x2) / 2.0, 2),
            "cy": round((y1 + y2) / 2.0, 2),
            "w": round(w, 2),
            "h": round(h, 2),
            "area": round(w * h, 2),
        })
    items = nms_items(items, NMS_IOU)
    # Keep deterministic order for logging; row/col assignment happens later.
    items = sorted(items, key=lambda x: (float(x["cy"]), float(x["cx"])))
    return items, len(preds), width, height


def fit_line_angle(points: List[Tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    arr = np.array(points, dtype=np.float32)
    mean = arr.mean(axis=0)
    centered = arr - mean
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        vx, vy = float(vt[0, 0]), float(vt[0, 1])
        return math.degrees(math.atan2(vy, vx))
    except Exception:
        # Fallback from first to last point.
        x0, y0 = points[0]
        x1, y1 = points[-1]
        return math.degrees(math.atan2(y1 - y0, x1 - x0))


def angle_diff_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    d = (float(a) - float(b) + 90.0) % 180.0 - 90.0
    return float(d)


def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def angle_diff_list(cur: Any, tpl: Any) -> List[float]:
    """Pairwise signed angle difference list, in degrees, normalized to [-90, 90)."""
    if not isinstance(cur, list) or not isinstance(tpl, list):
        return []
    out: List[float] = []
    for a, b in zip(cur, tpl):
        af = safe_float(a)
        bf = safe_float(b)
        if af is None or bf is None:
            continue
        d = angle_diff_deg(af, bf)
        if d is not None:
            out.append(float(d))
    return out


def abs_diff_float(a: Any, b: Any) -> Optional[float]:
    af = safe_float(a)
    bf = safe_float(b)
    if af is None or bf is None:
        return None
    return abs(af - bf)


def assign_grid_by_sort(items: List[Dict[str, Any]], rows: int, cols: int) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Assign row_id/col_id by sorted-y chunks and per-row sorted x.

    This is intentionally simple and stable for the fixed camera scene.  It assumes
    that the model has detected exactly rows*cols cells.  If count is different,
    the returned ok flag is False but partial information is still kept.
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    expected = rows * cols
    if len(items) != expected:
        return False, f"count_mismatch_for_grid_assign: got {len(items)}, expected {expected}", items

    ordered_y = sorted([dict(x) for x in items], key=lambda x: (float(x["cy"]), float(x["cx"])))
    out: List[Dict[str, Any]] = []
    for r in range(rows):
        row = ordered_y[r * cols:(r + 1) * cols]
        if len(row) != cols:
            return False, f"row_{r}_count_{len(row)}_not_{cols}", items
        row = sorted(row, key=lambda x: float(x["cx"]))
        for c, item in enumerate(row):
            item["row_id"] = int(r)
            item["col_id"] = int(c)
            item["slot_id"] = int(r * cols + c)
            out.append(item)
    out = sorted(out, key=lambda x: int(x["slot_id"]))
    return True, "ok", out


def grid_geometry(items: List[Dict[str, Any]], rows: int, cols: int) -> Dict[str, Any]:
    row_angles: List[float] = []
    col_angles: List[float] = []
    for r in range(rows):
        pts = [(float(x["cx"]), float(x["cy"])) for x in items if int(x.get("row_id", -1)) == r]
        pts = sorted(pts, key=lambda p: p[0])
        angle = fit_line_angle(pts)
        if angle is not None:
            # Normalize row angle close to horizontal.
            d0 = angle_diff_deg(angle, 0.0)
            row_angles.append(float(d0 if d0 is not None else angle))
    for c in range(cols):
        pts = [(float(x["cx"]), float(x["cy"])) for x in items if int(x.get("col_id", -1)) == c]
        pts = sorted(pts, key=lambda p: p[1])
        angle = fit_line_angle(pts)
        if angle is not None:
            # Normalize column angle close to vertical, represented as deviation from 90 deg.
            d90 = angle_diff_deg(angle, 90.0)
            col_angles.append(float(d90 if d90 is not None else angle))

    cxs = [float(x["cx"]) for x in items]
    cys = [float(x["cy"]) for x in items]
    ws = [float(x["w"]) for x in items]
    hs = [float(x["h"]) for x in items]
    return {
        "row_angles_deg": [round(x, 3) for x in row_angles],
        "col_angles_deg_from_vertical": [round(x, 3) for x in col_angles],
        "mean_row_angle_deg": round(float(np.mean(row_angles)), 3) if row_angles else None,
        "mean_col_angle_deg_from_vertical": round(float(np.mean(col_angles)), 3) if col_angles else None,
        "std_row_angle_deg": round(float(np.std(row_angles)), 3) if row_angles else None,
        "std_col_angle_deg": round(float(np.std(col_angles)), 3) if col_angles else None,
        "grid_center": [round(float(np.mean(cxs)), 2), round(float(np.mean(cys)), 2)] if cxs and cys else [0, 0],
        "grid_bbox": [round(min(cxs), 2), round(min(cys), 2), round(max(cxs), 2), round(max(cys), 2)] if cxs and cys else [0, 0, 0, 0],
        "mean_box_w": round(float(np.mean(ws)), 2) if ws else 0,
        "mean_box_h": round(float(np.mean(hs)), 2) if hs else 0,
    }


def make_template(items: List[Dict[str, Any]], rows: int, cols: int, image_width: int, image_height: int) -> Dict[str, Any]:
    geom = grid_geometry(items, rows, cols)
    cells: List[Dict[str, Any]] = []
    for item in sorted(items, key=lambda x: int(x.get("slot_id", 0))):
        cells.append({
            "slot_id": int(item["slot_id"]),
            "row_id": int(item["row_id"]),
            "col_id": int(item["col_id"]),
            "cx": float(item["cx"]),
            "cy": float(item["cy"]),
            "w": float(item["w"]),
            "h": float(item["h"]),
            "area": float(item["area"]),
        })
    return {
        "version": 1,
        "created_at_ms": now_ms(),
        "expected_rows": int(rows),
        "expected_cols": int(cols),
        "expected_count": int(rows * cols),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "geometry": geom,
        "cells": cells,
    }


def load_template(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"template is not object: {path}")
    cells = obj.get("cells")
    if not isinstance(cells, list):
        raise RuntimeError(f"template has no cells list: {path}")
    return obj


def points_by_slot(items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for item in items:
        try:
            out[int(item["slot_id"])] = item
        except Exception:
            pass
    return out


def template_cells_by_slot(template: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for cell in template.get("cells", []):
        if isinstance(cell, dict):
            try:
                out[int(cell["slot_id"])] = cell
            except Exception:
                pass
    return out


def estimate_affine_metrics(template_pts: List[Tuple[float, float]], current_pts: List[Tuple[float, float]]) -> Dict[str, Any]:
    if len(template_pts) < 3 or len(template_pts) != len(current_pts):
        return {"ok": False}
    src = np.array(template_pts, dtype=np.float32)
    dst = np.array(current_pts, dtype=np.float32)
    metrics: Dict[str, Any] = {"ok": False}
    try:
        M, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0, maxIters=2000)
        if M is None:
            return metrics
        A = M[:2, :2]
        t = M[:2, 2]
        col0 = A[:, 0]
        col1 = A[:, 1]
        sx = float(np.linalg.norm(col0))
        sy = float(np.linalg.norm(col1))
        rot = math.degrees(math.atan2(float(A[1, 0]), float(A[0, 0])))
        shear = 0.0
        if sx > 1e-6 and sy > 1e-6:
            shear = abs(float(np.dot(col0, col1)) / (sx * sy))
        pred = np.concatenate([src, np.ones((src.shape[0], 1), dtype=np.float32)], axis=1) @ M.T
        err = np.sqrt(((pred - dst) ** 2).sum(axis=1))
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        metrics.update({
            "ok": True,
            "matrix": [[round(float(v), 6) for v in row] for row in M.tolist()],
            "rotation_deg": round(float(rot), 3),
            "scale_x": round(sx, 4),
            "scale_y": round(sy, 4),
            "shear": round(float(shear), 4),
            "translation": [round(float(t[0]), 2), round(float(t[1]), 2)],
            "inlier_count": inlier_count,
            "mean_reproj_err_px": round(float(np.mean(err)), 3),
            "p95_reproj_err_px": round(float(np.percentile(err, 95)), 3),
        })
    except Exception as exc:
        metrics["error"] = str(exc)
    return metrics


def analyze(payload: Dict[str, Any], template_path: Optional[Path] = None, calibrate: bool = False) -> Dict[str, Any]:
    rows = max(1, int(EXPECTED_ROWS))
    cols = max(1, int(EXPECTED_COLS))
    expected_count = int(EXPECTED_COUNT or rows * cols)
    template_path = template_path or Path(TEMPLATE_PATH)

    items, raw_count, width, height = parse_cell_items(payload)
    grid_ok, grid_msg, grid_items = assign_grid_by_sort(items, rows, cols)
    geom = grid_geometry(grid_items if grid_ok else items, rows, cols) if items else {}

    if calibrate:
        if len(items) != rows * cols or not grid_ok:
            return {
                "ok": True,
                "final_result": "ERROR",
                "reason": "CALIBRATE_GRID_FAILED",
                "error": grid_msg,
                "raw_prediction_count": raw_count,
                "valid_cell_count": len(items),
                "expected_count": expected_count,
                "expected_rows": rows,
                "expected_cols": cols,
                "image_width": width,
                "image_height": height,
                "cells": grid_items,
            }
        tpl = make_template(grid_items, rows, cols, width, height)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(json.dumps(tpl, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "ok": True,
            "final_result": "OK",
            "reason": "CALIBRATED",
            "template_path": str(template_path),
            "raw_prediction_count": raw_count,
            "valid_cell_count": len(items),
            "expected_count": expected_count,
            "expected_rows": rows,
            "expected_cols": cols,
            "image_width": width,
            "image_height": height,
            "geometry": geom,
            "cells": grid_items,
        }

    template: Optional[Dict[str, Any]] = None
    template_loaded = False
    template_error = ""
    try:
        template = load_template(template_path)
        template_loaded = template is not None
    except Exception as exc:
        template_error = str(exc)
        template_loaded = False

    final_result = "OK"
    reason = "NONE"
    error_code = 0
    metrics: Dict[str, Any] = {}
    slot_results: List[Dict[str, Any]] = []

    if STRICT_COUNT == 1 and len(items) != expected_count:
        final_result = "NG"
        reason = "COUNT_MISMATCH"
    elif not grid_ok:
        final_result = "NG"
        reason = "GRID_ASSIGN_FAILED"
    elif not template_loaded or template is None:
        final_result = "ERROR"
        reason = "TEMPLATE_MISSING"
        error_code = 401
    else:
        tpl_slots = template_cells_by_slot(template)
        cur_slots = points_by_slot(grid_items)
        center_errs: List[float] = []
        width_ratios: List[float] = []
        height_ratios: List[float] = []
        area_ratios: List[float] = []
        template_pts: List[Tuple[float, float]] = []
        current_pts: List[Tuple[float, float]] = []
        bad_size_count = 0
        missing_slots = []
        edge_errs: List[float] = []

        for sid in range(rows * cols):
            cur = cur_slots.get(sid)
            tpl = tpl_slots.get(sid)
            if cur is None or tpl is None:
                missing_slots.append(sid)
                slot_results.append({"slot_id": sid, "status": "missing"})
                continue
            dx = float(cur["cx"]) - float(tpl["cx"])
            dy = float(cur["cy"]) - float(tpl["cy"])
            err = math.hypot(dx, dy)
            center_errs.append(err)
            row_id = sid // cols
            col_id = sid % cols
            if row_id == 0 or row_id == rows - 1 or col_id == 0 or col_id == cols - 1:
                edge_errs.append(err)
            template_pts.append((float(tpl["cx"]), float(tpl["cy"])))
            current_pts.append((float(cur["cx"]), float(cur["cy"])))

            wr = float(cur["w"]) / max(1e-6, float(tpl.get("w", 1.0)))
            hr = float(cur["h"]) / max(1e-6, float(tpl.get("h", 1.0)))
            ar = float(cur["area"]) / max(1e-6, float(tpl.get("area", 1.0)))
            width_ratios.append(wr)
            height_ratios.append(hr)
            area_ratios.append(ar)
            size_bad = wr < MIN_BOX_SIZE_RATIO or wr > MAX_BOX_SIZE_RATIO or hr < MIN_BOX_SIZE_RATIO or hr > MAX_BOX_SIZE_RATIO or ar < MIN_BOX_SIZE_RATIO or ar > MAX_BOX_SIZE_RATIO
            if size_bad:
                bad_size_count += 1
            slot_results.append({
                "slot_id": sid,
                "row_id": int(cur.get("row_id", -1)),
                "col_id": int(cur.get("col_id", -1)),
                "status": "size_bad" if size_bad else "ok",
                "center_error_px": round(err, 2),
                "dx": round(dx, 2),
                "dy": round(dy, 2),
                "w_ratio": round(wr, 3),
                "h_ratio": round(hr, 3),
                "area_ratio": round(ar, 3),
            })

        tpl_geom = template.get("geometry") if isinstance(template.get("geometry"), dict) else {}
        row_diff = angle_diff_deg(geom.get("mean_row_angle_deg"), tpl_geom.get("mean_row_angle_deg"))
        col_diff = angle_diff_deg(geom.get("mean_col_angle_deg_from_vertical"), tpl_geom.get("mean_col_angle_deg_from_vertical"))
        row_angle_diffs = angle_diff_list(geom.get("row_angles_deg"), tpl_geom.get("row_angles_deg"))
        max_row_angle_diff = max((abs(x) for x in row_angle_diffs), default=0.0)
        row_angle_std_diff = abs_diff_float(geom.get("std_row_angle_deg"), tpl_geom.get("std_row_angle_deg"))
        grid_center = geom.get("grid_center") if isinstance(geom.get("grid_center"), list) else [0, 0]
        tpl_center = tpl_geom.get("grid_center") if isinstance(tpl_geom.get("grid_center"), list) else [0, 0]
        center_offset = math.hypot(float(grid_center[0]) - float(tpl_center[0]), float(grid_center[1]) - float(tpl_center[1]))
        affine = estimate_affine_metrics(template_pts, current_pts)

        metrics = {
            "template_loaded": True,
            "template_path": str(template_path),
            "matched_count": len(center_errs),
            "missing_slots": missing_slots,
            "mean_center_error_px": round(float(np.mean(center_errs)), 3) if center_errs else 9999.0,
            "p95_center_error_px": round(float(np.percentile(center_errs, 95)), 3) if center_errs else 9999.0,
            "max_center_error_px": round(float(np.max(center_errs)), 3) if center_errs else 9999.0,
            "edge_cell_max_error_px": round(float(np.max(edge_errs)), 3) if edge_errs else 0.0,
            "edge_cell_mean_error_px": round(float(np.mean(edge_errs)), 3) if edge_errs else 0.0,
            "grid_center_offset_px": round(float(center_offset), 3),
            "row_angle_diff_deg": round(float(row_diff), 3) if row_diff is not None else None,
            "row_angle_diffs_deg": [round(float(x), 3) for x in row_angle_diffs],
            "max_row_angle_diff_deg": round(float(max_row_angle_diff), 3),
            "row_angle_std_diff_deg": round(float(row_angle_std_diff), 3) if row_angle_std_diff is not None else None,
            "col_angle_diff_deg": round(float(col_diff), 3) if col_diff is not None else None,
            "bad_size_count": bad_size_count,
            "width_ratio_mean": round(float(np.mean(width_ratios)), 4) if width_ratios else 0,
            "height_ratio_mean": round(float(np.mean(height_ratios)), 4) if height_ratios else 0,
            "area_ratio_mean": round(float(np.mean(area_ratios)), 4) if area_ratios else 0,
            "affine": affine,
        }

        if missing_slots:
            final_result = "NG"
            reason = "SLOT_MISSING"
        elif metrics["mean_center_error_px"] > MAX_MEAN_CENTER_ERR_PX:
            final_result = "NG"
            reason = "MEAN_CENTER_ERROR"
        elif metrics["p95_center_error_px"] > MAX_P95_CENTER_ERR_PX:
            final_result = "NG"
            reason = "P95_CENTER_ERROR"
        elif MAX_ROW_ANGLE_DIFF_MAX_DEG > 0 and metrics["max_row_angle_diff_deg"] > MAX_ROW_ANGLE_DIFF_MAX_DEG:
            final_result = "NG"
            reason = "ROW_ANGLE_MAX_DIFF"
        elif MAX_ROW_ANGLE_STD_DIFF_DEG > 0 and metrics.get("row_angle_std_diff_deg") is not None and float(metrics["row_angle_std_diff_deg"]) > MAX_ROW_ANGLE_STD_DIFF_DEG:
            final_result = "NG"
            reason = "ROW_ANGLE_STD_DIFF"
        elif MAX_EDGE_CELL_ERR_PX > 0 and metrics["edge_cell_max_error_px"] > MAX_EDGE_CELL_ERR_PX:
            final_result = "NG"
            reason = "EDGE_CELL_ERROR"
        elif MAX_CENTER_ERR_PX > 0 and metrics["max_center_error_px"] > MAX_CENTER_ERR_PX:
            final_result = "NG"
            reason = "MAX_CENTER_ERROR"
        elif metrics["grid_center_offset_px"] > MAX_GRID_CENTER_OFFSET_PX:
            final_result = "NG"
            reason = "GRID_CENTER_OFFSET"
        elif row_diff is not None and abs(row_diff) > MAX_ROW_ANGLE_DIFF_DEG:
            final_result = "NG"
            reason = "ROW_ANGLE_DIFF"
        elif col_diff is not None and abs(col_diff) > MAX_COL_ANGLE_DIFF_DEG:
            final_result = "NG"
            reason = "COL_ANGLE_DIFF"
        elif affine.get("ok") and abs(float(affine.get("rotation_deg", 0.0))) > MAX_AFFINE_ROT_DEG:
            final_result = "NG"
            reason = "AFFINE_ROTATION"
        elif affine.get("ok") and float(affine.get("shear", 0.0)) > MAX_AFFINE_SHEAR:
            final_result = "NG"
            reason = "AFFINE_SHEAR"
        elif ENABLE_SIZE_CHECK == 1 and bad_size_count > MAX_BAD_SIZE_COUNT:
            final_result = "NG"
            reason = "BOX_SIZE_ANOMALY"

    return {
        "ok": True,
        "final_result": final_result,
        "reason": reason,
        "error_code": error_code,
        "raw_prediction_count": raw_count,
        "valid_cell_count": len(items),
        "expected_count": expected_count,
        "expected_rows": rows,
        "expected_cols": cols,
        "image_width": width,
        "image_height": height,
        "template_loaded": template_loaded,
        "template_path": str(template_path),
        "template_error": template_error,
        "grid_assign_ok": grid_ok,
        "grid_assign_msg": grid_msg,
        "geometry": geom,
        "metrics": metrics,
        "slots": slot_results,
        "cells": grid_items if grid_ok else items,
    }


def draw_overlay(image_bytes: bytes, result: Dict[str, Any], out_path: Path) -> bool:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    final_result = str(result.get("final_result", ""))
    color_ok = (60, 220, 80)
    color_ng = (0, 0, 255)
    color = color_ok if final_result == "OK" else color_ng

    for cell in result.get("cells", []) or []:
        if not isinstance(cell, dict):
            continue
        bbox = cell.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cx = int(round(float(cell.get("cx", 0))))
        cy = int(round(float(cell.get("cy", 0))))
        cv2.circle(img, (cx, cy), 2, (0, 255, 255), -1)
        sid = cell.get("slot_id")
        if sid is not None:
            cv2.putText(img, str(sid), (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, f"{final_result} {result.get('reason','')}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    if metrics:
        text = f"count={result.get('valid_cell_count')}/{result.get('expected_count')} mean={metrics.get('mean_center_error_px')} p95={metrics.get('p95_center_error_px')} max={metrics.get('max_center_error_px')} edge={metrics.get('edge_cell_max_error_px')} rowMax={metrics.get('max_row_angle_diff_deg')} rowStdD={metrics.get('row_angle_std_diff_deg')}"
        cv2.putText(img, text[:120], (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), img))


def run_once(calibrate: bool = False, template_path: Optional[Path] = None, save_dir: Optional[Path] = None, print_payload: bool = False) -> Dict[str, Any]:
    print(f"[INFO] fetch RGB snapshot: {SNAPSHOT_URL}")
    rgb_bytes = http_get_bytes(SNAPSHOT_URL, HTTP_TIMEOUT_S)
    if not rgb_bytes:
        raise RuntimeError("snapshot is empty")
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "rgb.jpg").write_bytes(rgb_bytes)

    print(f"[INFO] post C++ infer: {INFER_URL}, bytes={len(rgb_bytes)}")
    payload = post_multipart_image(INFER_URL, rgb_bytes, timeout_s=HTTP_TIMEOUT_S)
    if save_dir:
        (save_dir / "infer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if print_payload:
        print("[RAW_INFER]", json.dumps(payload, ensure_ascii=False, indent=2))

    result = analyze(payload, template_path=template_path, calibrate=calibrate)
    if save_dir:
        (save_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            draw_overlay(rgb_bytes, result, save_dir / "overlay.jpg")
        except Exception as exc:
            print(f"[WARN] draw overlay failed: {exc}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot partition cell grid check using existing HP60C snapshot and C++ infer API.")
    parser.add_argument("--save-dir", default="", help="Optional directory to save rgb.jpg, infer.json, result.json and overlay.jpg.")
    parser.add_argument("--template", default="", help="Template JSON path. Default from VISIONOPS_PARTITION_TEMPLATE_PATH.")
    parser.add_argument("--calibrate", action="store_true", help="Use current normal image to create/update the template JSON.")
    parser.add_argument("--print-payload", action="store_true", help="Print raw infer payload too.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir).resolve() if args.save_dir else None
    template_path = Path(args.template).resolve() if args.template else Path(TEMPLATE_PATH)

    result = run_once(calibrate=args.calibrate, template_path=template_path, save_dir=save_dir, print_payload=args.print_payload)
    print("[RESULT]", json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f"[SUMMARY] final={result.get('final_result')} reason={result.get('reason')} "
        f"cells={result.get('valid_cell_count')}/{result.get('expected_count')} "
        f"template_loaded={result.get('template_loaded')} grid_assign={result.get('grid_assign_ok')}"
    )
    return 0 if result.get("final_result") == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
