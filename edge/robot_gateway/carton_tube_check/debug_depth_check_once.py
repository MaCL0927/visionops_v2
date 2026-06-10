#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps carton tube stage2 one-shot debug script.

Purpose:
  1. Fetch one HP60C RGB snapshot from the ROS1 C++ bridge.
  2. POST it to the existing C++ OBB /api/cpp/infer endpoint.
  3. Fetch one 16-bit depth PNG from the ROS1 C++ bridge.
  4. Use stand OBB centers to sample depth and check whether any tube is much higher.

This script does NOT start Modbus and does NOT modify tube_station.
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
DEFAULT_ENV = THIS_DIR / "carton_tube_check.env"


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


load_env_file(Path(os.environ.get("VISIONOPS_CARTON_TUBE_ENV", str(DEFAULT_ENV))))

SNAPSHOT_URL = getenv_str("VISIONOPS_CARTON_TUBE_SNAPSHOT_URL", "http://127.0.0.1:18181/stream/snapshot.jpg")
DEPTH_URL = getenv_str("VISIONOPS_CARTON_TUBE_DEPTH_URL", "http://127.0.0.1:18181/stream/depth.png")
DEPTH_META_URL = getenv_str("VISIONOPS_CARTON_TUBE_DEPTH_META_URL", "http://127.0.0.1:18181/stream/depth_meta")
INFER_URL = getenv_str("VISIONOPS_CARTON_TUBE_INFER_URL", "http://127.0.0.1:8090/api/cpp/infer")
STAND_CLASS_IDS = parse_int_set(getenv_str("VISIONOPS_CARTON_TUBE_STAND_CLASS_IDS", "0"), [0])
LYING_CLASS_IDS = parse_int_set(getenv_str("VISIONOPS_CARTON_TUBE_LYING_CLASS_IDS", "1"), [1])
STAND_NAMES = parse_name_set(getenv_str("VISIONOPS_CARTON_TUBE_STAND_NAMES", "stand"), ["stand"])
LYING_NAMES = parse_name_set(getenv_str("VISIONOPS_CARTON_TUBE_LYING_NAMES", "lying"), ["lying"])
MIN_CONF = getenv_float("VISIONOPS_CARTON_TUBE_MIN_CONF", 0.30)
MIN_STAND_COUNT = getenv_int("VISIONOPS_CARTON_TUBE_MIN_STAND_COUNT", 1)
DEPTH_ROI_RADIUS_PX = getenv_int("VISIONOPS_CARTON_TUBE_DEPTH_ROI_RADIUS_PX", 12)
DEPTH_PERCENTILE = getenv_float("VISIONOPS_CARTON_TUBE_DEPTH_PERCENTILE", 50.0)
MIN_VALID_DEPTH_PIXELS = getenv_int("VISIONOPS_CARTON_TUBE_MIN_VALID_DEPTH_PIXELS", 30)
MIN_DEPTH_MM = getenv_int("VISIONOPS_CARTON_TUBE_MIN_DEPTH_MM", 100)
MAX_DEPTH_MM = getenv_int("VISIONOPS_CARTON_TUBE_MAX_DEPTH_MM", 3000)
NORMAL_DEPTH_MM = getenv_float("VISIONOPS_CARTON_TUBE_NORMAL_DEPTH_MM", 0.0)
BASELINE_MODE = getenv_str("VISIONOPS_CARTON_TUBE_BASELINE_MODE", "row_median").strip().lower()
EXPECTED_ROWS = getenv_int("VISIONOPS_CARTON_TUBE_EXPECTED_ROWS", 5)
EXPECTED_COLS = getenv_int("VISIONOPS_CARTON_TUBE_EXPECTED_COLS", 8)
HEIGHT_THRESHOLD_MM = getenv_float("VISIONOPS_CARTON_TUBE_HEIGHT_THRESHOLD_MM", 35.0)
HTTP_TIMEOUT_S = getenv_int("VISIONOPS_CARTON_TUBE_HTTP_TIMEOUT_MS", 5000) / 1000.0


def now_ms() -> int:
    return int(time.time() * 1000)


def http_get_bytes(url: str, timeout_s: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "VisionOps-CartonTubeDebug/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        code = getattr(resp, "status", 200)
        if code < 200 or code >= 300:
            raise RuntimeError(f"GET {url} HTTP {code}")
        return resp.read()


def http_get_json(url: str, timeout_s: float) -> Dict[str, Any]:
    raw = http_get_bytes(url, timeout_s).decode("utf-8", errors="replace")
    try:
        obj = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON from {url}: {exc}: {raw[:200]}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"JSON from {url} is not an object")
    return obj


def post_multipart_image(url: str, image_bytes: bytes, filename: str = "hp60c_trigger.jpg", timeout_s: float = 5.0) -> Dict[str, Any]:
    boundary = "----VisionOpsCartonTubeBoundary" + str(now_ms())
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
            "User-Agent": "VisionOps-CartonTubeDebug/1.0",
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


def class_role(pred: Dict[str, Any]) -> str:
    cid = pred_class_id(pred)
    name = pred_name(pred)
    if cid in STAND_CLASS_IDS or name in STAND_NAMES:
        return "stand"
    if cid in LYING_CLASS_IDS or name in LYING_NAMES:
        return "lying"
    return "unknown"


def pred_center(pred: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    center = pred.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            return float(center[0]), float(center[1])
        except Exception:
            pass
    keys = (("center_x", "center_y"), ("cx", "cy"))
    for kx, ky in keys:
        if kx in pred and ky in pred:
            try:
                return float(pred[kx]), float(pred[ky])
            except Exception:
                pass
    obb = pred.get("obb")
    if isinstance(obb, dict) and isinstance(obb.get("points"), list):
        pts = obb.get("points") or []
        xs: List[float] = []
        ys: List[float] = []
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    xs.append(float(p[0]))
                    ys.append(float(p[1]))
                except Exception:
                    pass
        if xs and ys:
            return sum(xs) / len(xs), sum(ys) / len(ys)
    bbox = pred.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [float(x) for x in bbox[:4]]
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        except Exception:
            pass
    return None


def decode_depth_png(depth_bytes: bytes) -> "np.ndarray":
    arr = np.frombuffer(depth_bytes, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("failed to decode depth PNG")
    if depth.ndim != 2:
        raise RuntimeError(f"depth PNG should be single-channel, got shape={depth.shape}")
    if depth.dtype != np.uint16:
        raise RuntimeError(f"depth PNG should be uint16 16UC1-mm, got dtype={depth.dtype}")
    return depth


def sample_depth_mm(depth: "np.ndarray", cx: float, cy: float, radius_px: int) -> Dict[str, Any]:
    h, w = depth.shape[:2]
    x = int(round(cx))
    y = int(round(cy))
    r = max(1, int(radius_px))
    x1 = max(0, x - r)
    x2 = min(w, x + r + 1)
    y1 = max(0, y - r)
    y2 = min(h, y + r + 1)
    roi = depth[y1:y2, x1:x2]
    valid = roi[(roi >= MIN_DEPTH_MM) & (roi <= MAX_DEPTH_MM)]
    out: Dict[str, Any] = {
        "cx": round(float(cx), 2),
        "cy": round(float(cy), 2),
        "roi": [int(x1), int(y1), int(x2), int(y2)],
        "valid_pixels": int(valid.size),
        "depth_mm": None,
    }
    if valid.size < MIN_VALID_DEPTH_PIXELS:
        out["error"] = "not_enough_valid_depth_pixels"
        return out
    depth_mm = float(np.percentile(valid.astype(np.float32), DEPTH_PERCENTILE))
    out["depth_mm"] = round(depth_mm, 2)
    out["depth_min_mm"] = int(valid.min())
    out["depth_max_mm"] = int(valid.max())
    return out




def kmeans_1d(values: List[float], k: int, max_iter: int = 50) -> Tuple[List[int], List[float]]:
    """Small dependency-free 1D k-means used for row/column grouping."""
    n = len(values)
    if n == 0 or k <= 0:
        return [], []
    k = min(k, n)
    vmin = min(values)
    vmax = max(values)
    if k == 1 or abs(vmax - vmin) < 1e-6:
        return [0 for _ in values], [float(sum(values) / n)]

    centers = [vmin + (vmax - vmin) * (i + 0.5) / k for i in range(k)]
    labels = [0 for _ in values]
    for _ in range(max_iter):
        changed = False
        for i, v in enumerate(values):
            label = min(range(k), key=lambda j: abs(v - centers[j]))
            if label != labels[i]:
                labels[i] = label
                changed = True
        new_centers = centers[:]
        for j in range(k):
            bucket = [values[i] for i, lb in enumerate(labels) if lb == j]
            if bucket:
                new_centers[j] = float(sum(bucket) / len(bucket))
        if max(abs(new_centers[j] - centers[j]) for j in range(k)) < 1e-4 and not changed:
            centers = new_centers
            break
        centers = new_centers

    # Re-map labels so 0..k-1 are ordered by coordinate from small to large.
    order = sorted(range(k), key=lambda j: centers[j])
    remap = {old: new for new, old in enumerate(order)}
    labels = [remap[lb] for lb in labels]
    centers = [centers[old] for old in order]
    return labels, centers


def assign_grid_indices(items: List[Dict[str, Any]], rows: int, cols: int) -> Tuple[List[float], List[float]]:
    """Assign row_id/col_id to each item according to center y/x.

    This is only for debugging and matrix display. It does not require every slot to be detected.
    Missing tubes remain null in the matrix.
    """
    if not items:
        return [], []

    rows = max(1, int(rows))
    cols = max(1, int(cols))
    ys = [float(x.get("cy", x.get("center", [0, 0])[1])) for x in items]
    xs = [float(x.get("cx", x.get("center", [0, 0])[0])) for x in items]

    row_labels, row_centers = kmeans_1d(ys, min(rows, len(items)))
    col_labels, col_centers = kmeans_1d(xs, min(cols, len(items)))

    for item, r, c in zip(items, row_labels, col_labels):
        item["row_id"] = int(r)
        item["col_id"] = int(c)
        item["slot_id"] = int(r) * cols + int(c)

    return row_centers, col_centers


def empty_matrix(rows: int, cols: int, value: Any = None) -> List[List[Any]]:
    return [[value for _ in range(cols)] for _ in range(rows)]


def put_matrix_value(mat: List[List[Any]], row: Any, col: Any, value: Any, conflicts: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    try:
        r = int(row)
        c = int(col)
    except Exception:
        return
    if r < 0 or c < 0 or r >= len(mat) or c >= len(mat[0]):
        return
    if mat[r][c] is not None:
        conflicts.append({
            "row_id": r,
            "col_id": c,
            "old_value": mat[r][c],
            "new_value": value,
            "new_idx": item.get("idx"),
        })
    mat[r][c] = value


def build_matrices(items: List[Dict[str, Any]], rows: int, cols: int) -> Dict[str, Any]:
    depth_matrix = empty_matrix(rows, cols, None)
    baseline_matrix = empty_matrix(rows, cols, None)
    diff_matrix = empty_matrix(rows, cols, None)
    high_matrix = empty_matrix(rows, cols, None)
    conf_matrix = empty_matrix(rows, cols, None)
    idx_matrix = empty_matrix(rows, cols, None)
    conflicts: List[Dict[str, Any]] = []

    for item in items:
        r = item.get("row_id")
        c = item.get("col_id")
        put_matrix_value(depth_matrix, r, c, item.get("depth_mm"), conflicts, item)
        put_matrix_value(baseline_matrix, r, c, item.get("baseline_depth_mm"), conflicts, item)
        put_matrix_value(diff_matrix, r, c, item.get("height_diff_mm"), conflicts, item)
        put_matrix_value(high_matrix, r, c, item.get("height_high"), conflicts, item)
        put_matrix_value(conf_matrix, r, c, item.get("confidence"), conflicts, item)
        put_matrix_value(idx_matrix, r, c, item.get("idx"), conflicts, item)

    return {
        "depth_mm": depth_matrix,
        "baseline_depth_mm": baseline_matrix,
        "height_diff_mm": diff_matrix,
        "height_high": high_matrix,
        "confidence": conf_matrix,
        "idx": idx_matrix,
        "conflicts": conflicts,
    }


def format_matrix(mat: List[List[Any]], none_text: str = "----", width: int = 7, precision: int = 1) -> str:
    lines: List[str] = []
    for r, row in enumerate(mat):
        cells: List[str] = []
        for v in row:
            if v is None:
                cells.append(none_text.rjust(width))
            elif isinstance(v, bool):
                cells.append(("HIGH" if v else "ok").rjust(width))
            elif isinstance(v, (int, float)):
                if isinstance(v, float):
                    cells.append((f"{v:.{precision}f}").rjust(width))
                else:
                    cells.append(str(v).rjust(width))
            else:
                cells.append(str(v).rjust(width))
        lines.append(f"row{r:02d}: " + " ".join(cells))
    return "\n".join(lines)

def analyze(payload: Dict[str, Any], depth: "np.ndarray") -> Dict[str, Any]:
    preds = find_predictions(payload)
    width, height = image_size_from_payload(payload)
    valid_preds: List[Dict[str, Any]] = []
    stand_items: List[Dict[str, Any]] = []
    lying_items: List[Dict[str, Any]] = []

    for idx, pred in enumerate(preds):
        conf = pred_conf(pred)
        role = class_role(pred)
        center = pred_center(pred)
        if role == "unknown" or conf < MIN_CONF or center is None:
            continue
        item = {
            "idx": idx,
            "role": role,
            "class_id": pred_class_id(pred),
            "class_name": pred_name(pred),
            "confidence": round(conf, 4),
            "center": [round(center[0], 2), round(center[1], 2)],
        }
        valid_preds.append(item)
        if role == "stand":
            stand_items.append(item)
        elif role == "lying":
            lying_items.append(item)

    sampled: List[Dict[str, Any]] = []
    for item in stand_items:
        cx, cy = item["center"]
        sd = sample_depth_mm(depth, cx, cy, DEPTH_ROI_RADIUS_PX)
        item_with_depth = dict(item)
        item_with_depth.update(sd)
        sampled.append(item_with_depth)

    # Assign 5x8 grid indices for row-wise comparison and matrix display.
    rows = max(1, EXPECTED_ROWS)
    cols = max(1, EXPECTED_COLS)
    row_centers, col_centers = assign_grid_indices(sampled, rows, cols)

    valid_depths = [float(x["depth_mm"]) for x in sampled if x.get("depth_mm") is not None]
    baseline_mode_requested = BASELINE_MODE or "row_median"

    row_baselines: Dict[int, float] = {}
    if baseline_mode_requested == "row_median":
        for r in range(rows):
            vals = [float(x["depth_mm"]) for x in sampled if x.get("row_id") == r and x.get("depth_mm") is not None]
            if vals:
                row_baselines[r] = float(np.median(np.array(vals, dtype=np.float32)))
        baseline_depth = float(np.median(np.array(list(row_baselines.values()), dtype=np.float32))) if row_baselines else 0.0
        baseline_mode = "row_median"
    elif NORMAL_DEPTH_MM > 0 or baseline_mode_requested in {"fixed", "fixed_env", "normal_depth"}:
        baseline_depth = float(NORMAL_DEPTH_MM) if NORMAL_DEPTH_MM > 0 else 0.0
        baseline_mode = "fixed_env" if baseline_depth > 0 else "invalid_fixed_env"
    elif valid_depths:
        baseline_depth = float(np.median(np.array(valid_depths, dtype=np.float32)))
        baseline_mode = "current_frame_median"
    else:
        baseline_depth = 0.0
        baseline_mode = "invalid"

    high_items: List[Dict[str, Any]] = []
    max_height_diff = 0.0
    for item in sampled:
        item_baseline = baseline_depth
        if baseline_mode == "row_median":
            try:
                item_baseline = row_baselines.get(int(item.get("row_id", -1)), 0.0)
            except Exception:
                item_baseline = 0.0
        item["baseline_depth_mm"] = round(float(item_baseline), 2) if item_baseline > 0 else None

        if item.get("depth_mm") is None or item_baseline <= 0:
            item["height_diff_mm"] = None
            item["height_high"] = False
            continue
        diff = float(item_baseline) - float(item["depth_mm"])
        max_height_diff = max(max_height_diff, diff)
        item["height_diff_mm"] = round(diff, 2)
        item["height_high"] = bool(diff > HEIGHT_THRESHOLD_MM)
        if item["height_high"]:
            high_items.append(item)

    grid = build_matrices(sampled, rows, cols)

    if lying_items:
        final_result = "NG"
        reason = "LYING_DETECTED"
    elif len(stand_items) < MIN_STAND_COUNT:
        final_result = "NG"
        reason = "STAND_COUNT_LOW"
    elif not valid_depths:
        final_result = "NG"
        reason = "DEPTH_INVALID"
    elif high_items:
        final_result = "NG"
        reason = "HEIGHT_HIGH"
    else:
        final_result = "OK"
        reason = "NONE"

    return {
        "ok": True,
        "final_result": final_result,
        "reason": reason,
        "stand_count": len(stand_items),
        "lying_count": len(lying_items),
        "valid_prediction_count": len(valid_preds),
        "raw_prediction_count": len(preds),
        "image_width": width,
        "image_height": height,
        "depth_width": int(depth.shape[1]),
        "depth_height": int(depth.shape[0]),
        "baseline_depth_mm": round(baseline_depth, 2),
        "baseline_mode": baseline_mode,
        "baseline_mode_requested": baseline_mode_requested,
        "row_baseline_depth_mm": {str(k): round(v, 2) for k, v in sorted(row_baselines.items())},
        "height_threshold_mm": HEIGHT_THRESHOLD_MM,
        "expected_rows": rows,
        "expected_cols": cols,
        "row_centers_y": [round(float(x), 2) for x in row_centers],
        "col_centers_x": [round(float(x), 2) for x in col_centers],
        "max_height_diff_mm": round(max_height_diff, 2),
        "high_count": len(high_items),
        "grid": grid,
        "tubes": sampled,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot RGB OBB + depth height debug check for carton tubes.")
    parser.add_argument("--save-dir", default="", help="Optional directory to save rgb.jpg, depth.png, infer.json and result.json.")
    parser.add_argument("--print-payload", action="store_true", help="Print raw infer payload too.")
    args = parser.parse_args()

    save_dir: Optional[Path] = Path(args.save_dir).resolve() if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] depth_meta: {DEPTH_META_URL}")
    try:
        meta = http_get_json(DEPTH_META_URL, HTTP_TIMEOUT_S)
        print("[INFO] bridge meta:", json.dumps(meta, ensure_ascii=False))
    except Exception as exc:
        print(f"[WARN] failed to get depth meta: {exc}")

    print(f"[INFO] fetch RGB snapshot: {SNAPSHOT_URL}")
    rgb_bytes = http_get_bytes(SNAPSHOT_URL, HTTP_TIMEOUT_S)
    if save_dir:
        (save_dir / "rgb.jpg").write_bytes(rgb_bytes)

    print(f"[INFO] post OBB infer: {INFER_URL}, bytes={len(rgb_bytes)}")
    payload = post_multipart_image(INFER_URL, rgb_bytes, timeout_s=HTTP_TIMEOUT_S)
    if save_dir:
        (save_dir / "infer.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] fetch depth PNG: {DEPTH_URL}")
    depth_bytes = http_get_bytes(DEPTH_URL, HTTP_TIMEOUT_S)
    if save_dir:
        (save_dir / "depth.png").write_bytes(depth_bytes)
    depth = decode_depth_png(depth_bytes)

    result = analyze(payload, depth)
    if save_dir:
        (save_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print_payload:
        print("[RAW_INFER]", json.dumps(payload, ensure_ascii=False, indent=2))
    print("[RESULT]", json.dumps(result, ensure_ascii=False, indent=2))
    grid = result.get("grid") if isinstance(result.get("grid"), dict) else {}
    if grid:
        print("[MATRIX] depth_mm 5x8 / detected slots; ---- means missing or not detected")
        print(format_matrix(grid.get("depth_mm") or []))
        print("[MATRIX] baseline_depth_mm 5x8")
        print(format_matrix(grid.get("baseline_depth_mm") or []))
        print("[MATRIX] height_diff_mm 5x8 = baseline - current_depth; positive means closer/higher")
        print(format_matrix(grid.get("height_diff_mm") or []))
        print("[MATRIX] height_high 5x8")
        print(format_matrix(grid.get("height_high") or [], width=7))
    print(
        f"[SUMMARY] final={result['final_result']} reason={result['reason']} "
        f"stand={result['stand_count']} lying={result['lying_count']} "
        f"baseline_mode={result['baseline_mode']} baseline={result['baseline_depth_mm']}mm "
        f"high_count={result['high_count']} max_diff={result['max_height_diff_mm']}mm"
    )
    return 0 if result.get("final_result") == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
