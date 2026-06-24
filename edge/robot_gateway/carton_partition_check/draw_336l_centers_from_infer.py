#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-

import argparse
import json
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SNAPSHOT_URL = "http://127.0.0.1:18182/stream/snapshot.jpg"
DEFAULT_INFER_JSON = "/tmp/vision_robot_protocol_latest/coordinate_check/infer.json"
DEFAULT_OUT_DIR = "/tmp/vision_robot_protocol_latest/coordinate_check"


def http_get_bytes(url: str, timeout_s: float = 5.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "VisionOps-CenterOverlay/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError(f"empty response from {url}")
    return data


def decode_jpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("failed to decode RGB snapshot as image")
    return img


def load_infer_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"infer json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_prediction_center(pred: dict):
    """
    支持几种格式：
    1) center_x / center_y
    2) center: [x, y]
    3) bbox: [x1, y1, x2, y2]，fallback 用 bbox 中心
    """
    if "center_x" in pred and "center_y" in pred:
        return float(pred["center_x"]), float(pred["center_y"])

    center = pred.get("center")
    if isinstance(center, list) and len(center) >= 2:
        return float(center[0]), float(center[1])

    bbox = pred.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        x1, y1, x2, y2 = map(float, bbox[:4])
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    return None


def draw_centers(
    img: np.ndarray,
    infer: dict,
    min_conf: float = 0.0,
    draw_index: bool = True,
    point_radius: int = 6,
    point_thickness: int = -1,
) -> tuple[np.ndarray, int]:
    h, w = img.shape[:2]

    infer_w = float(infer.get("image_width") or w)
    infer_h = float(infer.get("image_height") or h)

    sx = w / infer_w if infer_w > 0 else 1.0
    sy = h / infer_h if infer_h > 0 else 1.0

    preds = infer.get("predictions", [])
    if not isinstance(preds, list):
        preds = []

    out = img.copy()
    drawn = 0

    for i, pred in enumerate(preds, start=1):
        if not isinstance(pred, dict):
            continue

        conf = float(pred.get("confidence", 0.0))
        if conf < min_conf:
            continue

        center = get_prediction_center(pred)
        if center is None:
            continue

        x, y = center
        x = int(round(x * sx))
        y = int(round(y * sy))

        if not (0 <= x < w and 0 <= y < h):
            continue

        # 中心点：实心圆 + 十字线，方便现场看清
        # cv2.circle(out, (x, y), point_radius, (0, 0, 255), point_thickness)
        cv2.drawMarker(
            out,
            (x, y),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=5,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        if draw_index:
            label = f"{i}"
            cv2.putText(
                out,
                label,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        drawn += 1

    # 左上角写基本信息
    text = f"centers={drawn}, image={w}x{h}, infer={int(infer_w)}x{int(infer_h)}"
    cv2.putText(
        out,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out, drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-url", default=DEFAULT_SNAPSHOT_URL)
    ap.add_argument("--infer-json", default=DEFAULT_INFER_JSON)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--no-index", action="store_true", help="不显示中心点序号")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_path = Path(args.infer_json)

    print(f"[INFO] fetch RGB snapshot: {args.snapshot_url}")
    rgb_bytes = http_get_bytes(args.snapshot_url, timeout_s=args.timeout)

    raw_img_path = out_dir / "center_points_rgb.jpg"
    raw_img_path.write_bytes(rgb_bytes)

    img = decode_jpeg(rgb_bytes)

    print(f"[INFO] load infer json: {infer_path}")
    infer = load_infer_json(infer_path)

    overlay, drawn = draw_centers(
        img,
        infer,
        min_conf=args.min_conf,
        draw_index=not args.no_index,
    )

    out_path = out_dir / "center_points_overlay.jpg"
    ok = cv2.imwrite(str(out_path), overlay)
    if not ok:
        raise RuntimeError(f"failed to write output image: {out_path}")

    print(f"[OK] saved raw image: {raw_img_path}")
    print(f"[OK] saved overlay:   {out_path}")
    print(f"[OK] drawn centers:   {drawn}")


if __name__ == "__main__":
    main()
