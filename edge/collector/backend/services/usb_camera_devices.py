#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""USB/V4L2 摄像头节点发现。

用于设置页自动列出 /dev/video*，并尽量推荐 Orbbec 深度相机的 RGB 节点。
依赖 v4l2-ctl 时信息更完整；没有 v4l2-ctl 时仍会回退列出 /dev/video*。
"""

import glob
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run(cmd: List[str], timeout: float = 2.0) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except Exception:
        return ""


def _realpath(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except Exception:
        return path


def _stable_path_for(video_node: str) -> str:
    """优先返回 /dev/v4l/by-id 下的稳定路径，避免 /dev/video7 重插后变化。"""
    target = _realpath(video_node)
    candidates = []
    for pattern in ("/dev/v4l/by-id/*", "/dev/v4l/by-path/*"):
        for item in glob.glob(pattern):
            try:
                if _realpath(item) == target:
                    candidates.append(item)
            except Exception:
                pass
    if not candidates:
        return ""
    # by-id 通常比 by-path 更直观；同类里优先 video-index0。
    candidates.sort(key=lambda x: (0 if "/by-id/" in x else 1, 0 if "index0" in x else 1, x))
    return candidates[0]


def _parse_v4l2_all(text: str) -> Dict[str, str]:
    info: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key in {"driver_name", "card_type", "bus_info"}:
            info[key] = value
    return info


def _parse_formats(text: str) -> Dict[str, Any]:
    formats: List[str] = []
    sizes: List[str] = []
    for line in text.splitlines():
        m = re.search(r"'([^']+)'", line)
        if m:
            fmt = m.group(1).strip()
            if fmt and fmt not in formats:
                formats.append(fmt)
        m = re.search(r"Size:\s+Discrete\s+(\d+)x(\d+)", line)
        if m:
            size = f"{m.group(1)}x{m.group(2)}"
            if size not in sizes:
                sizes.append(size)
    return {"formats": formats, "sizes": sizes}


def _try_read_frame(path: str) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                return {"readable": True, "width": int(w), "height": int(h)}
            return {"readable": False}
        finally:
            cap.release()
    except Exception:
        return {"readable": False}


def _score_device(item: Dict[str, Any]) -> int:
    text = " ".join(str(item.get(k, "")) for k in ["path", "stable_path", "name", "driver", "bus_info"]).lower()
    formats = set(str(x).upper() for x in item.get("formats") or [])
    sizes = set(str(x).lower() for x in item.get("sizes") or [])
    score = 0
    if item.get("readable"):
        score += 30
    if any(word in text for word in ["orbbec", "gemini", "astra", "depth"]):
        score += 30
    if formats.intersection({"MJPG", "YUYV", "RGB3", "BGR3"}):
        score += 20
    if "1280x800" in sizes or (item.get("width") == 1280 and item.get("height") == 800):
        score += 20
    if item.get("path", "").endswith("0"):
        score += 1
    return score


def list_usb_camera_devices(probe_read: bool = True) -> Dict[str, Any]:
    nodes = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0))
    items: List[Dict[str, Any]] = []

    for node in nodes:
        item: Dict[str, Any] = {
            "path": node,
            "stable_path": _stable_path_for(node),
        }
        all_text = _run(["v4l2-ctl", "-d", node, "--all"])
        fmt_text = _run(["v4l2-ctl", "-d", node, "--list-formats-ext"])
        info = _parse_v4l2_all(all_text)
        fmt = _parse_formats(fmt_text)

        item["name"] = info.get("card_type") or Path(node).name
        item["driver"] = info.get("driver_name", "")
        item["bus_info"] = info.get("bus_info", "")
        item["formats"] = fmt.get("formats", [])
        item["sizes"] = fmt.get("sizes", [])

        if probe_read:
            item.update(_try_read_frame(node))
        else:
            item["readable"] = None

        item["preferred_path"] = item.get("stable_path") or item["path"]
        item["orbbec"] = any(
            key in " ".join(str(item.get(k, "")) for k in ["stable_path", "name", "driver", "bus_info"]).lower()
            for key in ["orbbec", "gemini", "astra"]
        )
        item["score"] = _score_device(item)
        item["recommended"] = False
        items.append(item)

    items.sort(key=lambda x: (-int(x.get("score") or 0), x.get("path", "")))
    if items:
        items[0]["recommended"] = True

    return {
        "ok": True,
        "items": items,
        "recommended": items[0] if items else None,
        "message": f"found {len(items)} video nodes",
    }
