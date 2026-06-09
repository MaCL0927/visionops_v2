#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP proxy client for the C++ HP60C ROS1 bridge.

This module intentionally does NOT import rospy/sensor_msgs/cv_bridge.
Collector only talks to the bridge over HTTP; ROS image subscription is done in
an independent C++ process.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


class Ros1BridgeError(RuntimeError):
    pass


def _base_url() -> str:
    return os.environ.get("VISIONOPS_CPP_ROS1_BRIDGE_URL") or os.environ.get(
        "VISIONOPS_HP60C_ROS1_BRIDGE_URL", "http://127.0.0.1:18181"
    )


def _url(path: str, query_string: Optional[bytes] = None) -> str:
    if not path.startswith("/"):
        path = "/" + path
    url = _base_url().rstrip("/") + path
    if query_string:
        qs = query_string.decode("utf-8", errors="ignore")
        if qs:
            url += "?" + qs
    return url


def request_json(path: str, *, method: str = "GET", timeout: float = 3.0, query_string: Optional[bytes] = None) -> Dict[str, Any]:
    req = urllib.request.Request(
        _url(path, query_string=query_string),
        method=method.upper(),
        headers={"Accept": "application/json", "User-Agent": "VisionOps-Collector-Ros1BridgeProxy/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {"data": data}
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        raise Ros1BridgeError(f"ROS1 bridge HTTP {exc.code} for {method} {path}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise Ros1BridgeError(f"ROS1 bridge unavailable for {method} {path}: {exc.reason}") from exc
    except Exception as exc:
        raise Ros1BridgeError(f"ROS1 bridge request failed for {method} {path}: {exc}") from exc


def get_json(path: str, *, query_string: Optional[bytes] = None) -> Dict[str, Any]:
    return request_json(path, method="GET", query_string=query_string)


def post_json(path: str, *, query_string: Optional[bytes] = None) -> Dict[str, Any]:
    return request_json(path, method="POST", query_string=query_string)


def get_binary(path: str, *, timeout: float = 5.0) -> Tuple[bytes, str]:
    req = urllib.request.Request(
        _url(path),
        method="GET",
        headers={"Accept": "image/jpeg,*/*;q=0.5", "User-Agent": "VisionOps-Collector-Ros1BridgeProxy/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.headers.get("content-type") or "application/octet-stream"
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        raise Ros1BridgeError(f"ROS1 bridge image HTTP {exc.code} for GET {path}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise Ros1BridgeError(f"ROS1 bridge image unavailable for GET {path}: {exc.reason}") from exc
    except Exception as exc:
        raise Ros1BridgeError(f"ROS1 bridge image request failed for GET {path}: {exc}") from exc
