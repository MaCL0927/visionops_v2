#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Collector -> C++ inference service client.

v0.5.0 设计原则：
- Collector 只做控制平面代理；
- 不在 Python 中取 RTSP、不解码、不逐帧 JPEG 编码；
- 实时取流、RGA、RKNN、后处理都由 C++ 服务负责。
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from backend.config import (
    CPP_INFERENCE_IMAGE_TIMEOUT_SEC,
    CPP_INFERENCE_TIMEOUT_SEC,
    CPP_INFERENCE_URL,
)

logger = logging.getLogger("visionops.collector.cpp_inference")


class CppInferenceError(RuntimeError):
    """C++ inference service proxy error."""


def _base_url() -> str:
    return str(CPP_INFERENCE_URL or "http://127.0.0.1:18080").rstrip("/")


def _build_url(path: str, query_string: Optional[bytes] = None) -> str:
    if not path.startswith("/"):
        path = "/" + path
    url = _base_url() + path
    if query_string:
        qs = query_string.decode("utf-8", errors="ignore")
        if qs:
            url += "?" + qs
    return url


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def request_json(
    path: str,
    *,
    method: str = "GET",
    timeout: Optional[float] = None,
    query_string: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Request JSON from C++ inference service."""
    url = _build_url(path, query_string=query_string)
    method = method.upper().strip()
    timeout = float(timeout if timeout is not None else CPP_INFERENCE_TIMEOUT_SEC)

    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept": "application/json",
            "User-Agent": "VisionOps-Collector-CppProxy/0.5.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {}
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {"data": data}
            return data

    except urllib.error.HTTPError as exc:
        body = _read_error_body(exc)
        raise CppInferenceError(
            f"C++ service HTTP {exc.code} for {method} {path}: {body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise CppInferenceError(
            f"C++ service unavailable for {method} {path}: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CppInferenceError(
            f"C++ service returned non-JSON response for {method} {path}: {exc}"
        ) from exc
    except Exception as exc:
        raise CppInferenceError(
            f"C++ service request failed for {method} {path}: {exc}"
        ) from exc


def get_json(path: str, *, query_string: Optional[bytes] = None) -> Dict[str, Any]:
    return request_json(path, method="GET", query_string=query_string)


def post_json(path: str, *, query_string: Optional[bytes] = None) -> Dict[str, Any]:
    return request_json(path, method="POST", query_string=query_string)


def get_binary(
    path: str,
    *,
    timeout: Optional[float] = None,
    query_string: Optional[bytes] = None,
) -> Tuple[bytes, str]:
    """Proxy binary payload, mainly snapshot.jpg / annotated.jpg."""
    url = _build_url(path, query_string=query_string)
    timeout = float(timeout if timeout is not None else CPP_INFERENCE_IMAGE_TIMEOUT_SEC)

    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "image/jpeg,application/json;q=0.8,*/*;q=0.5",
            "User-Agent": "VisionOps-Collector-CppProxy/0.5.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            content_type = resp.headers.get("content-type") or "application/octet-stream"
            return data, content_type

    except urllib.error.HTTPError as exc:
        body = _read_error_body(exc)
        raise CppInferenceError(
            f"C++ image endpoint HTTP {exc.code} for GET {path}: {body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise CppInferenceError(
            f"C++ image endpoint unavailable for GET {path}: {exc.reason}"
        ) from exc
    except Exception as exc:
        raise CppInferenceError(
            f"C++ image endpoint request failed for GET {path}: {exc}"
        ) from exc


def service_summary() -> Dict[str, Any]:
    """Small local summary for debugging Collector proxy configuration."""
    return {
        "cpp_service_url": _base_url(),
        "timeout_sec": float(CPP_INFERENCE_TIMEOUT_SEC),
        "image_timeout_sec": float(CPP_INFERENCE_IMAGE_TIMEOUT_SEC),
    }
