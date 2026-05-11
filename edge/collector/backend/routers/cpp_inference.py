#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collector API proxy for C++ inference service.

v0.5.0:
- 前端只访问 Collector 的 /api/cpp/*；
- Collector 转发到本机 C++ 服务 http://127.0.0.1:18080；
- Python 不参与 RTSP 拉流、解码、推理、逐帧 JPEG 生成。
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from backend.config import CPP_INFERENCE_ENABLED
from backend.services.cpp_inference_client import (
    CppInferenceError,
    get_binary,
    get_json,
    post_json,
    service_summary,
)

router = APIRouter(prefix="/api/cpp", tags=["cpp-inference"])


def _ensure_enabled() -> None:
    if not CPP_INFERENCE_ENABLED:
        raise HTTPException(status_code=404, detail="C++ inference proxy is disabled")


def _json_or_502(func, *args, **kwargs) -> Dict[str, Any]:
    _ensure_enabled()
    try:
        return func(*args, **kwargs)
    except CppInferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/proxy_info")
def cpp_proxy_info() -> Dict[str, Any]:
    _ensure_enabled()
    return {
        "status": "ok",
        "proxy": "visionops-collector-cpp-proxy",
        "version": "v0.5.5",
        **service_summary(),
    }


@router.get("/health")
def cpp_health() -> Dict[str, Any]:
    return _json_or_502(get_json, "/health")


@router.get("/stats")
def cpp_stats() -> Dict[str, Any]:
    return _json_or_502(get_json, "/stats")


@router.post("/stream/start")
def cpp_stream_start(request: Request) -> Dict[str, Any]:
    return _json_or_502(post_json, "/stream/start", query_string=request.scope.get("query_string"))


@router.post("/stream/preview/start")
def cpp_stream_preview_start(request: Request) -> Dict[str, Any]:
    return _json_or_502(post_json, "/stream/preview/start", query_string=request.scope.get("query_string"))


@router.post("/stream/detect/start")
def cpp_stream_detect_start(request: Request) -> Dict[str, Any]:
    return _json_or_502(post_json, "/stream/detect/start", query_string=request.scope.get("query_string"))


@router.post("/inference/start")
def cpp_inference_start(request: Request) -> Dict[str, Any]:
    return _json_or_502(post_json, "/inference/start", query_string=request.scope.get("query_string"))


@router.post("/inference/stop")
def cpp_inference_stop(request: Request) -> Dict[str, Any]:
    return _json_or_502(post_json, "/inference/stop", query_string=request.scope.get("query_string"))


@router.post("/stream/stop")
def cpp_stream_stop(request: Request) -> Dict[str, Any]:
    return _json_or_502(post_json, "/stream/stop", query_string=request.scope.get("query_string"))


@router.get("/stream/status")
def cpp_stream_status(request: Request) -> Dict[str, Any]:
    return _json_or_502(get_json, "/stream/status", query_string=request.scope.get("query_string"))


@router.get("/stream/latest_result")
def cpp_stream_latest_result(request: Request) -> Dict[str, Any]:
    return _json_or_502(get_json, "/stream/latest_result", query_string=request.scope.get("query_string"))


def _image_response(path: str) -> Response:
    """Proxy image endpoints.

    Frontend may append cache-busting query strings such as `?t=...`.
    The C++ v0.4.x lightweight HTTP router matches the literal path and may not
    ignore query strings. Therefore Collector accepts browser query parameters,
    but intentionally does NOT forward them to the C++ image endpoint.
    """
    _ensure_enabled()
    try:
        data, content_type = get_binary(path)
    except CppInferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@router.get("/stream/snapshot.jpg")
def cpp_stream_snapshot(request: Request) -> Response:
    # Do not forward request.query_string to the C++ service.
    return _image_response("/stream/snapshot.jpg")


@router.get("/stream/annotated.jpg")
def cpp_stream_annotated(request: Request) -> Response:
    # Do not forward request.query_string to the C++ service.
    return _image_response("/stream/annotated.jpg")
