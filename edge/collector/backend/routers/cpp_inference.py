#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collector API proxy for C++ inference service.

v0.7.3.1 adds independent C++ camera runtime settings APIs while keeping the
old Python camera settings path untouched.
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
from backend.services.cpp_runtime_settings import (
    CppRuntimeSettingsError,
    apply_cpp_camera_settings,
    get_cpp_settings_payload,
    save_cpp_camera_settings,
    write_cpp_env,
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
        "version": "v0.7.3.1",
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


@router.get("/settings")
def cpp_get_settings() -> Dict[str, Any]:
    """Read editable/effective C++ camera settings.

    This does not touch the legacy Python camera settings. It reads
    runtime_overrides.yaml:cpp_inference.camera and cpp.env.
    """
    _ensure_enabled()
    try:
        return get_cpp_settings_payload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取 C++ 设置失败: {exc}") from exc


@router.post("/settings")
async def cpp_save_settings(request: Request) -> Dict[str, Any]:
    """Save C++ camera settings to runtime_overrides.yaml only.

    The C++ service is not restarted here. Use /settings/apply to write cpp.env
    and restart visionops-inference-cpp.service.
    """
    _ensure_enabled()
    try:
        payload = await request.json()
        settings = save_cpp_camera_settings(payload if isinstance(payload, dict) else {})
        return {
            "ok": True,
            "message": "C++ 相机设置已保存，尚未应用到运行中的 C++ 服务",
            "settings": settings,
            **{k: v for k, v in get_cpp_settings_payload().items() if k.endswith("path")},
        }
    except CppRuntimeSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存 C++ 设置失败: {exc}") from exc


@router.post("/settings/write_env")
async def cpp_write_settings_env(request: Request) -> Dict[str, Any]:
    """Write cpp.env from posted settings without restarting the C++ service."""
    _ensure_enabled()
    try:
        payload = await request.json()
        settings = save_cpp_camera_settings(payload if isinstance(payload, dict) else {})
        env_path = write_cpp_env(settings)
        return {
            "ok": True,
            "message": "C++ 设置已写入 cpp.env，尚未重启服务",
            "settings": settings,
            "cpp_env_path": str(env_path),
        }
    except CppRuntimeSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"写入 cpp.env 失败: {exc}") from exc


@router.post("/settings/apply")
async def cpp_apply_settings(request: Request) -> Dict[str, Any]:
    """Save settings, write cpp.env and restart the C++ inference service."""
    _ensure_enabled()
    try:
        payload: Dict[str, Any] | None = None
        try:
            raw = await request.json()
            payload = raw if isinstance(raw, dict) and raw else None
        except Exception:
            payload = None
        return apply_cpp_camera_settings(payload)
    except CppRuntimeSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"应用 C++ 设置失败: {exc}") from exc


def _image_response(path: str) -> Response:
    """Proxy image endpoints.

    Frontend may append cache-busting query strings such as `?t=...`.
    The C++ lightweight HTTP router may match the literal path and may not ignore
    query strings, so Collector intentionally does not forward query parameters
    to image endpoints.
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
