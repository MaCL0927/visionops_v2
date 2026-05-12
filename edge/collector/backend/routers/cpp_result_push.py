#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C++ latest_result -> robot_gateway 自动推送控制接口。"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from backend.services.cpp_result_push import cpp_result_push_service

router = APIRouter(prefix="/api/cpp/push", tags=["cpp-result-push"])


async def _optional_json(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@router.post("/start")
async def cpp_push_start(request: Request) -> Dict[str, Any]:
    payload = await _optional_json(request)
    try:
        return cpp_result_push_service.start(
            gateway_url=payload.get("gateway_url") or payload.get("url"),
            fps=payload.get("fps"),
            camera_id=payload.get("camera_id"),
            timeout_sec=payload.get("timeout_sec"),
            push_empty=payload.get("push_empty"),
            dedupe=payload.get("dedupe"),
            require_inference=payload.get("require_inference"),
            source=payload.get("source"),
            schema=payload.get("schema"),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/stop")
def cpp_push_stop() -> Dict[str, Any]:
    return cpp_result_push_service.stop()


@router.get("/status")
def cpp_push_status() -> Dict[str, Any]:
    return cpp_result_push_service.status()


@router.post("/once")
async def cpp_push_once(request: Request) -> Dict[str, Any]:
    payload = await _optional_json(request)
    force = bool(payload.get("force", True))
    try:
        return cpp_result_push_service.run_once(force=force)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
