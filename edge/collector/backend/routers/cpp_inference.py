#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collector API proxy for C++ inference service.

v0.7.3.1 adds independent C++ camera runtime settings APIs while keeping the
old Python camera settings path untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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
from backend.services.models import list_rknn_models
from backend.services.cpp_runtime_settings import (
    CppRuntimeSettingsError,
    apply_cpp_camera_settings,
    get_cpp_current_model_config,
    get_cpp_settings_payload,
    restart_cpp_service,
    save_cpp_camera_settings,
    wait_cpp_health,
    write_cpp_env,
    write_cpp_model_env,
    write_cpp_pipeline_env,
)

router = APIRouter(prefix="/api/cpp", tags=["cpp-inference"])


# v0.8.2.1 safety boundary:
# The C++ binary can be restarted with any single-model task through cpp.env,
# but current C++ v0.8.3.4 supports detection, classification, OBB, and segmentation postprocess in /infer and realtime path.
# v0.8.4 adds ROI 双模型 pipeline switching and C++ runtime support.
_CPP_DEFAULT_SWITCHABLE_TASKS = {"detection", "classification", "obb_detection", "segmentation", "roi_classification"}
_CPP_EXPERIMENTAL_SINGLE_TASKS = {"classification", "detection", "obb_detection", "segmentation"}


def _ensure_enabled() -> None:
    if not CPP_INFERENCE_ENABLED:
        raise HTTPException(status_code=404, detail="C++ inference proxy is disabled")


def _json_or_502(func, *args, **kwargs) -> Dict[str, Any]:
    _ensure_enabled()
    try:
        return func(*args, **kwargs)
    except CppInferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc




def _same_path(a: str, b: str) -> bool:
    """Compare two filesystem paths while tolerating empty/missing files."""
    if not a or not b:
        return False
    try:
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except Exception:
        return str(a).strip() == str(b).strip()




def _build_cpp_current_model_payload() -> Dict[str, Any]:
    """v0.8.1: read current C++ model config from cpp.env and compare runtime health."""
    configured = get_cpp_current_model_config()

    runtime_health: Dict[str, Any] = {}
    runtime_error = ""
    try:
        runtime_health = get_json("/health")
    except Exception as exc:
        runtime_error = str(exc)
        runtime_health = {}

    cfg_model = str(configured.get("model_path") or "")
    cfg_meta = str(configured.get("meta_path") or configured.get("class_names_file") or "")
    cfg_pipeline = str(configured.get("pipeline_config") or "")
    runtime_model = str(runtime_health.get("model") or runtime_health.get("model_path") or "")
    runtime_meta = str(
        runtime_health.get("class_names_file")
        or runtime_health.get("meta_path")
        or runtime_health.get("yaml_path")
        or ""
    )
    runtime_pipeline = str(runtime_health.get("pipeline_config") or "")
    runtime_task = str(runtime_health.get("task") or "")

    if str(configured.get("task") or "") == "roi_classification":
        model_matches = bool(runtime_health) and bool(cfg_pipeline) and _same_path(cfg_pipeline, runtime_pipeline)
    else:
        model_matches = bool(runtime_health) and (
            _same_path(cfg_model, runtime_model)
            or (bool(cfg_meta) and bool(runtime_meta) and _same_path(cfg_meta, runtime_meta))
        )
    task_matches = bool(runtime_health) and (
        not configured.get("task")
        or not runtime_task
        or str(configured.get("task")) == runtime_task
    )

    return {
        "ok": True,
        "backend": "cpp-rknn",
        "usage": "cpp_current_model_config",
        "current_cpp": {
            "configured": bool(configured.get("valid")),
            "config_source": "cpp.env",
            "config": configured,
            "runtime_reachable": bool(runtime_health),
            "runtime_error": runtime_error,
            "runtime": {
                "task": runtime_task,
                "model_path": runtime_model,
                "meta_path": runtime_meta,
                "pipeline_config": runtime_pipeline,
                "health": runtime_health,
            },
            "model_matches_runtime": model_matches,
            "task_matches_runtime": task_matches,
            "config_matches_runtime": bool(model_matches and task_matches),
        },
    }

def _build_cpp_models_payload() -> Dict[str, Any]:
    """v0.8.1: reuse model scanner and annotate cpp.env/runtime status.

    cpp.env is now the persistent source of truth for the configured C++ model.
    /health is only used to show whether the running service has already loaded
    the same model. Listing still works when the C++ process is stopped.
    """
    data = list_rknn_models()
    current_payload = _build_cpp_current_model_payload()
    current_cpp = current_payload.get("current_cpp", {})
    configured = current_cpp.get("config") if isinstance(current_cpp.get("config"), dict) else {}
    runtime = current_cpp.get("runtime") if isinstance(current_cpp.get("runtime"), dict) else {}

    configured_model_path = str(configured.get("model_path") or "")
    configured_meta_path = str(configured.get("meta_path") or configured.get("class_names_file") or "")
    runtime_model_path = str(runtime.get("model_path") or "")
    runtime_meta_path = str(runtime.get("meta_path") or "")

    single_tasks = {"classification", "detection", "obb_detection", "segmentation"}
    known_tasks = single_tasks | {"roi_classification"}

    for item in data.get("items", []):
        task = str(item.get("task") or "")
        is_pipeline = bool(item.get("is_pipeline"))
        item_path = str(item.get("path") or "")
        item_meta_path = str(item.get("meta_path") or item.get("pipeline_config") or "")

        cpp_configured = False
        cpp_runtime_loaded = False
        if is_pipeline:
            configured_pipeline = str(configured.get("pipeline_config") or "")
            runtime_pipeline = str((runtime.get("health") or {}).get("pipeline_config") or runtime.get("pipeline_config") or "") if isinstance(runtime, dict) else ""
            cpp_configured = _same_path(item_meta_path, configured_pipeline)
            cpp_runtime_loaded = _same_path(item_meta_path, runtime_pipeline)
        else:
            cpp_configured = _same_path(item_path, configured_model_path)
            if not cpp_configured and configured_meta_path:
                cpp_configured = _same_path(item_meta_path, configured_meta_path)

            cpp_runtime_loaded = _same_path(item_path, runtime_model_path)
            if not cpp_runtime_loaded and runtime_meta_path:
                cpp_runtime_loaded = _same_path(item_meta_path, runtime_meta_path)

        item["cpp_supported"] = task in known_tasks
        item["cpp_switch_ready"] = ((not is_pipeline) and task in _CPP_DEFAULT_SWITCHABLE_TASKS) or (is_pipeline and task == "roi_classification")
        item["cpp_experimental_switch_ready"] = (not is_pipeline) and task in _CPP_EXPERIMENTAL_SINGLE_TASKS
        item["cpp_configured"] = cpp_configured
        item["cpp_runtime_loaded"] = cpp_runtime_loaded
        # Keep v0.8.0 frontend compatibility. In v0.8.1 this means current runtime
        # if reachable, otherwise the model configured in cpp.env.
        item["cpp_loaded"] = cpp_runtime_loaded or cpp_configured
        item["cpp_switch_mode"] = "roi_pipeline" if is_pipeline else "single_model"
        if is_pipeline and task == "roi_classification":
            item["cpp_note"] = "ROI 双模型 pipeline，可切换为 C++ 当前模型"
        elif task not in known_tasks:
            item["cpp_note"] = "当前 C++ 模型管理暂未识别该任务类型"
        elif not item.get("has_meta", True):
            item["cpp_note"] = "缺少模型配置 yaml，无法作为 C++ 切换候选"
        elif task not in _CPP_DEFAULT_SWITCHABLE_TASKS:
            item["cpp_note"] = "已扫描到该模型；C++ 当前版本暂未默认允许切换该任务"
        else:
            item["cpp_note"] = ""

    data["backend"] = "cpp-rknn"
    data["usage"] = "cpp_model_management"
    data["current_cpp"] = {
        # Compatibility fields used by the existing frontend.
        "reachable": bool(current_cpp.get("runtime_reachable")),
        "error": str(current_cpp.get("runtime_error") or ""),
        "task": str(configured.get("task") or runtime.get("task") or ""),
        "model_path": configured_model_path or runtime_model_path,
        "pipeline_config": str(configured.get("pipeline_config") or (runtime.get("health") or {}).get("pipeline_config") or "") if isinstance(runtime, dict) else "",
        "meta_path": configured_meta_path or runtime_meta_path,
        "health": (runtime.get("health") if isinstance(runtime.get("health"), dict) else {}),
        # v0.8.1 structured fields.
        "configured": bool(current_cpp.get("configured")),
        "config_source": "cpp.env",
        "config": configured,
        "runtime_reachable": bool(current_cpp.get("runtime_reachable")),
        "runtime": runtime,
        "model_matches_runtime": bool(current_cpp.get("model_matches_runtime")),
        "task_matches_runtime": bool(current_cpp.get("task_matches_runtime")),
        "config_matches_runtime": bool(current_cpp.get("config_matches_runtime")),
    }
    return data


def _norm_ref(value: Any) -> str:
    return str(value or "").strip()


def _model_ref_matches(item: Dict[str, Any], model_ref: str) -> bool:
    """Match frontend model reference against scanner item safely.

    Supported references:
    - full filename: xxx.rknn
    - stem: xxx
    - absolute model path: /opt/visionops/models/xxx.rknn
    - meta path: /opt/visionops/models/xxx.yaml
    """
    ref = _norm_ref(model_ref)
    if not ref:
        return False

    candidates = {
        _norm_ref(item.get("name")),
        _norm_ref(item.get("stem")),
        _norm_ref(item.get("path")),
        _norm_ref(item.get("meta_name")),
        _norm_ref(item.get("meta_path")),
    }
    if ref in candidates:
        return True

    ref_path = Path(ref)
    ref_name = ref_path.name
    ref_stem = ref_path.stem
    if ref_name and ref_name in candidates:
        return True
    if ref_stem and ref_stem in candidates:
        return True

    item_path = _norm_ref(item.get("path"))
    item_meta = _norm_ref(item.get("meta_path"))
    return _same_path(ref, item_path) or _same_path(ref, item_meta)



def _find_cpp_model_or_pipeline(model_ref: str) -> Dict[str, Any]:
    data = list_rknn_models()
    matches = [item for item in data.get("items", []) if _model_ref_matches(item, model_ref)]
    if not matches:
        # For pipeline bundles the frontend may pass pipeline_config path.
        for item in data.get("items", []):
            ref = _norm_ref(model_ref)
            candidates = {_norm_ref(item.get("pipeline_config")), _norm_ref(item.get("path")), _norm_ref(item.get("name")), _norm_ref(item.get("stem"))}
            if ref in candidates or _same_path(ref, str(item.get("pipeline_config") or "")):
                matches.append(item)
    if not matches:
        raise HTTPException(status_code=404, detail=f"未找到模型或 pipeline: {model_ref}")
    if len(matches) > 1:
        names = ", ".join(str(x.get("name") or x.get("stem") or x.get("path") or x.get("pipeline_config")) for x in matches[:5])
        raise HTTPException(status_code=400, detail=f"模型引用不唯一，请使用完整 model_name/path: {names}")
    return dict(matches[0])


def _pipeline_item_to_cpp_env_config(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task": "roi_classification",
        "pipeline_config": str(item.get("pipeline_config") or item.get("meta_path") or item.get("path") or ""),
    }


def _validate_cpp_pipeline_item(item: Dict[str, Any]) -> None:
    if not bool(item.get("is_pipeline")):
        raise HTTPException(status_code=400, detail="该条目不是 ROI pipeline")
    task = str(item.get("task") or "").strip()
    if task != "roi_classification":
        raise HTTPException(status_code=400, detail=f"当前 C++ pipeline 切换仅支持 roi_classification，收到: {task or 'empty'}")
    cfg = Path(str(item.get("pipeline_config") or item.get("meta_path") or item.get("path") or ""))
    if not cfg.exists() or not cfg.is_file():
        raise HTTPException(status_code=400, detail=f"pipeline.yaml 不存在: {cfg}")

def _find_cpp_single_model(model_ref: str) -> Dict[str, Any]:
    data = list_rknn_models()
    matches = [item for item in data.get("items", []) if _model_ref_matches(item, model_ref)]
    if not matches:
        raise HTTPException(status_code=404, detail=f"未找到模型: {model_ref}")
    if len(matches) > 1:
        names = ", ".join(str(x.get("name") or x.get("stem") or x.get("path")) for x in matches[:5])
        raise HTTPException(status_code=400, detail=f"模型引用不唯一，请使用完整 model_name 或 path: {names}")
    return dict(matches[0])


def _model_item_to_cpp_env_config(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_path": str(item.get("path") or ""),
        "class_names_file": str(item.get("meta_path") or ""),
        "meta_path": str(item.get("meta_path") or ""),
        "task": str(item.get("task") or ""),
        "num_classes": int(item.get("num_classes") or 0),
        "input_size": item.get("input_size") or [],
        "conf_threshold": item.get("conf_threshold"),
        "nms_threshold": item.get("nms_threshold"),
        "topk": item.get("topk"),
    }


def _validate_cpp_single_model_item(item: Dict[str, Any], *, allow_experimental_task: bool = False) -> None:
    if bool(item.get("is_pipeline")):
        raise HTTPException(status_code=400, detail="该条目是 ROI/组合模型，不属于 v0.8.2.1 单模型切换范围")

    model_path = Path(str(item.get("path") or ""))
    meta_path = Path(str(item.get("meta_path") or ""))
    task = str(item.get("task") or "").strip()

    if not model_path.exists() or not model_path.is_file():
        raise HTTPException(status_code=400, detail=f"模型文件不存在: {model_path}")
    if not bool(item.get("has_meta")) or not meta_path.exists() or not meta_path.is_file():
        raise HTTPException(status_code=400, detail=f"缺少同名 YAML 配置文件: {meta_path}")
    if task not in _CPP_EXPERIMENTAL_SINGLE_TASKS:
        raise HTTPException(status_code=400, detail=f"不支持的单模型任务类型: {task or 'empty'}")
    if (not allow_experimental_task) and task not in _CPP_DEFAULT_SWITCHABLE_TASKS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"当前 C++ v0.8.3.3 默认允许 detection/classification/obb_detection；{task} 的 C++ 后处理将在后续 segmentation 版本接入。"
                "如确实只是测试 cpp.env 写入和服务重启，可在请求体中传 allow_experimental_task=true。"
            ),
        )
    try:
        num_classes = int(item.get("num_classes") or 0)
    except Exception:
        num_classes = 0
    if num_classes <= 0:
        raise HTTPException(status_code=400, detail="num_classes 无效，无法写入 C++ 模型配置")

    input_size = item.get("input_size") or []
    if not (isinstance(input_size, list) and len(input_size) >= 2):
        raise HTTPException(status_code=400, detail="input_size 无效，无法写入 C++ 模型配置")


def _try_stop_cpp_stream() -> Dict[str, Any]:
    try:
        data = post_json("/stream/stop")
        return {"ok": True, "result": data}
    except Exception as exc:
        # Stopping stream is a safety step. It should not block switching when the
        # service is already stopped or temporarily unreachable before restart.
        return {"ok": False, "warning": str(exc)}


def _switch_result_matches_selected(health: Dict[str, Any], selected: Dict[str, Any]) -> Dict[str, Any]:
    runtime_model = str(health.get("model") or health.get("model_path") or "")
    runtime_meta = str(health.get("class_names_file") or health.get("meta_path") or health.get("yaml_path") or "")
    runtime_pipeline = str(health.get("pipeline_config") or "")
    runtime_task = str(health.get("task") or "")
    selected_model = str(selected.get("model_path") or "")
    selected_meta = str(selected.get("meta_path") or selected.get("class_names_file") or "")
    selected_pipeline = str(selected.get("pipeline_config") or "")
    selected_task = str(selected.get("task") or "")
    if selected_task == "roi_classification":
        model_matches = bool(selected_pipeline) and _same_path(runtime_pipeline, selected_pipeline)
    else:
        model_matches = _same_path(runtime_model, selected_model) or (
            bool(runtime_meta) and bool(selected_meta) and _same_path(runtime_meta, selected_meta)
        )
    task_matches = (not runtime_task) or (runtime_task == selected_task)
    return {
        "runtime_model_path": runtime_model,
        "runtime_meta_path": runtime_meta,
        "runtime_pipeline_config": runtime_pipeline,
        "runtime_task": runtime_task,
        "model_matches_runtime": bool(model_matches),
        "task_matches_runtime": bool(task_matches),
        "config_matches_runtime": bool(model_matches and task_matches),
    }


@router.get("/proxy_info")
def cpp_proxy_info() -> Dict[str, Any]:
    _ensure_enabled()
    return {
        "status": "ok",
        "proxy": "visionops-collector-cpp-proxy",
        "version": "v0.7.3.1",
        **service_summary(),
    }


@router.get("/models")
def cpp_models() -> Dict[str, Any]:
    """v0.8.0：C++ 模型列表。

    复用原 Python 架构下的 list_rknn_models() 扫描逻辑，并额外标记
    当前 C++ 推理服务正在加载哪个模型。
    """
    _ensure_enabled()
    try:
        return _build_cpp_models_payload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取 C++ 模型列表失败: {exc}") from exc


@router.post("/refresh_models")
def cpp_refresh_models() -> Dict[str, Any]:
    """v0.8.0：刷新 C++ 模型列表，供前端刷新按钮使用。"""
    _ensure_enabled()
    try:
        data = _build_cpp_models_payload()
        data["message"] = f"已刷新 C++ 模型列表，共找到 {len(data.get('items', []))} 个模型"
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"刷新 C++ 模型列表失败: {exc}") from exc




@router.get("/model/current")
def cpp_model_current() -> Dict[str, Any]:
    """v0.8.1：读取 cpp.env 中配置的当前 C++ 模型，并对比运行中 /health。"""
    _ensure_enabled()
    try:
        return _build_cpp_current_model_payload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取 C++ 当前模型配置失败: {exc}") from exc


@router.post("/model/switch")
async def cpp_model_switch(request: Request) -> Dict[str, Any]:
    """v0.8.2.1：后端单模型切换接口。

    Steps:
    1. Match the requested model from list_rknn_models() instead of trusting raw paths.
    2. Validate it is a single RKNN model with YAML metadata.
    3. Update model-related keys in cpp.env while preserving camera settings.
    4. Restart visionops-inference-cpp.service.
    5. Wait for /health and verify the running C++ service loaded the selected model.
    """
    _ensure_enabled()
    try:
        try:
            raw = await request.json()
        except Exception:
            raw = {}
        payload = raw if isinstance(raw, dict) else {}

        model_ref = (
            payload.get("model_name")
            or payload.get("model_stem")
            or payload.get("model_path")
            or payload.get("name")
            or payload.get("stem")
            or ""
        )
        model_ref = str(model_ref).strip()
        if not model_ref:
            raise HTTPException(status_code=400, detail="缺少 model_name / model_stem / model_path")

        restart = bool(payload.get("restart", True))
        stop_stream = bool(payload.get("stop_stream", True))
        allow_experimental_task = bool(payload.get("allow_experimental_task", False))

        item = _find_cpp_model_or_pipeline(model_ref)
        is_pipeline = bool(item.get("is_pipeline"))
        if is_pipeline:
            _validate_cpp_pipeline_item(item)
            selected = _pipeline_item_to_cpp_env_config(item)
        else:
            _validate_cpp_single_model_item(item, allow_experimental_task=allow_experimental_task)
            selected = _model_item_to_cpp_env_config(item)

        stop_info: Optional[Dict[str, Any]] = None
        if stop_stream:
            stop_info = _try_stop_cpp_stream()

        env_path = write_cpp_pipeline_env(selected) if is_pipeline else write_cpp_model_env(selected)

        restart_info: Optional[Dict[str, Any]] = None
        health: Dict[str, Any] = {}
        match_info: Dict[str, Any] = {
            "model_matches_runtime": False,
            "task_matches_runtime": False,
            "config_matches_runtime": False,
        }

        if restart:
            restart_info = restart_cpp_service()
            health = wait_cpp_health(timeout_sec=float(payload.get("health_timeout_sec") or 15.0))
            match_info = _switch_result_matches_selected(health, selected)
            if not match_info.get("config_matches_runtime"):
                if selected.get("task") == "roi_classification":
                    raise CppRuntimeSettingsError(
                        "C++ 服务已重启，但 /health 中的 ROI pipeline 与 cpp.env 新配置不一致："
                        f"runtime_pipeline={match_info.get('runtime_pipeline_config')}, "
                        f"selected_pipeline={selected.get('pipeline_config')}, "
                        f"runtime_task={match_info.get('runtime_task')}"
                    )
                raise CppRuntimeSettingsError(
                    "C++ 服务已重启，但 /health 中的模型与 cpp.env 新配置不一致："
                    f"runtime_model={match_info.get('runtime_model_path')}, selected_model={selected.get('model_path')}"
                )

        current = _build_cpp_current_model_payload()
        models = _build_cpp_models_payload()

        return {
            "ok": True,
            "message": "C++ 单模型配置已写入 cpp.env" + ("，并已重启 C++ 推理服务" if restart else "，尚未重启服务"),
            "backend": "cpp-rknn",
            "usage": "cpp_model_or_roi_pipeline_switch",
            "selected_model": {
                "name": item.get("name"),
                "stem": item.get("stem"),
                "task": selected.get("task"),
                "model_path": selected.get("model_path"),
                "pipeline_config": selected.get("pipeline_config"),
                "meta_path": selected.get("meta_path"),
                "num_classes": selected.get("num_classes"),
                "input_size": selected.get("input_size"),
            },
            "cpp_env_path": str(env_path),
            "stop_stream": stop_info,
            "restart": restart_info,
            "health": health,
            **match_info,
            "current_cpp": current.get("current_cpp", {}),
            "models_count": len(models.get("items", [])),
        }
    except HTTPException:
        raise
    except CppRuntimeSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"切换 C++ 单模型失败: {exc}") from exc


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
