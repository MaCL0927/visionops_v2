#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import APIRouter, HTTPException

from backend.services.settings_schema import VisionOpsRuntimeSettings, model_to_dict
from backend.services.settings_store import (
    get_algorithm_runtime_config,
    get_settings_path,
    load_settings,
    reset_settings,
    save_settings,
    write_runtime_algorithm_env,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def get_runtime_settings():
    settings = load_settings()
    return {
        "ok": True,
        "path": str(get_settings_path()),
        "settings": model_to_dict(settings),
        "message": "settings loaded",
    }


@router.post("")
def update_runtime_settings(settings: VisionOpsRuntimeSettings):
    try:
        saved = save_settings(settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存设置失败: {exc}") from exc
    return {
        "ok": True,
        "path": str(get_settings_path()),
        "settings": model_to_dict(saved),
        "message": "设置已保存",
    }


@router.post("/reset")
def reset_runtime_settings():
    try:
        settings = reset_settings()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"恢复默认设置失败: {exc}") from exc
    return {
        "ok": True,
        "path": str(get_settings_path()),
        "settings": model_to_dict(settings),
        "message": "已恢复默认设置",
    }


@router.get("/algorithm/effective")
def get_effective_algorithm_settings():
    settings = load_settings()
    algorithm = get_algorithm_runtime_config(settings)
    return {
        "ok": True,
        "algorithm": algorithm,
        "message": "algorithm settings loaded",
    }


@router.post("/algorithm/apply")
def apply_algorithm_settings():
    """v2.2：让算法运行时参数进入 runtime_algorithm.env，并通知验证服务下次按新参数重载。"""
    try:
        settings = load_settings()
        env_path = write_runtime_algorithm_env(settings)
        # 不直接重启 systemd：下一次模型验证/生产推理会由 validation_infer 检查参数签名，
        # 必要时通过 switch_model.sh 用新 runtime_algorithm.env 重载 engine。
        try:
            from backend.services.validation_infer import invalidate_algorithm_runtime
            invalidate_algorithm_runtime("algorithm settings updated")
        except Exception:
            pass
        return {
            "ok": True,
            "message": "算法设置已应用，新的阈值/TopK/推理间隔将在下一次推理时生效",
            "runtime_env": str(env_path),
            "algorithm": get_algorithm_runtime_config(settings),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"应用算法设置失败: {exc}") from exc

@router.post("/camera/apply")
def apply_camera_settings():
    """v2.1：保存后立即让 RTSP 相机与预览参数生效。"""
    try:
        from backend.services.camera import camera_service
        status = camera_service.reload_from_runtime(start=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"应用相机设置失败: {exc}") from exc
    return {
        "ok": True,
        "message": "相机设置已应用，正在重新连接 RTSP 预览",
        "camera": status,
    }


@router.post("/camera/test")
def test_camera_settings():
    """v2.1：测试当前 RTSP 配置是否能读取到一帧。"""
    try:
        from backend.services.camera import camera_service
        camera_service.reload_from_runtime(start=True)
        # 读取一帧即可判断 RTSP 参数、账号密码、通道是否基本正确。
        camera_service.get_latest_jpeg(timeout=6.0)
        status = camera_service.status()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"相机连接测试失败: {exc}") from exc
    return {
        "ok": True,
        "message": "相机连接测试通过",
        "camera": status,
    }

