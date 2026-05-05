#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from backend.config import (
    DATA_ROOT,
    DEVICE_ID,
    USER_ID,
    UI_VERSION,
    DEFAULT_DATASET_NAME,
    CAMERA_SOURCE,
    UPLOAD_ENABLED,
    UPLOAD_HOST,
    UPLOAD_USER,
    UPLOAD_PORT,
    UPLOAD_TARGET_DIR,
    PRODUCTION_DETECT_INTERVAL_MS,
    PRODUCTION_GATEWAY_PUSH_URL,
    PRODUCTION_CAMERA_ID,
)
from backend.services.models import list_rknn_models
from backend.services.validation_images import get_realtime_image_path, get_validation_image_path, list_validation_images, save_realtime_image_bytes, save_realtime_image_data
from backend.services.validation_infer import classify_image_with_model
from backend.services.production_push import production_push_service
from backend.services.gateway_push import push_result_to_gateway
from backend.services.camera import backend_camera_enabled, camera_service, mjpeg_stream, read_one_jpeg
from backend.services.storage import (
    FOLDER_TO_SUBDIR,
    create_dataset,
    create_upload_package,
    delete_image,
    clear_capture_images,
    ensure_data_root,
    ensure_default_dataset_dirs,
    ensure_dataset_dirs,
    get_counts,
    label_image,
    list_datasets,
    list_images,
    save_capture,
    save_capture_bytes,
    save_labeled_capture,
    sanitize_dataset_name,
    dataset_path,
    default_dataset_name,
)

router = APIRouter(prefix="/api")


class DatasetCreateRequest(BaseModel):
    name: str = DEFAULT_DATASET_NAME


class CaptureRequest(BaseModel):
    image_data: Optional[str] = None
    folder: str = "all"  # all / positive / negative
    dataset: str = DEFAULT_DATASET_NAME
    device_id: Optional[str] = None
    user_id: Optional[str] = None


class LabeledCaptureRequest(BaseModel):
    image_data: str
    label: str
    dataset: str = DEFAULT_DATASET_NAME
    device_id: Optional[str] = None
    user_id: Optional[str] = None


class LabelRequest(BaseModel):
    dataset: str = DEFAULT_DATASET_NAME
    filename: str
    label: str
    mode: str = "copy"


class DeleteRequest(BaseModel):
    dataset: str = DEFAULT_DATASET_NAME
    folder: str
    filename: str


class ClearDatasetImagesRequest(BaseModel):
    dataset: str = DEFAULT_DATASET_NAME


class UploadRequest(BaseModel):
    dataset: str = DEFAULT_DATASET_NAME
    device_id: str
    customer_id: str
    contact_info: Optional[str] = ""
    remark: Optional[str] = ""


class ValidationClassifyRequest(BaseModel):
    dataset: str = DEFAULT_DATASET_NAME
    model_name: str
    image_id: str


class ValidationCaptureClassifyRequest(BaseModel):
    dataset: str = DEFAULT_DATASET_NAME
    model_name: str
    image_data: Optional[str] = None
    device_id: Optional[str] = None
    user_id: Optional[str] = None


class ValidationRealtimeClassifyRequest(BaseModel):
    dataset: str = DEFAULT_DATASET_NAME
    model_name: str
    image_data: Optional[str] = None


class ProductionPushStartRequest(BaseModel):
    dataset: str = DEFAULT_DATASET_NAME
    model_name: str
    gateway_url: Optional[str] = None
    interval_ms: Optional[int] = None
    camera_id: Optional[int] = None


@router.get("/health")
def health():
    dirs = ensure_default_dataset_dirs()
    return {
        "status": "ok",
        "ui_version": UI_VERSION,
        "mode": "fixed-local-capture",
        "device_id": DEVICE_ID,
        "user_id": USER_ID,
        "data_root": str(DATA_ROOT),
        "dataset": dirs["dataset"],
        "dirs": dirs,
        "counts": get_counts(dirs["dataset"]),
        "camera": {
            "backend_enabled": backend_camera_enabled(),
            **camera_service.status(),
        },
        "upload": {
            "enabled": UPLOAD_ENABLED,
            "host": UPLOAD_HOST,
            "user": UPLOAD_USER,
            "port": UPLOAD_PORT,
            "target_dir": UPLOAD_TARGET_DIR,
        },
    }


@router.get("/datasets")
def datasets():
    ensure_default_dataset_dirs()
    return {"ok": True, "data_root": str(ensure_data_root()), "device_id": DEVICE_ID, "user_id": USER_ID, "items": list_datasets()}


@router.post("/datasets")
def datasets_create(req: DatasetCreateRequest):
    # 兼容旧版接口。v4.5 工人界面不再提供新增数据集入口。
    try:
        item = create_dataset(req.name or default_dataset_name())
        return {"ok": True, "message": f"本地采集目录已初始化：{item['name']}", "item": item, "items": list_datasets()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/dataset/init")
def init_dataset(req: DatasetCreateRequest):
    try:
        item = create_dataset(req.name or default_dataset_name())
        return {"ok": True, "message": "本地采集目录已创建", "dirs": item["dirs"], "counts": item["counts"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/dataset/summary")
def dataset_summary(dataset: str = DEFAULT_DATASET_NAME):
    try:
        dirs = ensure_dataset_dirs(dataset or default_dataset_name())
        return {"ok": True, "device_id": DEVICE_ID, "user_id": USER_ID, "dirs": dirs, "counts": get_counts(dirs["dataset"])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/dataset/images")
def dataset_images(dataset: str = DEFAULT_DATASET_NAME, folder: str = "all"):
    try:
        ds = sanitize_dataset_name(dataset or default_dataset_name())
        return {"ok": True, "dataset": ds, "folder": folder, "items": list_images(ds, folder), "counts": get_counts(ds)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/datasets/{dataset}/image/{folder}/{filename}")
def dataset_image(dataset: str, folder: str, filename: str):
    try:
        dataset = sanitize_dataset_name(dataset or default_dataset_name())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if folder not in FOLDER_TO_SUBDIR:
        raise HTTPException(status_code=400, detail="folder 只能是 all / positive / negative")
    safe_name = Path(filename).name
    path = dataset_path(dataset) / FOLDER_TO_SUBDIR[folder] / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(path)


@router.post("/capture")
def capture(req: CaptureRequest):
    try:
        ds = req.dataset or default_dataset_name()
        if backend_camera_enabled() and not req.image_data:
            camera_service.start()
            jpeg = camera_service.get_latest_frame_jpeg(quality=90, timeout=3.0)
            item = save_capture_bytes(ds, jpeg, req.folder, req.device_id or DEVICE_ID, req.user_id or USER_ID, ".jpg")
        else:
            item = save_capture(ds, req.image_data or "", req.folder, req.device_id or DEVICE_ID, req.user_id or USER_ID)
        folder_text = {"all": "全部图片", "positive": "正样本", "negative": "负样本"}.get(req.folder, req.folder)
        return {"ok": True, "message": f"已保存到 {folder_text} 文件夹：{item['filename']}", "item": item, "counts": get_counts(ds)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/capture/labeled")
def capture_labeled(req: LabeledCaptureRequest):
    """分类模式暂存图片确认后保存。

    只写入 positive/negative，不再同步写入 all_images。
    """
    try:
        ds = req.dataset or default_dataset_name()
        item = save_labeled_capture(ds, req.image_data or "", req.label, req.device_id or DEVICE_ID, req.user_id or USER_ID)
        label_text = "合格" if req.label == "positive" else "不合格"
        return {"ok": True, "message": f"已保存为{label_text}样本：{item['filename']}", "item": item, "counts": get_counts(ds)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/label")
def label(req: LabelRequest):
    try:
        item = label_image(req.dataset or default_dataset_name(), req.filename, req.label, req.mode)
        label_text = "正样本" if req.label == "positive" else "负样本"
        return {"ok": True, "message": f"已标注为{label_text}", "item": item, "counts": get_counts(req.dataset or default_dataset_name())}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/dataset/image/delete")
def image_delete(req: DeleteRequest):
    try:
        result = delete_image(req.dataset or default_dataset_name(), req.folder, req.filename)
        return {"ok": True, "message": "图片已删除", **result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/dataset/images/clear")
def dataset_images_clear(req: ClearDatasetImagesRequest):
    try:
        result = clear_capture_images(req.dataset or default_dataset_name())
        return {"ok": True, "message": "已清空检测图片、合格、不合格图片", **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload")
def upload(req: UploadRequest):
    try:
        package = create_upload_package(req.dataset or default_dataset_name(), req.device_id, req.customer_id, req.contact_info or "", req.remark or "")
        return {"ok": True, "message": package.get("remote_upload", {}).get("message", "已完成打包上传"), "package": package}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/camera/status")
def camera_status():
    return {"ok": True, "camera": camera_service.status()}


@router.post("/camera/stop")
def camera_stop():
    """停止后端摄像头读取线程，用于离开拍照采集/实时检测页面时释放资源。"""
    try:
        camera_service.stop()
        return {"ok": True, "message": "摄像头已停止", "camera": camera_service.status()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/camera/frame")
def camera_frame():
    try:
        jpeg = read_one_jpeg()
        return Response(content=jpeg, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/camera/stream")
def camera_stream():
    try:
        return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/infer")
def infer():
    return {"ok": True, "message": "演示模式：已完成检测", "latency_ms": 69.3}


@router.get("/models")
def models():
    try:
        return list_rknn_models()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/refresh_models")
def refresh_models():
    try:
        data = list_rknn_models()
        data["message"] = f"已刷新模型列表，共找到 {len(data['items'])} 个模型"
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/validation/images")
def validation_images(dataset: str = DEFAULT_DATASET_NAME):
    try:
        return list_validation_images(dataset or default_dataset_name())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/validation/image/{image_id}")
def validation_image(image_id: str, dataset: str = DEFAULT_DATASET_NAME):
    try:
        path = get_validation_image_path(image_id, dataset or default_dataset_name())
        return FileResponse(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/validation/classify_image")
def validation_classify_image(req: ValidationClassifyRequest):
    try:
        ds = req.dataset or default_dataset_name()
        image_path = get_validation_image_path(req.image_id, ds)
        result = classify_image_with_model(req.model_name, image_path)
        result["gateway_push"] = push_result_to_gateway(
            result,
            source="factory_image",
            dataset=ds,
            model_name=req.model_name,
            image_id=req.image_id,
            camera_id=PRODUCTION_CAMERA_ID,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/validation/capture_classify")
def validation_capture_classify(req: ValidationCaptureClassifyRequest):
    try:
        ds = req.dataset or default_dataset_name()
        if backend_camera_enabled() and not req.image_data:
            camera_service.start()
            jpeg = camera_service.get_latest_frame_jpeg(quality=92, timeout=3.0)
            item = save_capture_bytes(ds, jpeg, "all", req.device_id or DEVICE_ID, req.user_id or USER_ID, ".jpg")
        else:
            item = save_capture(ds, req.image_data or "", "all", req.device_id or DEVICE_ID, req.user_id or USER_ID)

        image_path = get_validation_image_path(item["filename"], ds)
        result = classify_image_with_model(req.model_name, image_path)
        result["captured"] = {
            "id": item["filename"],
            "name": item["filename"],
            "filename": item["filename"],
            "dataset": item.get("dataset", ds),
            "folder": "all",
            "url": f"/api/validation/image/{item['filename']}",
            "size_bytes": item.get("size_bytes"),
        }
        result["gateway_push"] = push_result_to_gateway(
            result,
            source="factory_capture",
            dataset=ds,
            model_name=req.model_name,
            image_id=item["filename"],
            camera_id=PRODUCTION_CAMERA_ID,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@router.get("/validation/realtime_image/{filename}")
def validation_realtime_image(filename: str, dataset: str = DEFAULT_DATASET_NAME):
    try:
        path = get_realtime_image_path(filename, dataset or default_dataset_name())
        return FileResponse(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/validation/realtime_classify_once")
def validation_realtime_classify_once(req: ValidationRealtimeClassifyRequest):
    """实时检测的一次单帧分类。

    注意：实时帧只写入 validation_tmp/realtime_latest.jpg，不进入 all_images，避免污染采集数据。
    """
    try:
        ds = req.dataset or default_dataset_name()
        if backend_camera_enabled() and not req.image_data:
            camera_service.start()
            jpeg = camera_service.get_latest_frame_jpeg(quality=90, timeout=2.0)
            frame = save_realtime_image_bytes(ds, jpeg, ".jpg")
        else:
            frame = save_realtime_image_data(ds, req.image_data or "")

        image_path = get_realtime_image_path(frame["filename"], ds)
        result = classify_image_with_model(req.model_name, image_path)
        result["mode"] = "realtime"
        result["realtime"] = frame
        result["gateway_push"] = push_result_to_gateway(
            result,
            source="factory_realtime",
            dataset=ds,
            model_name=req.model_name,
            image_id=frame["filename"],
            camera_id=PRODUCTION_CAMERA_ID,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/production/push/start")
def production_push_start(req: ProductionPushStartRequest):
    """生产模式：启动后端连续检测 + Gateway 推送。"""
    try:
        return production_push_service.start(
            model_name=req.model_name,
            dataset=req.dataset or DEFAULT_DATASET_NAME,
            gateway_url=req.gateway_url or PRODUCTION_GATEWAY_PUSH_URL,
            interval_ms=req.interval_ms or PRODUCTION_DETECT_INTERVAL_MS,
            camera_id=req.camera_id or PRODUCTION_CAMERA_ID,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/production/push/stop")
def production_push_stop():
    try:
        return production_push_service.stop()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/production/push/status")
def production_push_status():
    try:
        return production_push_service.status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
