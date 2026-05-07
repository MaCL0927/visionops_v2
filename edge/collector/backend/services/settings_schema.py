#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict
from pydantic import BaseModel, Field


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """兼容 Pydantic v1/v2。"""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class UsbCameraSettings(BaseModel):
    device_node: str = "/dev/video0"
    backend: str = "opencv"
    buffer_size: int = 1


class RtspCameraSettings(BaseModel):
    # 不写入默认密码，避免 Web 一打开就用错误密码反复连接相机并触发相机锁定。
    ip: str = ""
    port: int = 554
    channel: str = "102"
    username: str = "admin"
    password: str = ""
    transport: str = "tcp"
    url: str = ""


class IndustrialCameraSettings(BaseModel):
    vendor: str = "hikrobot"
    trigger_mode: str = "continuous"
    trigger_source: str = "Line0"
    roi_x: int = 0
    roi_y: int = 0
    roi_wh: str = "1280x720"


class MockCameraSettings(BaseModel):
    pattern: str = "detection_scene"
    fps: int = 10
    timestamp: bool = True


class CameraCommonSettings(BaseModel):
    resolution: str = "1280x720"
    fps: int = 6
    rotation: str = "0"
    exposure_mode: str = "auto"
    exposure: int = 0
    gain: int = 0
    white_balance: str = "auto"
    brightness: int = 50
    contrast: int = 50
    preview_width: int = 960
    jpeg_quality: int = 75
    reconnect_max_fails: int = 30


class CameraCalibrationSettings(BaseModel):
    source: str = "usb"
    calibration_type: str = "mono"
    checkerboard: str = "9x6"
    square_size_mm: float = 25.0
    active_file: str = ""


class CameraSettings(BaseModel):
    type: str = "rtsp"
    usb: UsbCameraSettings = Field(default_factory=UsbCameraSettings)
    rtsp: RtspCameraSettings = Field(default_factory=RtspCameraSettings)
    industrial: IndustrialCameraSettings = Field(default_factory=IndustrialCameraSettings)
    mock: MockCameraSettings = Field(default_factory=MockCameraSettings)
    common: CameraCommonSettings = Field(default_factory=CameraCommonSettings)
    calibration: CameraCalibrationSettings = Field(default_factory=CameraCalibrationSettings)


class NetworkInterfaceSettings(BaseModel):
    role: str = ""
    dhcp: bool = False
    ip: str = ""
    netmask: str = "255.255.255.0"
    gateway: str = ""


class VisionBoxNetworkSettings(BaseModel):
    eth0: NetworkInterfaceSettings = Field(default_factory=lambda: NetworkInterfaceSettings(role="camera_lan", dhcp=False, ip="192.168.2.10", netmask="255.255.255.0", gateway=""))
    eth1: NetworkInterfaceSettings = Field(default_factory=lambda: NetworkInterfaceSettings(role="factory_lan", dhcp=True, ip="192.168.1.200", netmask="255.255.255.0", gateway="192.168.1.1"))


class UploadSettings(BaseModel):
    enabled: bool = True
    host: str = ""
    user: str = "pc"
    port: int = 22
    target_dir: str = "/home/pc/桌面/visionops_v2/data/incoming"
    timeout_sec: int = 120


class VisionBoxSettings(BaseModel):
    device_id: str = "rk3588-001"
    customer_id: str = "CUST-001"
    default_mode: str = "production"
    collector_port: int = 8090
    production_port: int = 8080
    validation_port: int = 8082
    models_dir: str = "/opt/visionops/models"
    data_dir: str = "/opt/visionops/edge/collector/data/local_dataset"
    log_keep_days: int = 7
    usb_auto_mount: bool = True
    disk_warn_percent: int = 80
    time_sync: str = "ntp"
    network: VisionBoxNetworkSettings = Field(default_factory=VisionBoxNetworkSettings)
    upload: UploadSettings = Field(default_factory=UploadSettings)


class AlgorithmCommonSettings(BaseModel):
    model_mode: str = "auto"
    input_size: str = "640x640"
    npu_core: str = "auto"
    realtime_interval_ms: int = 1000
    production_fps_limit: int = 5
    production_detect_interval_ms: int = 1000
    warmup_runs: int = 3
    label_display: str = "class_conf"
    max_results: int = 100
    log_level: str = "INFO"


class ClassificationSettings(BaseModel):
    topk: int = 5
    score_threshold: float = 0.5
    low_confidence_policy: str = "review"


class DetectionSettings(BaseModel):
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    show_center: bool = True
    max_detections: int = 100


class ObbDetectionSettings(BaseModel):
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    nms_mode: str = "rotated"
    show_angle: bool = True
    show_polygon: bool = True


class SegmentationSettings(BaseModel):
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    mask_threshold: float = 0.5
    mask_alpha: float = 0.35
    show_mask: bool = True
    show_box: bool = True
    show_mode: str = "mask_box"


class AlgorithmSettings(BaseModel):
    common: AlgorithmCommonSettings = Field(default_factory=AlgorithmCommonSettings)
    classification: ClassificationSettings = Field(default_factory=ClassificationSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    obb_detection: ObbDetectionSettings = Field(default_factory=ObbDetectionSettings)
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)


class VisionOpsRuntimeSettings(BaseModel):
    version: str = "2.2"
    camera: CameraSettings = Field(default_factory=CameraSettings)
    vision_box: VisionBoxSettings = Field(default_factory=VisionBoxSettings)
    algorithm: AlgorithmSettings = Field(default_factory=AlgorithmSettings)
