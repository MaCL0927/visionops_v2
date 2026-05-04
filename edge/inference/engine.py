"""
RK3588 边缘推理引擎（Detection / Classification / OBB / Segmentation 通用版）

功能：
- 基于 rknnlite2 运行 RKNN 模型
- FastAPI 推理服务
- 支持 YOLO Detection 后处理
- 支持 Image Classification 后处理
- 支持 YOLOv8 OBB 旋转框后处理
- 支持 YOLOv8 Segmentation 分割后处理（box + mask polygon）

典型启动：

Detection:
python edge/inference/engine.py \
  --model /opt/visionops/models/current.rknn \
  --task detection \
  --input-size 640 640 \
  --class-names-file /opt/visionops/edge/runtime/class_names.yaml \
  --port 8080

Classification:
python edge/inference/engine.py \
  --model /opt/visionops/models/current.rknn \
  --task classification \
  --input-size 224 224 \
  --num-classes 2 \
  --class-names-file /opt/visionops/edge/runtime/class_names.yaml \
  --port 8080
"""

# from __future__ import annotations

import os
import time
import logging
import threading
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import deque

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visionops.inference")


# ────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────
@dataclass
class InferenceConfig:
    model_path: str = "/opt/visionops/models/current.rknn"
    target_platform: str = "rk3588"

    # 新增：任务类型。默认 detection，保证旧启动方式不受影响。
    task: str = "detection"  # detection / classification / obb_detection / segmentation
    npu_core: str = "auto"

    # detection 默认 [640, 640]；classification 建议启动时传 [224, 224]
    input_size: List[int] = field(default_factory=lambda: [640, 640])

    class_names: List[str] = field(default_factory=lambda: ["person", "smoke"])
    num_classes: int = 2
    class_names_file: Optional[str] = None

    # detection / obb_detection / segmentation 参数
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45

    # segmentation 参数
    mask_threshold: float = 0.5

    # classification 参数
    topk: int = 5

    metrics_port: int = 9091
    warmup_runs: int = 3
    debug_shapes: bool = True

    def normalized_task(self) -> str:
        task = str(self.task).lower().strip()
        if task in {"detect", "detection", "yolo_detection", "object_detection"}:
            return "detection"
        if task in {"cls", "classify", "classification", "image_classification"}:
            return "classification"
        if task in {
            "obb", "obb_detection", "oriented_detection", "oriented_bbox_detection",
            "rotated_detection", "rotated_bbox_detection", "yolo_obb", "yolov8_obb",
        }:
            return "obb_detection"
        if task in {
            "seg", "segment", "segmentation", "instance_segmentation",
            "yolo_seg", "yolov8_seg", "mask_segmentation",
        }:
            return "segmentation"
        raise ValueError(f"不支持的 task: {self.task}")


# ────────────────────────────────────────────────
# 性能指标
# ────────────────────────────────────────────────
class MetricsCollector:
    def __init__(self, window_size: int = 100):
        self.latencies = deque(maxlen=window_size)
        self.total_inferences = 0
        self.errors = 0
        self._lock = threading.Lock()

    def record(self, latency_ms: float, success: bool = True):
        with self._lock:
            self.total_inferences += 1
            if success:
                self.latencies.append(latency_ms)
            else:
                self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            base = {
                "total_inferences": self.total_inferences,
                "errors": self.errors,
                "successes": self.total_inferences - self.errors,
            }
            if self.total_inferences > 0:
                base["error_rate"] = round(float(self.errors / self.total_inferences), 6)
            else:
                base["error_rate"] = 0.0

            if not self.latencies:
                return base

            lats = list(self.latencies)
            mean_latency = float(np.mean(lats))
            base.update(
                {
                    "latency_ms": {
                        "mean": round(mean_latency, 2),
                        "p50": round(float(np.percentile(lats, 50)), 2),
                        "p95": round(float(np.percentile(lats, 95)), 2),
                        "p99": round(float(np.percentile(lats, 99)), 2),
                        "min": round(float(np.min(lats)), 2),
                        "max": round(float(np.max(lats)), 2),
                    },
                    "throughput_fps": round(float(1000.0 / mean_latency), 2) if mean_latency > 0 else 0.0,
                }
            )
            return base

    def prometheus_format(self) -> str:
        stats = self.get_stats()
        lines = [
            "# HELP visionops_inference_total Total inference requests",
            "# TYPE visionops_inference_total counter",
            f'visionops_inference_total {stats["total_inferences"]}',
            "# HELP visionops_inference_errors Total inference errors",
            "# TYPE visionops_inference_errors counter",
            f'visionops_inference_errors {stats["errors"]}',
            "# HELP visionops_inference_error_rate Inference error rate",
            "# TYPE visionops_inference_error_rate gauge",
            f'visionops_inference_error_rate {stats["error_rate"]}',
        ]
        if "latency_ms" in stats:
            lm = stats["latency_ms"]
            lines.extend(
                [
                    "# HELP visionops_inference_latency_mean Mean inference latency in ms",
                    "# TYPE visionops_inference_latency_mean gauge",
                    f'visionops_inference_latency_mean {lm["mean"]}',
                    "# HELP visionops_throughput_fps Estimated throughput in FPS",
                    "# TYPE visionops_throughput_fps gauge",
                    f'visionops_throughput_fps {stats["throughput_fps"]}',
                ]
            )
        return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """稳定 softmax。分类 RKNN 通常输出 logits，这里统一转概率。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-12)


def clip_box_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    box[0] = np.clip(box[0], 0, w - 1)
    box[1] = np.clip(box[1], 0, h - 1)
    box[2] = np.clip(box[2], 0, w - 1)
    box[3] = np.clip(box[3], 0, h - 1)
    return box


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """boxes: [..., 4] in [cx, cy, w, h]."""
    out = np.zeros_like(boxes)
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        ious = box_iou_xyxy(boxes[i], boxes[order[1:]])
        inds = np.where(ious <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thresh: float,
) -> List[int]:
    keep_all: List[int] = []
    for cls in np.unique(class_ids):
        inds = np.where(class_ids == cls)[0]
        keep = nms_xyxy(boxes[inds], scores[inds], iou_thresh)
        keep_all.extend(list(inds[keep]))
    return keep_all


def normalize_yolo_single_output(
    out: np.ndarray,
    expected_channels: int,
) -> np.ndarray:
    """将 YOLO 单输出统一成 [N, C]，支持 [1,C,N] / [1,N,C] / [C,N] / [N,C]。"""
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"不支持的 YOLO 单输出维度: {out.shape}")
    if arr.shape[0] == expected_channels and arr.shape[1] != expected_channels:
        arr = arr.T
    elif arr.shape[1] == expected_channels:
        pass
    else:
        raise ValueError(
            f"YOLO 输出维度与预期不匹配: shape={out.shape}, expected_channels={expected_channels}"
        )
    return arr


def angle_to_radians(angle: float) -> float:
    a = float(angle)
    if abs(a) > 2 * np.pi:
        a = float(np.deg2rad(a))
    return a


def xywhr_to_points(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    angle = angle_to_radians(angle)
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    dx = w / 2.0
    dy = h / 2.0
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    points = corners @ rot.T
    points[:, 0] += cx
    points[:, 1] += cy
    return points


def clip_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    return points


def points_to_xyxy(points: np.ndarray, w: int, h: int) -> np.ndarray:
    box = np.array(
        [float(np.min(points[:, 0])), float(np.min(points[:, 1])),
         float(np.max(points[:, 0])), float(np.max(points[:, 1]))],
        dtype=np.float32,
    )
    return clip_box_xyxy(box, w, h)


def safe_class_name(class_names: List[str], cls_id: int) -> str:
    return class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)



def build_detection_prediction(
    cls_id: int,
    cls_name: str,
    score: float,
    bbox_xyxy: Any,
) -> Dict[str, Any]:
    """构造检测结果，统一补充 bbox 中心点坐标。"""
    box = [float(x) for x in np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)[:4].tolist()]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return {
        "class_id": int(cls_id),
        "class_name": str(cls_name),
        "confidence": round(float(score), 4),
        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "center": [round(cx, 2), round(cy, 2)],
        "center_x": round(cx, 2),
        "center_y": round(cy, 2),
    }





def normalize_yolo_seg_output(out: np.ndarray, num_classes: int) -> np.ndarray:
    """
    将 YOLOv8-seg 单输出统一成 [N, 4 + C + M]。

    常见 RKNN/ONNX 输出：
      - [1, 4 + C + 32, 8400]
      - [1, 8400, 4 + C + 32]
      - [4 + C + 32, 8400]
      - [8400, 4 + C + 32]
    """
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"不支持的 YOLOv8-seg 输出维度: {out.shape}")

    min_channels = 4 + int(num_classes) + 1
    # [C, N] -> [N, C]。通常 C 远小于 N。
    if arr.shape[0] >= min_channels and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    elif arr.shape[1] >= min_channels:
        pass
    else:
        raise ValueError(
            f"YOLOv8-seg 输出维度与类别数不匹配: shape={out.shape}, "
            f"num_classes={num_classes}, min_channels={min_channels}"
        )
    return arr.astype(np.float32)


def normalize_seg_proto(proto: np.ndarray, mask_dim: int) -> np.ndarray:
    """将 proto 统一成 [M, H, W]。"""
    arr = np.asarray(proto, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"不支持的 YOLOv8-seg proto 输出维度: {proto.shape}")

    if arr.shape[0] == mask_dim:
        return arr.astype(np.float32)
    if arr.shape[-1] == mask_dim:
        return arr.transpose(2, 0, 1).astype(np.float32)

    raise ValueError(
        f"YOLOv8-seg proto 通道数不匹配: shape={proto.shape}, mask_dim={mask_dim}"
    )


def crop_mask_by_box(mask: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """在 input 尺度上按 bbox 裁剪 mask，bbox 外置零。"""
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(float(x))) for x in box_xyxy]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    cropped = np.zeros_like(mask, dtype=np.float32)
    if x2 > x1 and y2 > y1:
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return cropped


def project_letterbox_mask_to_original(
    mask_input: np.ndarray,
    meta: Dict[str, Any],
) -> np.ndarray:
    """将 input letterbox 尺度的 mask 还原到原图尺寸。"""
    import cv2

    orig_h, orig_w = meta["orig_shape"]
    input_h, input_w = meta["input_shape"]
    ratio = float(meta.get("ratio", 1.0))
    dw, dh = meta.get("pad", (0.0, 0.0))

    # 与 letterbox 中 left/top 的 round 逻辑保持一致。
    left = max(0, int(round(float(dw) - 0.1)))
    top = max(0, int(round(float(dh) - 0.1)))
    resized_w = min(input_w - left, int(round(orig_w * ratio)))
    resized_h = min(input_h - top, int(round(orig_h * ratio)))

    if resized_w <= 0 or resized_h <= 0:
        return cv2.resize(mask_input, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    valid = mask_input[top: top + resized_h, left: left + resized_w]
    if valid.size == 0:
        return np.zeros((orig_h, orig_w), dtype=np.float32)

    return cv2.resize(valid, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def mask_to_segments(
    binary_mask: np.ndarray,
    max_segments: int = 3,
    max_points_per_segment: int = 200,
) -> Tuple[List[List[List[float]]], float]:
    """
    二值 mask -> polygon segments。

    返回：
      segments: [[[x,y], ...], ...]
      area: mask 像素面积
    """
    import cv2

    mask_u8 = (binary_mask.astype(np.uint8) * 255)
    area = float(np.count_nonzero(mask_u8))
    if area <= 0:
        return [], 0.0

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], area

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    segments: List[List[List[float]]] = []

    for cnt in contours[:max_segments]:
        if cv2.contourArea(cnt) < 4.0:
            continue
        epsilon = max(1.0, 0.002 * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            continue

        if approx.shape[0] > max_points_per_segment:
            idx = np.linspace(0, approx.shape[0] - 1, max_points_per_segment).astype(np.int32)
            approx = approx[idx]

        segment = [[round(float(x), 2), round(float(y), 2)] for x, y in approx.tolist()]
        segments.append(segment)

    return segments, area

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定版 softmax，用于 DFL 解码。"""
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def make_grid_points(h: int, w: int) -> np.ndarray:
    """生成 YOLO 特征图中心点，shape=[H*W, 2]，坐标单位为 grid。"""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    points = np.stack([xx + 0.5, yy + 0.5], axis=-1).reshape(-1, 2)
    return points.astype(np.float32)


def dfl_decode_np(box_logits: np.ndarray, reg_max: int = 16) -> np.ndarray:
    """
    DFL 解码。

    box_logits: [N, 4 * reg_max]
    return: [N, 4]，分别为 l, t, r, b，单位为 grid。
    """
    box_logits = np.asarray(box_logits, dtype=np.float32)
    n = box_logits.shape[0]
    x = box_logits.reshape(n, 4, reg_max)
    prob = softmax_np(x, axis=2)
    proj = np.arange(reg_max, dtype=np.float32)
    dist = (prob * proj).sum(axis=2)
    return dist.astype(np.float32)


def dist2rbox_np(
    distances: np.ndarray,
    angles: np.ndarray,
    anchors: np.ndarray,
    stride: float,
) -> np.ndarray:
    """
    将 DFL 距离 + angle 解码为 xywhr。

    distances: [N, 4]，l,t,r,b，单位 grid
    angles: [N]，弧度
    anchors: [N, 2]，grid center
    stride: 8/16/32

    return: [N, 5]，cx,cy,w,h,angle，坐标单位为 input pixels。
    """
    distances = np.asarray(distances, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32).reshape(-1)
    anchors = np.asarray(anchors, dtype=np.float32)

    lt = distances[:, 0:2]
    rb = distances[:, 2:4]
    wh = lt + rb
    offset = (rb - lt) / 2.0

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    ox = offset[:, 0]
    oy = offset[:, 1]

    cx = ox * cos_a - oy * sin_a + anchors[:, 0]
    cy = ox * sin_a + oy * cos_a + anchors[:, 1]

    return np.stack(
        [cx * stride, cy * stride, wh[:, 0] * stride, wh[:, 1] * stride, angles],
        axis=1,
    ).astype(np.float32)


def is_rockchip_obb_outputs(outputs: List[np.ndarray], num_classes: int) -> bool:
    """判断是否为 airockchip/ultralytics_yolov8 format=rknn 导出的 OBB 多输出结构。"""
    if outputs is None or len(outputs) < 4:
        return False
    expected_c = 64 + int(num_classes)
    has_heads = 0
    has_angle = False
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim == 4 and arr.shape[1] == expected_c and arr.shape[2] in {80, 40, 20}:
            has_heads += 1
        # angle 常见为 [1,1,8400]，有时 RKNN 可能返回 [1,1,1,8400]
        if arr.ndim in {3, 4} and arr.shape[1] == 1 and int(np.prod(arr.shape[2:])) == 8400:
            has_angle = True
    return has_heads >= 3 and has_angle


def decode_rockchip_obb_outputs(
    outputs: List[np.ndarray],
    num_classes: int,
    input_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解码 Rockchip YOLOv8-OBB RKNN 专用多输出。

    典型输出：
      [1, 64 + nc, 80, 80]
      [1, 64 + nc, 40, 40]
      [1, 64 + nc, 20, 20]
      [1, 1, 8400]

    其中 64=4*reg_max，reg_max=16；最后一个输出是 angle.sigmoid()。

    return:
      boxes_xywhr: [N, 5]，cx,cy,w,h,angle，坐标单位为 input pixels
      class_scores: [N, nc]
      angles: [N]
    """
    reg_max = 16
    expected_c = reg_max * 4 + int(num_classes)

    detect_heads: List[np.ndarray] = []
    angle_out: Optional[np.ndarray] = None

    for out in outputs:
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[1] == expected_c:
            detect_heads.append(arr)
        elif arr.ndim in {3, 4} and arr.shape[1] == 1 and int(np.prod(arr.shape[2:])) == 8400:
            angle_out = arr.reshape(arr.shape[0], 1, -1)

    if len(detect_heads) != 3:
        raise ValueError(
            f"Rockchip OBB 应有 3 个检测头，实际得到 {len(detect_heads)} 个，"
            f"shapes={[tuple(np.asarray(o).shape) for o in outputs]}"
        )
    if angle_out is None:
        raise ValueError(
            f"Rockchip OBB 缺少 angle 输出，shapes={[tuple(np.asarray(o).shape) for o in outputs]}"
        )

    # 按 80 -> 40 -> 20 排序，与 angle 的 8400 展平顺序对应。
    detect_heads = sorted(detect_heads, key=lambda x: x.shape[2], reverse=True)
    strides = [8.0, 16.0, 32.0]

    angle_flat = angle_out.reshape(-1).astype(np.float32)
    expected_n = sum(int(h.shape[2] * h.shape[3]) for h in detect_heads)
    if angle_flat.size < expected_n:
        raise ValueError(
            f"angle 输出长度不足: got={angle_flat.size}, expected={expected_n}, "
            f"shape={angle_out.shape}"
        )
    if angle_flat.size > expected_n:
        angle_flat = angle_flat[:expected_n]

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    start = 0
    for head, stride in zip(detect_heads, strides):
        _, c, h, w = head.shape
        if c != expected_c:
            raise ValueError(f"检测头通道数不匹配: got={c}, expected={expected_c}")

        # [1, C, H, W] -> [H*W, C]
        pred = head[0].transpose(1, 2, 0).reshape(-1, c)
        n = pred.shape[0]
        end = start + n

        box_logits = pred[:, : reg_max * 4]
        cls_logits = pred[:, reg_max * 4 : reg_max * 4 + num_classes]

        # Rockchip OBB 的 Detect.forward(self, x, "Obb") 返回 raw cls logits。
        class_scores = sigmoid(cls_logits)

        # OBB.forward(format=rknn) 返回的是 angle.sigmoid()，需还原为官方角度定义。
        angle_sigmoid = angle_flat[start:end]
        angles = (angle_sigmoid - 0.25) * np.pi

        distances = dfl_decode_np(box_logits, reg_max=reg_max)
        anchors = make_grid_points(h, w)
        boxes_xywhr = dist2rbox_np(
            distances=distances,
            angles=angles,
            anchors=anchors,
            stride=stride,
        )

        all_boxes.append(boxes_xywhr)
        all_scores.append(class_scores)
        start = end

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    angles = boxes[:, 4]
    return boxes, scores, angles


def parse_input_size(value: Optional[str], default: List[int]) -> List[int]:
    """
    支持从环境变量读取 INPUT_SIZE：
    - "224,224"
    - "224 224"
    """
    if not value:
        return default
    try:
        normalized = value.replace(",", " ").split()
        if len(normalized) != 2:
            raise ValueError
        h, w = int(normalized[0]), int(normalized[1])
        if h <= 0 or w <= 0:
            raise ValueError
        return [h, w]
    except Exception:
        logger.warning(f"INPUT_SIZE 环境变量无效: {value}，使用默认值: {default}")
        return default


def load_class_names_config(path: Optional[str]) -> Tuple[Optional[List[str]], Optional[int], Optional[str], Optional[List[int]]]:
    """
    读取类别配置文件。

    支持字段：
    task: classification / detection，可选
    num_classes: 2
    class_names: [no, yes]
    input_size: [224, 224]，可选
    """
    if not path:
        return None, None, None, None

    p = Path(path)
    if not p.exists():
        logger.warning(f"类别配置文件不存在，继续使用默认类别配置: {p}")
        return None, None, None, None

    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"读取类别配置文件失败，继续使用默认类别配置: {p}, err={e}")
        return None, None, None, None

    class_names = data.get("class_names")
    num_classes = data.get("num_classes")
    task = data.get("task")
    input_size = data.get("input_size")

    if not isinstance(class_names, list) or not class_names:
        logger.warning(f"类别配置文件中的 class_names 无效: {p}")
        return None, None, task, input_size if isinstance(input_size, list) else None

    if num_classes is None:
        num_classes = len(class_names)

    try:
        num_classes = int(num_classes)
    except Exception:
        logger.warning(f"类别配置文件中的 num_classes 无效: {p}")
        return None, None, task, input_size if isinstance(input_size, list) else None

    if len(class_names) != num_classes:
        logger.warning(
            f"class_names 与 num_classes 不一致，继续使用默认类别配置: "
            f"len(class_names)={len(class_names)}, num_classes={num_classes}"
        )
        return None, None, task, input_size if isinstance(input_size, list) else None

    if input_size is not None:
        try:
            if not isinstance(input_size, list) or len(input_size) != 2:
                raise ValueError
            input_size = [int(input_size[0]), int(input_size[1])]
        except Exception:
            logger.warning(f"类别配置文件中的 input_size 无效，忽略: {input_size}")
            input_size = None

    return class_names, num_classes, task, input_size


def _dfl(position: np.ndarray) -> np.ndarray:
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num

    x = position.reshape(n, p_num, mc, h, w)
    x = np.exp(x - np.max(x, axis=2, keepdims=True))
    x = x / np.sum(x, axis=2, keepdims=True)

    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = (x * acc).sum(axis=2)
    return y


def _rockchip_box_process(position: np.ndarray, img_size=(640, 640)) -> np.ndarray:
    grid_h, grid_w = position.shape[2:4]

    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1).astype(np.float32)

    stride = np.array(
        [img_size[1] // grid_h, img_size[0] // grid_w],
        dtype=np.float32,
    ).reshape(1, 2, 1, 1)

    position = _dfl(position)

    box_xy1 = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]

    xyxy = np.concatenate((box_xy1 * stride, box_xy2 * stride), axis=1)
    return xyxy


def _flatten_output(x: np.ndarray) -> np.ndarray:
    ch = x.shape[1]
    x = x.transpose(0, 2, 3, 1)
    return x.reshape(-1, ch)


def _decode_rockchip_yolov8_outputs(
    outputs: List[np.ndarray],
    num_classes: int,
    conf_threshold: float,
    iou_threshold: float,
    meta: Dict[str, Any],
    img_size: Tuple[int, int] = (640, 640),
) -> Dict[str, Any]:
    """
    适配 airockchip/ultralytics_yolov8 导出的 RKNN 多输出格式。

    典型输出顺序：
    0: box_80  [1, 64, 80, 80]
    1: cls_80  [1, C, 80, 80]
    2: sum_80  [1, 1, 80, 80]  可忽略
    3: box_40
    4: cls_40
    5: sum_40
    6: box_20
    7: cls_20
    8: sum_20
    """
    if outputs is None or len(outputs) < 6:
        return {"predictions": [], "task": "detection"}

    boxes_all = []
    cls_all = []

    if len(outputs) >= 9:
        group_size = 3
        branch_num = 3
    else:
        group_size = 2
        branch_num = len(outputs) // 2

    for i in range(branch_num):
        box_out = outputs[group_size * i]
        cls_out = outputs[group_size * i + 1]

        if box_out.ndim != 4 or cls_out.ndim != 4:
            logger.warning(
                f"Rockchip YOLOv8 输出维度异常: "
                f"box_out={box_out.shape}, cls_out={cls_out.shape}"
            )
            continue

        boxes = _rockchip_box_process(box_out, img_size=img_size)
        boxes = _flatten_output(boxes)

        cls_scores = _flatten_output(cls_out)

        if cls_scores.shape[1] > num_classes:
            cls_scores = cls_scores[:, :num_classes]

        # Rockchip 导出的 cls 分支通常已经带 sigmoid；若值域像 logits，再做 sigmoid。
        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = sigmoid(cls_scores)

        boxes_all.append(boxes)
        cls_all.append(cls_scores)

    if not boxes_all:
        return {"predictions": [], "task": "detection"}

    boxes = np.concatenate(boxes_all, axis=0)
    cls_scores = np.concatenate(cls_all, axis=0)

    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(len(cls_scores)), class_ids]

    keep_mask = scores >= conf_threshold
    if not np.any(keep_mask):
        return {"predictions": [], "task": "detection"}

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    # 映射回原图
    ratio = meta["ratio"]
    dw, dh = meta["pad"]
    orig_h, orig_w = meta["orig_shape"]

    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= max(ratio, 1e-6)

    for i in range(len(boxes)):
        boxes[i] = clip_box_xyxy(boxes[i], orig_w, orig_h)

    wh = boxes[:, 2:4] - boxes[:, 0:2]
    valid_mask = (wh[:, 0] > 2.0) & (wh[:, 1] > 2.0)
    if not np.any(valid_mask):
        return {"predictions": [], "task": "detection"}

    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    class_ids = class_ids[valid_mask]

    keep = multiclass_nms(
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        iou_thresh=iou_threshold,
    )

    predictions = []
    for i in keep:
        cls_id = int(class_ids[i])
        predictions.append(
            build_detection_prediction(
                cls_id=cls_id,
                cls_name=str(cls_id),
                score=float(scores[i]),
                bbox_xyxy=boxes[i],
            )
        )

    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "predictions": predictions,
        "task": "detection",
    }


def is_rockchip_seg_outputs(
    outputs: List[np.ndarray],
    num_classes: int,
    mask_dim: int = 32,
) -> bool:
    """
    判断是否为 airockchip/ultralytics_yolov8 format=rknn 导出的 YOLOv8-seg 多输出结构。

    当前已验证的输出顺序：
      0: box_80        [1, 64, 80, 80]
      1: cls_80        [1, C, 80, 80]
      2: score/sum_80  [1, 1, 80, 80]   可忽略
      3: mask_80       [1, 32, 80, 80]
      4: box_40        [1, 64, 40, 40]
      5: cls_40        [1, C, 40, 40]
      6: score/sum_40  [1, 1, 40, 40]   可忽略
      7: mask_40       [1, 32, 40, 40]
      8: box_20        [1, 64, 20, 20]
      9: cls_20        [1, C, 20, 20]
     10: score/sum_20  [1, 1, 20, 20]   可忽略
     11: mask_20       [1, 32, 20, 20]
     12: proto         [1, 32, 160, 160]
    """
    if outputs is None or len(outputs) < 13:
        return False

    shapes = [tuple(np.asarray(o).shape) for o in outputs]
    expected = [
        (1, 64, 80, 80),
        (1, int(num_classes), 80, 80),
        (1, 1, 80, 80),
        (1, mask_dim, 80, 80),
        (1, 64, 40, 40),
        (1, int(num_classes), 40, 40),
        (1, 1, 40, 40),
        (1, mask_dim, 40, 40),
        (1, 64, 20, 20),
        (1, int(num_classes), 20, 20),
        (1, 1, 20, 20),
        (1, mask_dim, 20, 20),
        (1, mask_dim, 160, 160),
    ]

    if len(shapes) >= 13 and shapes[:13] == expected:
        return True

    # 兼容未来输出顺序略有变化的情况：按关键特征宽松判断。
    has_box_heads = 0
    has_cls_heads = 0
    has_coeff_heads = 0
    has_proto = False

    for arr in outputs:
        a = np.asarray(arr)
        if a.ndim != 4 or a.shape[0] != 1:
            continue
        c, h, w = int(a.shape[1]), int(a.shape[2]), int(a.shape[3])
        if (h, w) in {(80, 80), (40, 40), (20, 20)}:
            if c == 64:
                has_box_heads += 1
            elif c == int(num_classes):
                has_cls_heads += 1
            elif c == mask_dim:
                has_coeff_heads += 1
        elif (h, w) == (160, 160) and c == mask_dim:
            has_proto = True

    return has_box_heads >= 3 and has_cls_heads >= 3 and has_coeff_heads >= 3 and has_proto


def decode_rockchip_seg_outputs(
    outputs: List[np.ndarray],
    num_classes: int,
    input_size: Tuple[int, int],
    mask_dim: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    解码 Rockchip YOLOv8-seg RKNN 多输出。

    return:
      boxes_input_xyxy: [N, 4]，input letterbox 尺度 xyxy
      class_scores:    [N, C]
      mask_coeffs:     [N, M]
      proto:           [M, Hp, Wp]
    """
    if not is_rockchip_seg_outputs(outputs, num_classes=num_classes, mask_dim=mask_dim):
        raise ValueError(
            "不是已知 Rockchip YOLOv8-seg 多输出结构，"
            f"shapes={[tuple(np.asarray(o).shape) for o in outputs]}"
        )

    # 当前 Rockchip seg 输出顺序为每个尺度 4 个输出：box / cls / score_sum / mask_coeff。
    groups = [
        (0, 1, 3),   # 80x80
        (4, 5, 7),   # 40x40
        (8, 9, 11),  # 20x20
    ]

    boxes_all: List[np.ndarray] = []
    cls_all: List[np.ndarray] = []
    coeff_all: List[np.ndarray] = []

    for box_idx, cls_idx, coeff_idx in groups:
        box_out = np.asarray(outputs[box_idx], dtype=np.float32)
        cls_out = np.asarray(outputs[cls_idx], dtype=np.float32)
        coeff_out = np.asarray(outputs[coeff_idx], dtype=np.float32)

        boxes = _rockchip_box_process(box_out, img_size=input_size)
        boxes = _flatten_output(boxes).astype(np.float32)

        cls_scores = _flatten_output(cls_out).astype(np.float32)
        if cls_scores.shape[1] > int(num_classes):
            cls_scores = cls_scores[:, : int(num_classes)]

        # airockchip 的 cls 分支多数已经是概率；如果值域像 logits，再做 sigmoid。
        if cls_scores.size > 0 and (float(np.max(cls_scores)) > 1.0 or float(np.min(cls_scores)) < 0.0):
            cls_scores = sigmoid(cls_scores)

        coeffs = _flatten_output(coeff_out).astype(np.float32)
        if coeffs.shape[1] > mask_dim:
            coeffs = coeffs[:, :mask_dim]

        boxes_all.append(boxes)
        cls_all.append(cls_scores)
        coeff_all.append(coeffs)

    boxes_input_xyxy = np.concatenate(boxes_all, axis=0)
    class_scores = np.concatenate(cls_all, axis=0)
    mask_coeffs = np.concatenate(coeff_all, axis=0)

    proto = normalize_seg_proto(outputs[12], mask_dim=mask_dim)

    return boxes_input_xyxy, class_scores, mask_coeffs, proto


# ────────────────────────────────────────────────
# RKNN 推理引擎
# ────────────────────────────────────────────────
class RKNNInferenceEngine:
    NPU_CORE_MAP = {
        "auto": 0b000,
        "core_0": 0b001,
        "core_1": 0b010,
        "core_2": 0b100,
        "core_0_1": 0b011,
        "core_0_1_2": 0b111,
    }

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.rknn = None
        self.is_loaded = False
        self.metrics = MetricsCollector()
        self._lock = threading.Lock()
        self._simulate_mode = False
        self._shape_logged = False
        self._postprocess_format_logged = False

    def load_model(self) -> bool:
        model_path = self.config.model_path
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False

        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            logger.warning("rknnlite2 未安装，进入模拟模式")
            self.is_loaded = True
            self._simulate_mode = True
            return True

        self.rknn = RKNNLite()
        logger.info(f"加载RKNN模型: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            logger.error(f"加载模型失败: {ret}")
            return False

        core_mask = self.NPU_CORE_MAP.get(self.config.npu_core.lower(), 0)
        logger.info(f"初始化NPU，核心配置: {self.config.npu_core}")
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            logger.error(f"初始化NPU运行时失败: {ret}")
            return False

        self.is_loaded = True
        logger.info(f"✓ RKNN模型加载成功，task={self.config.normalized_task()}, input_size={self.config.input_size}")
        self._warmup()
        return True

    def _warmup(self):
        logger.info(f"预热推理 ({self.config.warmup_runs} 次)...")
        h, w = self.config.input_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(self.config.warmup_runs):
            try:
                self._run_inference_raw(dummy)
            except Exception as e:
                logger.warning(f"预热推理异常，可忽略但建议修复: {e}")
        logger.info("预热完成")

    def _letterbox(
        self,
        image: np.ndarray,
        new_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        import cv2

        orig_h, orig_w = image.shape[:2]
        new_h, new_w = new_shape

        r = min(new_w / orig_w, new_h / orig_h)
        resized_w, resized_h = int(round(orig_w * r)), int(round(orig_h * r))

        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
        dw = (new_w - resized_w) / 2
        dh = (new_h - resized_h) / 2

        left = int(round(dw - 0.1))
        top = int(round(dh - 0.1))
        canvas[top:top + resized_h, left:left + resized_w] = resized

        return canvas, r, (dw, dh)

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        预处理分任务处理：
        - detection：letterbox + BGR->RGB，保持 YOLO 坐标映射信息
        - classification：直接 resize + BGR->RGB，不做 letterbox

        注意：分类 RKNN 的 mean/std 已在转换时写入 rknn.config，
        因此这里保持 uint8 RGB 输入，不再手动 normalize。
        """
        import cv2

        orig_h, orig_w = image.shape[:2]
        h, w = self.config.input_size

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        task = self.config.normalized_task()

        if task == "classification":
            img = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            meta = {
                "task": "classification",
                "orig_shape": (orig_h, orig_w),
                "input_shape": (h, w),
                "resize": True,
            }
            return img, meta

        # detection / obb_detection：保持 letterbox + BGR -> RGB
        img, ratio, (dw, dh) = self._letterbox(image, (h, w))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        meta = {
            "task": task,
            "orig_shape": (orig_h, orig_w),
            "input_shape": (h, w),
            "ratio": ratio,
            "pad": (dw, dh),
        }
        return img, meta

    def _run_inference_raw(self, image_hwc: np.ndarray) -> Optional[List[np.ndarray]]:
        if self._simulate_mode:
            c = self.config.num_classes
            if self.config.normalized_task() == "classification":
                # 模拟分类 logits: [1, C]
                fake = np.random.randn(1, c).astype(np.float32)
                return [fake]

            n = 20
            if self.config.normalized_task() == "obb_detection":
                # 模拟 OBB 输出：[1, 4 + C + 1, N]
                fake = np.random.rand(1, 4 + c + 1, n).astype(np.float32)
                fake[:, 0:4, :] *= float(self.config.input_size[0])
                fake[:, 4:4 + c, :] = np.random.rand(1, c, n).astype(np.float32)
                fake[:, 4 + c, :] = np.random.rand(1, n).astype(np.float32) * np.pi / 2
                return [fake]

            if self.config.normalized_task() == "segmentation":
                # 模拟 YOLOv8-seg 输出：[1, 4 + C + 32, N] + [1, 32, H/4, W/4]
                mask_dim = 32
                h, w = self.config.input_size
                fake_det = np.random.rand(1, 4 + c + mask_dim, n).astype(np.float32)
                fake_det[:, 0:4, :] *= float(max(h, w))
                fake_proto = np.random.randn(1, mask_dim, max(1, h // 4), max(1, w // 4)).astype(np.float32)
                return [fake_det, fake_proto]

            # 模拟检测输出：[1, N, 5 + C]
            fake = np.random.rand(1, n, 5 + c).astype(np.float32)
            return [fake]

        # RKNN 常见需要 4D NHWC 输入，这里显式补 batch 维。
        inp = np.expand_dims(image_hwc, axis=0)  # [1, H, W, 3]

        with self._lock:
            outputs = self.rknn.inference(inputs=[inp])

        if self.config.debug_shapes and not self._shape_logged and outputs is not None:
            for i, out in enumerate(outputs):
                logger.info(
                    f"[DEBUG] output[{i}] shape={getattr(out, 'shape', None)}, "
                    f"dtype={getattr(out, 'dtype', None)}"
                )
            self._shape_logged = True

        return outputs

    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        t0 = time.perf_counter()
        try:
            preprocessed, meta = self._preprocess(image)
            outputs = self._run_inference_raw(preprocessed)

            task = self.config.normalized_task()
            if task == "classification":
                result = self._postprocess_classification(outputs, meta)
            elif task == "detection":
                result = self._postprocess_detection(outputs, meta)
            elif task == "obb_detection":
                result = self._postprocess_obb(outputs, meta)
            elif task == "segmentation":
                result = self._postprocess_segmentation(outputs, meta)
            else:
                raise ValueError(f"不支持的任务类型: {task}")

            latency_ms = (time.perf_counter() - t0) * 1000
            self.metrics.record(latency_ms, success=True)
            result["latency_ms"] = round(latency_ms, 2)
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            self.metrics.record(latency_ms, success=False)
            logger.exception(f"推理异常: {e}")
            raise

    def _postprocess_classification(
        self,
        outputs: Optional[List[np.ndarray]],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Classification 后处理。

        支持常见 RKNN 输出：
        - [1, C]
        - [C]
        - [1, C, 1, 1]
        - 其他可 flatten 成 C 的输出

        输出逻辑：logits -> softmax -> top1 / topk。
        """
        if outputs is None or len(outputs) == 0:
            return {
                "task": "classification",
                "prediction": None,
                "topk": [],
                "message": "empty outputs",
            }

        shape_info = [tuple(o.shape) for o in outputs]
        out = outputs[0]
        logits = np.asarray(out, dtype=np.float32).reshape(-1)

        num_classes = int(self.config.num_classes)
        if logits.size < num_classes:
            return {
                "task": "classification",
                "prediction": None,
                "topk": [],
                "message": "output size smaller than num_classes",
                "output_shapes": shape_info,
                "num_classes": num_classes,
            }

        # 有些 RKNN 输出可能多出维度，这里只取前 num_classes。
        if logits.size > num_classes:
            logger.warning(
                f"classification 输出维度大于 num_classes，将截取前 {num_classes} 个值: "
                f"logits.size={logits.size}"
            )
            logits = logits[:num_classes]

        probs = softmax(logits)

        topk = max(1, min(int(self.config.topk), num_classes))
        top_indices = probs.argsort()[::-1][:topk]

        topk_results = []
        for idx in top_indices:
            cls_id = int(idx)
            cls_name = (
                self.config.class_names[cls_id]
                if cls_id < len(self.config.class_names)
                else str(cls_id)
            )
            topk_results.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(float(probs[cls_id]), 6),
                    "logit": round(float(logits[cls_id]), 6),
                }
            )

        top1 = topk_results[0] if topk_results else None

        return {
            "task": "classification",
            "prediction": top1,
            "topk": topk_results,
            "num_classes": num_classes,
            "class_names": self.config.class_names,
            "output_shapes": shape_info,
            "meta": meta,
        }

    def _postprocess_detection(
        self,
        outputs: Optional[List[np.ndarray]],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Detection 后处理入口。

        支持两种格式：
        1. 普通 Ultralytics 单输出：
           [1, 4+C, 8400] 或 [1, 8400, 4+C]
        2. Rockchip YOLOv8 多输出：
           [box_80, cls_80, sum_80, box_40, cls_40, sum_40, box_20, cls_20, sum_20]
        """
        if outputs is None or len(outputs) == 0:
            return {
                "predictions": [],
                "task": "detection",
                "message": "empty outputs",
            }

        shape_info = [tuple(o.shape) for o in outputs]

        # --------------------------------------------------
        # Rockchip YOLOv8 多输出格式
        # --------------------------------------------------
        if len(outputs) >= 6:
            if not self._postprocess_format_logged:
                logger.info(f"[POSTPROCESS] 使用 Rockchip YOLOv8 多输出后处理: {shape_info}")
                self._postprocess_format_logged = True

            result = _decode_rockchip_yolov8_outputs(
                outputs=outputs,
                num_classes=self.config.num_classes,
                conf_threshold=self.config.conf_threshold,
                iou_threshold=self.config.nms_threshold,
                meta=meta,
                img_size=tuple(self.config.input_size),
            )

            # 补 class_name
            for pred in result.get("predictions", []):
                cls_id = pred["class_id"]
                pred["class_name"] = (
                    self.config.class_names[cls_id]
                    if cls_id < len(self.config.class_names)
                    else str(cls_id)
                )

            return result

        # --------------------------------------------------
        # 单输出格式
        # --------------------------------------------------
        out = outputs[0]

        if out.ndim == 3 and out.shape[0] == 1 and out.shape[1] == (4 + self.config.num_classes):
            pred = out[0].transpose(1, 0)
            return self._decode_rknn_1x6x8400(pred, meta)

        if out.ndim == 3 and out.shape[0] == 1 and out.shape[2] == (4 + self.config.num_classes):
            pred = out[0]
            return self._decode_rknn_1x6x8400(pred, meta)

        if out.ndim == 2 and out.shape[1] == (4 + self.config.num_classes):
            return self._decode_rknn_1x6x8400(out, meta)

        # 兼容旧式 [N, 5+C] 输出。
        if out.ndim == 2 and out.shape[1] >= (5 + self.config.num_classes):
            return self._decode_flat_predictions(out, meta)

        logger.warning(f"未匹配到已实现的 detection 输出格式: {shape_info}")
        return {
            "predictions": [],
            "task": "detection",
            "message": "unsupported output format",
            "output_shapes": shape_info,
        }

    def _decode_rknn_1x6x8400(
        self,
        pred: np.ndarray,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        适配 RKNN 单输出：
            pred.shape == (N, 4 + C)
        假设格式：
            [x, y, w, h, cls0, cls1, ...]
        """
        if pred is None or pred.size == 0:
            return {"predictions": [], "task": "detection"}

        num_classes = self.config.num_classes
        if pred.shape[1] != 4 + num_classes:
            logger.warning(f"输出维度不符合预期，pred.shape={pred.shape}")
            return {"predictions": [], "task": "detection"}

        boxes = pred[:, :4].astype(np.float32)
        cls_scores = pred[:, 4:4 + num_classes].astype(np.float32)

        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = sigmoid(cls_scores)

        class_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(cls_scores)), class_ids]

        keep_mask = scores >= self.config.conf_threshold
        if not np.any(keep_mask):
            return {"predictions": [], "task": "detection"}

        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        boxes_xyxy = xywh_to_xyxy(boxes)

        ratio = meta["ratio"]
        dw, dh = meta["pad"]
        orig_h, orig_w = meta["orig_shape"]

        boxes_xyxy[:, [0, 2]] -= dw
        boxes_xyxy[:, [1, 3]] -= dh
        boxes_xyxy /= max(ratio, 1e-6)

        for i in range(len(boxes_xyxy)):
            boxes_xyxy[i] = clip_box_xyxy(boxes_xyxy[i], orig_w, orig_h)

        wh = boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2]
        valid_mask = (wh[:, 0] > 2.0) & (wh[:, 1] > 2.0)
        if not np.any(valid_mask):
            return {"predictions": [], "task": "detection"}

        boxes_xyxy = boxes_xyxy[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]

        keep = multiclass_nms(
            boxes=boxes_xyxy,
            scores=scores,
            class_ids=class_ids,
            iou_thresh=self.config.nms_threshold,
        )

        predictions = []
        for i in keep:
            cls_id = int(class_ids[i])
            cls_name = (
                self.config.class_names[cls_id]
                if cls_id < len(self.config.class_names)
                else str(cls_id)
            )
            predictions.append(
                build_detection_prediction(
                    cls_id=cls_id,
                    cls_name=cls_name,
                    score=float(scores[i]),
                    bbox_xyxy=boxes_xyxy[i],
                )
            )

        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return {
            "predictions": predictions,
            "task": "detection",
        }

    def _decode_flat_predictions(
        self,
        pred: np.ndarray,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        解析 [N, 5+C]：
        [cx, cy, w, h, obj, cls0, cls1, ...]
        """
        if pred.size == 0:
            return {"predictions": [], "task": "detection"}

        num_classes = self.config.num_classes
        if pred.shape[-1] < 5 + num_classes:
            logger.warning(f"输出维度不足，无法解析检测头: {pred.shape}")
            return {"predictions": [], "task": "detection"}

        boxes_xywh = pred[:, 0:4].astype(np.float32)
        obj = pred[:, 4].astype(np.float32)
        cls_scores = pred[:, 5:5 + num_classes].astype(np.float32)

        if obj.max() > 1.0 or obj.min() < 0.0:
            obj = sigmoid(obj)
        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = sigmoid(cls_scores)

        class_ids = np.argmax(cls_scores, axis=1)
        class_conf = cls_scores[np.arange(len(cls_scores)), class_ids]
        scores = obj * class_conf

        mask = scores >= self.config.conf_threshold
        if not np.any(mask):
            return {"predictions": [], "task": "detection"}

        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        boxes_xyxy = xywh_to_xyxy(boxes_xywh)

        ratio = meta["ratio"]
        dw, dh = meta["pad"]
        orig_h, orig_w = meta["orig_shape"]

        boxes_xyxy[:, [0, 2]] -= dw
        boxes_xyxy[:, [1, 3]] -= dh
        boxes_xyxy /= max(ratio, 1e-6)

        for i in range(len(boxes_xyxy)):
            boxes_xyxy[i] = clip_box_xyxy(boxes_xyxy[i], orig_w, orig_h)

        keep = multiclass_nms(
            boxes=boxes_xyxy,
            scores=scores,
            class_ids=class_ids,
            iou_thresh=self.config.nms_threshold,
        )

        predictions = []
        for i in keep:
            cls_id = int(class_ids[i])
            predictions.append(
                build_detection_prediction(
                    cls_id=cls_id,
                    cls_name=(
                        self.config.class_names[cls_id]
                        if cls_id < len(self.config.class_names)
                        else str(cls_id)
                    ),
                    score=float(scores[i]),
                    bbox_xyxy=boxes_xyxy[i],
                )
            )

        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return {
            "predictions": predictions,
            "task": "detection",
        }

    def _postprocess_segmentation(
        self,
        outputs: Optional[List[np.ndarray]],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        YOLOv8 Segmentation 后处理。

        兼容两种输出：
        1. 标准单输出 + proto：
             output[0]: [1, 4 + num_classes + mask_dim, 8400]
             output[1]: [1, mask_dim, mask_h, mask_w]

        2. Rockchip format=rknn 多输出 + proto：
             [box_80, cls_80, sum_80, mask_80,
              box_40, cls_40, sum_40, mask_40,
              box_20, cls_20, sum_20, mask_20,
              proto]
        """
        if outputs is None or len(outputs) < 2:
            return {
                "predictions": [],
                "task": "segmentation",
                "message": "empty outputs or missing proto output",
                "output_shapes": [tuple(np.asarray(o).shape) for o in outputs] if outputs else [],
            }

        shape_info = [tuple(np.asarray(o).shape) for o in outputs]
        num_classes = int(self.config.num_classes)
        input_h, input_w = meta["input_shape"]
        orig_h, orig_w = meta["orig_shape"]
        ratio = float(meta["ratio"])
        dw, dh = meta["pad"]

        # --------------------------------------------------
        # Rockchip YOLOv8-seg 多输出格式
        # --------------------------------------------------
        if is_rockchip_seg_outputs(outputs, num_classes=num_classes, mask_dim=32):
            mask_dim = 32
            boxes_input_xyxy, class_scores, mask_coeffs, proto = decode_rockchip_seg_outputs(
                outputs=outputs,
                num_classes=num_classes,
                input_size=tuple(self.config.input_size),
                mask_dim=mask_dim,
            )
            decode_format = "rockchip_yolov8_seg_multi_output"

            if not self._postprocess_format_logged:
                logger.info(
                    f"[POSTPROCESS] 使用 Rockchip YOLOv8 Segmentation 多输出后处理: "
                    f"shapes={shape_info}, num_classes={num_classes}, mask_dim={mask_dim}"
                )
                self._postprocess_format_logged = True

        # --------------------------------------------------
        # 标准 YOLOv8-seg 单输出 + proto 格式
        # --------------------------------------------------
        else:
            det_out = outputs[0]
            proto_out = outputs[1]
            preds = normalize_yolo_seg_output(det_out, num_classes=num_classes)

            mask_dim = int(preds.shape[1] - 4 - num_classes)
            if mask_dim <= 0:
                return {
                    "predictions": [],
                    "task": "segmentation",
                    "message": f"invalid mask_dim={mask_dim}",
                    "output_shapes": shape_info,
                }

            proto = normalize_seg_proto(proto_out, mask_dim=mask_dim)

            boxes_xywh = preds[:, 0:4].astype(np.float32)
            class_scores = preds[:, 4:4 + num_classes].astype(np.float32)
            mask_coeffs = preds[:, 4 + num_classes:4 + num_classes + mask_dim].astype(np.float32)

            if class_scores.size > 0 and (float(np.max(class_scores)) > 1.0 or float(np.min(class_scores)) < 0.0):
                class_scores = sigmoid(class_scores)

            # 单输出模型有时可能输出 0~1 归一化坐标。
            if boxes_xywh.size > 0 and float(np.nanmax(boxes_xywh[:, 0:4])) <= 2.0:
                boxes_xywh[:, [0, 2]] *= float(input_w)
                boxes_xywh[:, [1, 3]] *= float(input_h)

            boxes_input_xyxy = xywh_to_xyxy(boxes_xywh)
            decode_format = "yolov8_seg_single_output_with_proto"

            if not self._postprocess_format_logged:
                logger.info(
                    f"[POSTPROCESS] 使用 YOLOv8 Segmentation 单输出后处理: "
                    f"shapes={shape_info}, num_classes={num_classes}, mask_dim={mask_dim}"
                )
                self._postprocess_format_logged = True

        if class_scores.size == 0:
            return {
                "predictions": [],
                "task": "segmentation",
                "output_shapes": shape_info,
                "decode_format": decode_format,
                "message": "empty class scores",
            }

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]

        if os.getenv("SEG_DEBUG", "0") == "1":
            logger.info(
                "[SEG DEBUG] decoded stats: "
                f"score_min={float(np.min(scores)):.6f}, "
                f"score_max={float(np.max(scores)):.6f}, "
                f"box_min={float(np.min(boxes_input_xyxy)):.6f}, "
                f"box_max={float(np.max(boxes_input_xyxy)):.6f}, "
                f"coeff_min={float(np.min(mask_coeffs)):.6f}, "
                f"coeff_max={float(np.max(mask_coeffs)):.6f}, "
                f"proto_min={float(np.min(proto)):.6f}, "
                f"proto_max={float(np.max(proto)):.6f}, "
                f"threshold={self.config.conf_threshold}, "
                f"decode_format={decode_format}"
            )

        keep_mask = scores >= self.config.conf_threshold
        if not np.any(keep_mask):
            return {
                "predictions": [],
                "task": "segmentation",
                "output_shapes": shape_info,
                "decode_format": decode_format,
                "message": "no predictions above confidence threshold",
                "debug": {
                    "score_max": round(float(np.max(scores)), 6) if scores.size else None,
                    "score_min": round(float(np.min(scores)), 6) if scores.size else None,
                    "conf_threshold": self.config.conf_threshold,
                },
            }

        candidate_indices = np.where(keep_mask)[0]

        # 防止低阈值时 NMS 处理过多候选，边缘端先保留 topK 候选。
        max_candidates = int(os.getenv("SEG_MAX_CANDIDATES", "1000"))
        if candidate_indices.size > max_candidates:
            top_order = scores[candidate_indices].argsort()[::-1][:max_candidates]
            candidate_indices = candidate_indices[top_order]

        boxes_input_xyxy = boxes_input_xyxy[candidate_indices]
        scores = scores[candidate_indices]
        class_ids = class_ids[candidate_indices]
        mask_coeffs = mask_coeffs[candidate_indices]

        for i in range(len(boxes_input_xyxy)):
            boxes_input_xyxy[i] = clip_box_xyxy(boxes_input_xyxy[i], input_w, input_h)

        boxes_orig_xyxy = boxes_input_xyxy.copy()
        boxes_orig_xyxy[:, [0, 2]] -= float(dw)
        boxes_orig_xyxy[:, [1, 3]] -= float(dh)
        boxes_orig_xyxy /= max(ratio, 1e-6)

        for i in range(len(boxes_orig_xyxy)):
            boxes_orig_xyxy[i] = clip_box_xyxy(boxes_orig_xyxy[i], orig_w, orig_h)

        wh = boxes_orig_xyxy[:, 2:4] - boxes_orig_xyxy[:, 0:2]
        valid_mask = (wh[:, 0] > 2.0) & (wh[:, 1] > 2.0)
        if not np.any(valid_mask):
            return {
                "predictions": [],
                "task": "segmentation",
                "output_shapes": shape_info,
                "decode_format": decode_format,
                "message": "no valid segmentation boxes",
            }

        boxes_input_xyxy = boxes_input_xyxy[valid_mask]
        boxes_orig_xyxy = boxes_orig_xyxy[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]
        mask_coeffs = mask_coeffs[valid_mask]

        keep = multiclass_nms(
            boxes=boxes_orig_xyxy,
            scores=scores,
            class_ids=class_ids,
            iou_thresh=self.config.nms_threshold,
        )

        if not keep:
            return {
                "predictions": [],
                "task": "segmentation",
                "output_shapes": shape_info,
                "decode_format": decode_format,
                "message": "all predictions removed by nms",
            }

        proto_flat = proto.reshape(mask_dim, -1)
        predictions: List[Dict[str, Any]] = []

        import cv2

        for idx in keep:
            cls_id = int(class_ids[idx])
            score = float(scores[idx])
            bbox_orig = boxes_orig_xyxy[idx]
            bbox_input = boxes_input_xyxy[idx]
            coeff = mask_coeffs[idx]

            mask_prob = sigmoid(coeff @ proto_flat).reshape(proto.shape[1], proto.shape[2])
            mask_input = cv2.resize(mask_prob, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            mask_input = crop_mask_by_box(mask_input, bbox_input)
            mask_orig = project_letterbox_mask_to_original(mask_input, meta)
            binary_mask = mask_orig >= float(self.config.mask_threshold)
            segments, mask_area = mask_to_segments(binary_mask)

            x1, y1, x2, y2 = [float(x) for x in bbox_orig.tolist()]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            predictions.append(
                {
                    "class_id": cls_id,
                    "class_name": safe_class_name(self.config.class_names, cls_id),
                    "confidence": round(score, 6),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "center": [round(cx, 2), round(cy, 2)],
                    "center_x": round(cx, 2),
                    "center_y": round(cy, 2),
                    "mask": {
                        "threshold": float(self.config.mask_threshold),
                        "area": round(float(mask_area), 2),
                        "shape": [int(orig_h), int(orig_w)],
                        "segments": segments,
                        "polygon": segments[0] if segments else [],
                    },
                }
            )

        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return {
            "predictions": predictions,
            "task": "segmentation",
            "num_classes": num_classes,
            "class_names": self.config.class_names,
            "output_shapes": shape_info,
            "decode_format": decode_format,
            "mask": {
                "mask_dim": mask_dim,
                "proto_shape": [int(x) for x in proto.shape],
                "threshold": float(self.config.mask_threshold),
                "return_format": "polygon_segments",
            },
            "nms": {
                "type": "horizontal_bbox_nms",
                "iou_threshold": self.config.nms_threshold,
            },
            "meta": meta,
        }

    def _postprocess_obb(
        self,
        outputs: Optional[List[np.ndarray]],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        OBB 后处理。

        同时兼容两种输出：
        1. 普通 Ultralytics OBB 单输出：
           [1, 4 + num_classes + 1, 8400]
           例如 4 类时 [1, 9, 8400]

        2. airockchip/ultralytics_yolov8 format=rknn 专用多输出：
           [1, 64 + num_classes, 80, 80]
           [1, 64 + num_classes, 40, 40]
           [1, 64 + num_classes, 20, 20]
           [1, 1, 8400]

        当前版本仍使用水平外接框 NMS，返回 bbox + obb.points。
        """
        if outputs is None or len(outputs) == 0:
            return {"predictions": [], "task": "obb_detection", "message": "empty outputs"}

        shape_info = [tuple(np.asarray(o).shape) for o in outputs]
        num_classes = int(self.config.num_classes)
        expected_channels = 4 + num_classes + 1

        if is_rockchip_obb_outputs(outputs, num_classes=num_classes):
            if not self._postprocess_format_logged:
                logger.info(
                    f"[POSTPROCESS] 使用 Rockchip YOLOv8 OBB 多输出后处理: shapes={shape_info}"
                )
                self._postprocess_format_logged = True

            boxes_xywhr, class_scores, angles = decode_rockchip_obb_outputs(
                outputs=outputs,
                num_classes=num_classes,
                input_size=tuple(meta["input_shape"]),
            )
            boxes_xywh = boxes_xywhr[:, :4].astype(np.float32)
            decode_format = "rockchip_yolov8_obb_multi_output"
        else:
            out = outputs[0]
            if not self._postprocess_format_logged:
                logger.info(
                    f"[POSTPROCESS] 使用 OBB 单输出后处理: shapes={shape_info}, "
                    f"expected_channels={expected_channels}"
                )
                self._postprocess_format_logged = True

            preds = normalize_yolo_single_output(out, expected_channels=expected_channels)
            if preds.shape[1] > expected_channels:
                preds = preds[:, :expected_channels]

            boxes_xywh = preds[:, 0:4].astype(np.float32)
            class_scores = preds[:, 4:4 + num_classes].astype(np.float32)
            angles = preds[:, 4 + num_classes].astype(np.float32)
            decode_format = "ultralytics_obb_single_output"

            # 普通单输出中 class_scores 通常已是 sigmoid 后概率；如果像 logits，则转 sigmoid。
            if class_scores.size > 0 and (class_scores.max() > 1.0 or class_scores.min() < 0.0):
                class_scores = sigmoid(class_scores)

        if class_scores.size == 0:
            return {
                "predictions": [],
                "task": "obb_detection",
                "output_shapes": shape_info,
                "decode_format": decode_format,
                "message": "empty class scores",
            }

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]

        if os.getenv("OBB_DEBUG", "0") == "1":
            logger.info(
                "[OBB DEBUG] raw decoded stats: "
                f"boxes_min={float(np.min(boxes_xywh)):.6f}, "
                f"boxes_max={float(np.max(boxes_xywh)):.6f}, "
                f"cls_min={float(np.min(class_scores)):.6f}, "
                f"cls_max={float(np.max(class_scores)):.6f}, "
                f"angle_min={float(np.min(angles)):.6f}, "
                f"angle_max={float(np.max(angles)):.6f}, "
                f"threshold={self.config.conf_threshold}, "
                f"decode_format={decode_format}"
            )
            top_idx = scores.argsort()[::-1][:10]
            logger.info(
                "[OBB DEBUG] top10 scores: "
                + ", ".join(
                    [
                        f"(score={float(scores[i]):.6f}, cls={int(class_ids[i])})"
                        for i in top_idx
                    ]
                )
            )

        keep_mask = scores >= self.config.conf_threshold
        if not np.any(keep_mask):
            return {
                "predictions": [],
                "task": "obb_detection",
                "output_shapes": shape_info,
                "decode_format": decode_format,
                "message": "no predictions above confidence threshold",
                "debug": {
                    "score_max": round(float(np.max(scores)), 6) if scores.size else None,
                    "score_min": round(float(np.min(scores)), 6) if scores.size else None,
                    "cls_score_max": round(float(np.max(class_scores)), 6) if class_scores.size else None,
                    "cls_score_min": round(float(np.min(class_scores)), 6) if class_scores.size else None,
                    "conf_threshold": self.config.conf_threshold,
                },
            }

        boxes_xywh = boxes_xywh[keep_mask]
        angles = angles[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        ratio = float(meta["ratio"])
        dw, dh = meta["pad"]
        orig_h, orig_w = meta["orig_shape"]
        input_h, input_w = meta["input_shape"]

        # 单输出模型有时可能输出 0~1 归一化坐标；Rockchip 多输出解码后已是 input pixels。
        if boxes_xywh.size > 0 and float(np.nanmax(boxes_xywh[:, 0:4])) <= 2.0:
            boxes_xywh[:, [0, 2]] *= float(input_w)
            boxes_xywh[:, [1, 3]] *= float(input_h)

        obb_points_list: List[np.ndarray] = []
        bbox_xyxy_list: List[np.ndarray] = []
        valid_indices: List[int] = []

        for i, (box, angle) in enumerate(zip(boxes_xywh, angles)):
            cx, cy, bw, bh = [float(x) for x in box.tolist()]

            # 映射回原图坐标。
            cx = (cx - float(dw)) / max(ratio, 1e-6)
            cy = (cy - float(dh)) / max(ratio, 1e-6)
            bw = bw / max(ratio, 1e-6)
            bh = bh / max(ratio, 1e-6)

            if bw <= 2.0 or bh <= 2.0:
                continue

            points = xywhr_to_points(cx, cy, bw, bh, float(angle))
            points = clip_points(points, orig_w, orig_h)
            bbox = points_to_xyxy(points, orig_w, orig_h)

            if bbox[2] - bbox[0] <= 2.0 or bbox[3] - bbox[1] <= 2.0:
                continue

            obb_points_list.append(points)
            bbox_xyxy_list.append(bbox)
            valid_indices.append(i)

        if not bbox_xyxy_list:
            return {
                "predictions": [],
                "task": "obb_detection",
                "output_shapes": shape_info,
                "decode_format": decode_format,
                "message": "no valid obb boxes",
            }

        bboxes = np.stack(bbox_xyxy_list, axis=0)
        valid_scores = scores[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        valid_angles = angles[valid_indices]

        keep = multiclass_nms(
            boxes=bboxes,
            scores=valid_scores,
            class_ids=valid_class_ids,
            iou_thresh=self.config.nms_threshold,
        )

        predictions: List[Dict[str, Any]] = []
        for idx in keep:
            cls_id = int(valid_class_ids[idx])
            score = float(valid_scores[idx])
            bbox = bboxes[idx]
            points = obb_points_list[idx]
            angle = float(valid_angles[idx])
            cx = float(np.mean(points[:, 0]))
            cy = float(np.mean(points[:, 1]))
            edge_w = float(np.linalg.norm(points[1] - points[0]))
            edge_h = float(np.linalg.norm(points[2] - points[1]))

            predictions.append(
                {
                    "class_id": cls_id,
                    "class_name": safe_class_name(self.config.class_names, cls_id),
                    "confidence": round(score, 6),
                    "bbox": [round(float(x), 2) for x in bbox.tolist()],
                    "center": [round(cx, 2), round(cy, 2)],
                    "center_x": round(cx, 2),
                    "center_y": round(cy, 2),
                    "obb": {
                        "cx": round(cx, 2),
                        "cy": round(cy, 2),
                        "w": round(edge_w, 2),
                        "h": round(edge_h, 2),
                        "angle": round(angle_to_radians(angle), 6),
                        "angle_unit": "radian",
                        "points": [
                            [round(float(x), 2), round(float(y), 2)]
                            for x, y in points.tolist()
                        ],
                    },
                }
            )

        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return {
            "predictions": predictions,
            "task": "obb_detection",
            "num_classes": num_classes,
            "class_names": self.config.class_names,
            "output_shapes": shape_info,
            "decode_format": decode_format,
            "nms": {
                "type": "horizontal_bbox_nms",
                "iou_threshold": self.config.nms_threshold,
                "note": "当前 OBB 初版使用水平外接框 NMS，后续可升级 rotated NMS",
            },
            "meta": meta,
        }

    def release(self):
        if self.rknn and not self._simulate_mode:
            self.rknn.release()
        self.rknn = None
        self.is_loaded = False
        logger.info("RKNN资源已释放")


# ────────────────────────────────────────────────
# FastAPI
# ────────────────────────────────────────────────
def create_app(config: Optional[InferenceConfig] = None):
    try:
        from fastapi import FastAPI, HTTPException, UploadFile, File
        from fastapi.responses import PlainTextResponse, JSONResponse
    except ImportError as e:
        raise ImportError("请安装 fastapi 与 python-multipart") from e

    if config is None:
        config = InferenceConfig()

    engine = RKNNInferenceEngine(config)

    @asynccontextmanager
    async def lifespan(app: Any):
        logger.info("启动推理服务...")
        success = engine.load_model()
        if not success:
            logger.error("模型加载失败！")
        yield
        engine.release()
        logger.info("推理服务已关闭")

    app = FastAPI(
        title="VisionOps Edge Inference",
        description="RK3588 Detection / Classification / OBB / Segmentation Inference Service",
        version="2.2.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok" if engine and engine.is_loaded else "error",
            "model_path": config.model_path,
            "platform": config.target_platform,
            "task": config.normalized_task(),
            "input_size": config.input_size,
            "num_classes": config.num_classes,
            "class_names": config.class_names,
            "class_names_file": config.class_names_file,
            "conf_threshold": config.conf_threshold if config.normalized_task() in {"detection", "obb_detection", "segmentation"} else None,
            "nms_threshold": config.nms_threshold if config.normalized_task() in {"detection", "obb_detection", "segmentation"} else None,
            "mask_threshold": config.mask_threshold if config.normalized_task() == "segmentation" else None,
            "topk": config.topk if config.normalized_task() == "classification" else None,
            "simulate_mode": engine._simulate_mode,
        }

    @app.post("/infer")
    async def infer_endpoint(file: UploadFile = File(...)):
        import cv2

        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码图像")
        result = engine.infer(image)
        return JSONResponse(result)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return engine.metrics.prometheus_format()

    @app.get("/stats")
    async def stats():
        return engine.metrics.get_stats()

    @app.post("/reload")
    async def reload_model(model_path: Optional[str] = None):
        if model_path:
            config.model_path = model_path
        engine.release()
        success = engine.load_model()
        return {"success": success, "model_path": config.model_path}

    return app


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="VisionOps RK3588 Detection / Classification / OBB / Segmentation Inference Service")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", "/opt/visionops/models/current.rknn"))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    parser.add_argument(
        "--task",
        default=os.getenv("TASK", "detection"),
        choices=["detection", "classification", "obb_detection", "obb", "segmentation", "seg", "segment"],
        help="推理任务类型：detection、classification、obb_detection 或 segmentation",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="模型输入尺寸，例如 detection: --input-size 640 640，classification: --input-size 224 224",
    )
    parser.add_argument(
        "--npu-core",
        default=os.getenv("NPU_CORE", "auto"),
        choices=["auto", "core_0", "core_1", "core_2", "core_0_1", "core_0_1_2"],
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=int(os.getenv("NUM_CLASSES", "6")),
    )
    parser.add_argument(
        "--class-names-file",
        type=str,
        default=os.getenv("CLASS_NAMES_FILE"),
        help="类别配置文件路径，例如 /opt/visionops/edge/runtime/class_names.yaml",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.getenv("METRICS_PORT", "9091")),
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=float(os.getenv("CONF_THRESHOLD", "0.25")),
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=float(os.getenv("NMS_THRESHOLD", "0.45")),
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=float(os.getenv("MASK_THRESHOLD", "0.5")),
        help="segmentation mask 二值化阈值",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=int(os.getenv("TOPK", "5")),
        help="classification top-k 输出数量",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=int(os.getenv("WARMUP_RUNS", "3")),
    )
    parser.add_argument(
        "--debug-shapes",
        action="store_true",
        default=os.getenv("DEBUG_SHAPES", "1") == "1",
        help="是否打印 RKNN 输出 shape。默认由 DEBUG_SHAPES 控制，默认开启。",
    )

    args = parser.parse_args()

    # 先根据任务类型设置默认输入尺寸。
    # 命令行 --input-size 优先级最高，其次是 INPUT_SIZE 环境变量，最后是任务默认值。
    task_default_input_size = [224, 224] if args.task == "classification" else [640, 640]
    input_size = args.input_size or parse_input_size(os.getenv("INPUT_SIZE"), task_default_input_size)

    if args.task == "classification":
        default_names = [str(i) for i in range(args.num_classes)]
    else:
        default_names = ["2wheelers", "auto", "bus", "car", "pedestrian", "truck"]

    if args.task == "detection" and args.num_classes == 6:
        class_names = default_names
    else:
        class_names = [str(i) for i in range(args.num_classes)]

    config = InferenceConfig(
        model_path=args.model,
        task=args.task,
        input_size=input_size,
        npu_core=args.npu_core,
        num_classes=args.num_classes,
        class_names=class_names,
        class_names_file=args.class_names_file,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        mask_threshold=args.mask_threshold,
        topk=args.topk,
        metrics_port=args.metrics_port,
        warmup_runs=args.warmup_runs,
        debug_shapes=args.debug_shapes,
    )

    file_class_names, file_num_classes, file_task, file_input_size = load_class_names_config(config.class_names_file)

    # 类别文件中的 task 只在命令行没有显式传 TASK 时作为提示使用；
    # 为避免误切任务，这里不自动覆盖 --task，只记录日志。
    if file_task and file_task != config.task:
        logger.info(f"类别配置文件 task={file_task}，当前启动 task={config.task}，以启动参数为准")

    if file_input_size and args.input_size is None and os.getenv("INPUT_SIZE") is None:
        config.input_size = file_input_size
        logger.info(f"已从类别配置文件加载 input_size={config.input_size}")

    if file_class_names is not None:
        config.class_names = file_class_names
        config.num_classes = int(file_num_classes)
        logger.info(
            f"已从类别配置文件加载类别信息: "
            f"num_classes={config.num_classes}, class_names={config.class_names}"
        )
    else:
        logger.info(
            f"未使用外部类别配置文件，继续使用当前配置: "
            f"num_classes={config.num_classes}, class_names={config.class_names}"
        )

    logger.info(
        f"启动配置: task={config.task}, model={config.model_path}, "
        f"input_size={config.input_size}, npu_core={config.npu_core}, port={args.port}"
    )

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
