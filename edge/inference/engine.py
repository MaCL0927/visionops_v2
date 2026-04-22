"""
RK3588 边缘推理引擎（Detection 版骨架）
- 基于 rknnlite2 运行 RKNN 模型
- FastAPI 推理服务
- 面向 YOLO Detection 模型的后处理
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import deque

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("visionops.inference")


# ────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────
@dataclass
class InferenceConfig:
    model_path: str = "/opt/visionops/models/current.rknn"
    target_platform: str = "rk3588"
    npu_core: str = "auto"  # auto / core_0 / core_0_1 / core_0_1_2

    # 你的 detection 主线当前默认 640x640
    input_size: List[int] = field(default_factory=lambda: [640, 640])

    # detection 配置
    class_names: List[str] = field(default_factory=lambda: ["person", "smoke"])
    num_classes: int = 2
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45

    metrics_port: int = 9091
    warmup_runs: int = 3

    # 调试：是否打印 output shape
    debug_shapes: bool = True


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
            if not self.latencies:
                return {"total_inferences": self.total_inferences, "errors": self.errors}

            lats = list(self.latencies)
            return {
                "total_inferences": self.total_inferences,
                "errors": self.errors,
                "latency_ms": {
                    "mean": round(float(np.mean(lats)), 2),
                    "p50": round(float(np.percentile(lats, 50)), 2),
                    "p95": round(float(np.percentile(lats, 95)), 2),
                    "p99": round(float(np.percentile(lats, 99)), 2),
                    "min": round(float(np.min(lats)), 2),
                    "max": round(float(np.max(lats)), 2),
                },
                "throughput_fps": round(float(1000.0 / np.mean(lats)), 2) if np.mean(lats) > 0 else 0.0,
            }

    def prometheus_format(self) -> str:
        stats = self.get_stats()
        lines = [
            "# HELP visionops_inference_total Total inference requests",
            "# TYPE visionops_inference_total counter",
            f'visionops_inference_total {stats["total_inferences"]}',
            "# HELP visionops_inference_errors Total inference errors",
            "# TYPE visionops_inference_errors counter",
            f'visionops_inference_errors {stats["errors"]}',
        ]
        if "latency_ms" in stats:
            lm = stats["latency_ms"]
            lines.append(f'visionops_inference_latency_mean {lm["mean"]}')
            lines.append(f'visionops_throughput_fps {stats["throughput_fps"]}')
        return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clip_box_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    box[0] = np.clip(box[0], 0, w - 1)
    box[1] = np.clip(box[1], 0, h - 1)
    box[2] = np.clip(box[2], 0, w - 1)
    box[3] = np.clip(box[3], 0, h - 1)
    return box


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    boxes: [..., 4] in [cx, cy, w, h]
    """
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
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
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
    iou_thresh: float
) -> List[int]:
    keep_all = []
    for cls in np.unique(class_ids):
        inds = np.where(class_ids == cls)[0]
        keep = nms_xyxy(boxes[inds], scores[inds], iou_thresh)
        keep_all.extend(list(inds[keep]))
    return keep_all


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
        logger.info("✓ RKNN模型加载成功")
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
        new_shape: Tuple[int, int]
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
        import cv2

        orig_h, orig_w = image.shape[:2]
        h, w = self.config.input_size

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # letterbox + BGR -> RGB
        img, ratio, (dw, dh) = self._letterbox(image, (h, w))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        meta = {
            "orig_shape": (orig_h, orig_w),
            "input_shape": (h, w),
            "ratio": ratio,
            "pad": (dw, dh),
        }
        return img, meta

    def _run_inference_raw(self, image_hwc: np.ndarray) -> Optional[List[np.ndarray]]:
        if self._simulate_mode:
            # 模拟一个检测输出：[1, N, 5 + C]
            n = 20
            c = self.config.num_classes
            fake = np.random.rand(1, n, 5 + c).astype(np.float32)
            return [fake]

        # RKNN 常见需要 4D 输入，这里显式补 batch 维
        inp = np.expand_dims(image_hwc, axis=0)  # [1, H, W, 3]

        with self._lock:
            outputs = self.rknn.inference(inputs=[inp])

        if self.config.debug_shapes and not self._shape_logged and outputs is not None:
            for i, out in enumerate(outputs):
                logger.info(f"[DEBUG] output[{i}] shape={getattr(out, 'shape', None)}, dtype={getattr(out, 'dtype', None)}")
            self._shape_logged = True

        return outputs

    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        t0 = time.perf_counter()
        try:
            preprocessed, meta = self._preprocess(image)
            outputs = self._run_inference_raw(preprocessed)

            result = self._postprocess_detection(outputs, meta)

            latency_ms = (time.perf_counter() - t0) * 1000
            self.metrics.record(latency_ms, success=True)
            result["latency_ms"] = round(latency_ms, 2)
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            self.metrics.record(latency_ms, success=False)
            logger.exception(f"推理异常: {e}")
            raise

    def _postprocess_detection(
        self,
        outputs: Optional[List[np.ndarray]],
        meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detection 后处理入口。

        当前专门适配：
        output[0].shape == (1, 6, 8400)

        解释方式：
        - 先取 outputs[0][0] => (6, 8400)
        - transpose => (8400, 6)
        - 前4维: bbox
        - 后2维: class logits / scores
        """
        if outputs is None or len(outputs) == 0:
            return {
                "predictions": [],
                "task": "detection",
                "message": "empty outputs",
            }

        out = outputs[0]

        # 你当前确认的 RKNN 输出格式：(1, 6, 8400)
        if out.ndim == 3 and out.shape[0] == 1 and out.shape[1] == (4 + self.config.num_classes):
            pred = out[0].transpose(1, 0)   # (8400, 6)
            return self._decode_rknn_1x6x8400(pred, meta)

        # 兼容另一种可能：(1, 8400, 6)
        if out.ndim == 3 and out.shape[0] == 1 and out.shape[2] == (4 + self.config.num_classes):
            pred = out[0]   # (8400, 6)
            return self._decode_rknn_1x6x8400(pred, meta)

        # 兼容旧的 flat 形式
        if out.ndim == 2 and out.shape[1] == (4 + self.config.num_classes):
            return self._decode_rknn_1x6x8400(out, meta)

        shape_info = [tuple(o.shape) for o in outputs]
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
        meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        专门适配 RKNN 输出:
            pred.shape == (8400, 6)

        假设格式:
            [x, y, w, h, cls0, cls1]

        注意：
        1. 这里不再使用 objectness
        2. 分数直接取 max(class_scores)
        3. 框默认按 xywh 解释
        """

        if pred is None or pred.size == 0:
            return {"predictions": [], "task": "detection"}

        num_classes = self.config.num_classes
        if pred.shape[1] != 4 + num_classes:
            logger.warning(f"输出维度不符合预期，pred.shape={pred.shape}")
            return {"predictions": [], "task": "detection"}

        boxes = pred[:, :4].astype(np.float32)              # (8400, 4)
        cls_scores = pred[:, 4:4 + num_classes].astype(np.float32)   # (8400, 2)

        # 如果类别分数看起来不是概率，则做 sigmoid
        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = sigmoid(cls_scores)

        class_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(cls_scores)), class_ids]

        # 阈值过滤
        keep_mask = scores >= self.config.conf_threshold
        if not np.any(keep_mask):
            return {"predictions": [], "task": "detection"}

        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        # --------------------------------------------------
        # 框格式处理：
        # 当前先按 xywh -> xyxy 处理
        # 如果结果仍明显异常，再切换成“已经是 xyxy”试验
        # --------------------------------------------------
        boxes_xyxy = xywh_to_xyxy(boxes)

        # --------------------------------------------------
        # 将输入尺度坐标映射回原图
        # --------------------------------------------------
        ratio = meta["ratio"]
        dw, dh = meta["pad"]
        orig_h, orig_w = meta["orig_shape"]

        boxes_xyxy[:, [0, 2]] -= dw
        boxes_xyxy[:, [1, 3]] -= dh
        boxes_xyxy /= max(ratio, 1e-6)

        for i in range(len(boxes_xyxy)):
            boxes_xyxy[i] = clip_box_xyxy(boxes_xyxy[i], orig_w, orig_h)

        # 去掉明显无效框
        wh = boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2]
        valid_mask = (wh[:, 0] > 2.0) & (wh[:, 1] > 2.0)
        if not np.any(valid_mask):
            return {"predictions": [], "task": "detection"}

        boxes_xyxy = boxes_xyxy[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]

        # NMS
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
            predictions.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(float(scores[i]), 4),
                "bbox": [round(float(x), 2) for x in boxes_xyxy[i].tolist()],
            })

        predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "predictions": predictions,
            "task": "detection",
        }

    def _decode_flat_predictions(
        self,
        pred: np.ndarray,
        meta: Dict[str, Any]
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

        # 兼容两种情况：
        # - 若输出还没过 sigmoid，这里可加 sigmoid
        # - 若输出已经是概率，sigmoid 影响会有偏差
        # 这里先保守采用“仅当值域明显超出[0,1]时再 sigmoid”
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

        # 将 letterbox 坐标映射回原图
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
            predictions.append({
                "class_id": cls_id,
                "class_name": self.config.class_names[cls_id] if cls_id < len(self.config.class_names) else str(cls_id),
                "confidence": round(float(scores[i]), 4),
                "bbox": [round(float(x), 2) for x in boxes_xyxy[i].tolist()],
            })

        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return {
            "predictions": predictions,
            "task": "detection",
        }

    def release(self):
        if self.rknn and not self._simulate_mode:
            self.rknn.release()
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
        description="RK3588 YOLO Detection Inference Service",
        version="2.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok" if engine.is_loaded else "model_not_loaded",
            "model_path": config.model_path,
            "platform": config.target_platform,
            "task": "detection",
            "class_names": config.class_names,
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

    parser = argparse.ArgumentParser(description="VisionOps RK3588 Detection Inference Service")
    parser.add_argument("--model", default="/opt/visionops/models/current.rknn")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--npu-core", default="auto",
                        choices=["auto", "core_0", "core_1", "core_2", "core_0_1", "core_0_1_2"])
    parser.add_argument("--num-classes", type=int, default=6)
    args = parser.parse_args()

    default_names = ["2wheelers", "auto", "bus", "car", "pedestrian", "truck"]
    if args.num_classes == 6:
        class_names = default_names
    else:
        class_names = [str(i) for i in range(args.num_classes)]

    cfg = InferenceConfig(
        model_path=args.model,
        npu_core=args.npu_core,
        num_classes=args.num_classes,
        class_names=class_names,
    )

    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
