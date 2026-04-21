"""
RK3588 边缘推理引擎
- 基于 rknnlite2 运行 RKNN 模型
- 支持三NPU核心负载均衡
- 内置性能指标上报（Prometheus格式）
- FastAPI推理服务接口
"""
import os
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import deque
import threading

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("visionops.inference")


# ────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────
@dataclass
class InferenceConfig:
    model_path: str = "/opt/visionops/models/current.rknn"
    target_platform: str = "rk3588"
    npu_core: str = "auto"           # auto / core_0 / core_0_1 / core_0_1_2
    input_size: List[int] = field(default_factory=lambda: [640, 640])
    num_classes: int = 10
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    metrics_port: int = 9091         # Prometheus metrics端口
    max_batch_queue: int = 32
    warmup_runs: int = 3             # 预热推理次数


# ────────────────────────────────────────────────
# 性能指标收集器
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
                return {"total": self.total_inferences, "errors": self.errors}
            lats = list(self.latencies)
            return {
                "total_inferences": self.total_inferences,
                "errors": self.errors,
                "latency_ms": {
                    "mean": round(np.mean(lats), 2),
                    "p50": round(np.percentile(lats, 50), 2),
                    "p95": round(np.percentile(lats, 95), 2),
                    "p99": round(np.percentile(lats, 99), 2),
                    "min": round(min(lats), 2),
                    "max": round(max(lats), 2),
                },
                "throughput_fps": round(1000.0 / np.mean(lats), 1) if np.mean(lats) > 0 else 0,
            }

    def prometheus_format(self) -> str:
        stats = self.get_stats()
        lines = [
            f"# HELP visionops_inference_total Total inference requests",
            f"# TYPE visionops_inference_total counter",
            f'visionops_inference_total {stats["total_inferences"]}',
            f"# HELP visionops_inference_errors Total inference errors",
            f"# TYPE visionops_inference_errors counter",
            f'visionops_inference_errors {stats["errors"]}',
        ]
        if "latency_ms" in stats:
            lm = stats["latency_ms"]
            for pct, key in [("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")]:
                lines.append(
                    f'visionops_inference_latency_ms{{quantile="{pct}"}} {lm[key]}'
                )
            lines.append(f'visionops_inference_latency_mean {lm["mean"]}')
            lines.append(f'visionops_throughput_fps {stats["throughput_fps"]}')
        return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────
# RKNN推理引擎
# ────────────────────────────────────────────────
class RKNNInferenceEngine:
    NPU_CORE_MAP = {
        "auto":       0b000,  # RKNN_NPU_CORE_AUTO
        "core_0":     0b001,  # RKNN_NPU_CORE_0
        "core_1":     0b010,  # RKNN_NPU_CORE_1
        "core_2":     0b100,  # RKNN_NPU_CORE_2
        "core_0_1":   0b011,  # RKNN_NPU_CORE_0_1
        "core_0_1_2": 0b111,  # RKNN_NPU_CORE_0_1_2
    }

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.rknn = None
        self.is_loaded = False
        self.metrics = MetricsCollector()
        self._lock = threading.Lock()

    def load_model(self) -> bool:
        """加载RKNN模型到NPU"""
        model_path = self.config.model_path
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False

        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            logger.warning("rknnlite2 未安装，使用模拟模式")
            self.is_loaded = True
            self._simulate_mode = True
            return True

        self._simulate_mode = False
        self.rknn = RKNNLite()

        logger.info(f"加载RKNN模型: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            logger.error(f"加载模型失败: {ret}")
            return False

        # 初始化NPU运行时
        npu_core = self.NPU_CORE_MAP.get(self.config.npu_core.lower(), 0)
        logger.info(f"初始化NPU，核心配置: {self.config.npu_core}")
        ret = self.rknn.init_runtime(core_mask=npu_core)
        if ret != 0:
            logger.error(f"初始化NPU运行时失败: {ret}")
            return False

        self.is_loaded = True
        logger.info("✓ RKNN模型加载成功")

        # 预热
        self._warmup()
        return True

    def _warmup(self):
        """预热推理，稳定NPU性能"""
        logger.info(f"预热推理 ({self.config.warmup_runs} 次)...")
        dummy = np.random.randint(0, 255,
            (self.config.input_size[0], self.config.input_size[1], 3), dtype=np.uint8)
        for _ in range(self.config.warmup_runs):
            self._run_inference_raw(dummy)
        logger.info("预热完成")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        import cv2
        h, w = self.config.input_size
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _run_inference_raw(self, image: np.ndarray) -> Optional[List]:
        """原始推理调用"""
        if getattr(self, '_simulate_mode', False):
            time.sleep(0.01)  # 模拟10ms推理
            return [np.random.rand(1, self.config.num_classes)]

        with self._lock:
            outputs = self.rknn.inference(inputs=[image])
        return outputs

    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        """
        执行推理，返回结果字典
        image: BGR或RGB numpy数组
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        t0 = time.perf_counter()
        try:
            preprocessed = self._preprocess(image)
            outputs = self._run_inference_raw(preprocessed)
            latency_ms = (time.perf_counter() - t0) * 1000

            self.metrics.record(latency_ms, success=True)

            result = self._postprocess(outputs, image.shape[:2])
            result["latency_ms"] = round(latency_ms, 2)
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            self.metrics.record(latency_ms, success=False)
            logger.error(f"推理异常: {e}")
            raise

    def _postprocess(self, outputs: List, original_shape: tuple) -> Dict[str, Any]:
        """
        后处理（根据任务类型扩展）
        默认：分类任务
        """
        if outputs is None or len(outputs) == 0:
            return {"predictions": [], "task": "unknown"}

        logits = outputs[0]  # shape: [1, num_classes]
        if logits.ndim > 1:
            logits = logits[0]

        probs = self._softmax(logits)
        top_k = min(5, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = [
            {"class_id": int(idx), "confidence": float(probs[idx])}
            for idx in top_indices
            if probs[idx] >= self.config.conf_threshold
        ]

        return {
            "predictions": predictions,
            "task": "classification",
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    def release(self):
        if self.rknn and not getattr(self, '_simulate_mode', False):
            self.rknn.release()
            self.is_loaded = False
            logger.info("RKNN资源已释放")


# ────────────────────────────────────────────────
# FastAPI服务
# ────────────────────────────────────────────────
def create_app(config: InferenceConfig = None):
    try:
        from fastapi import FastAPI, HTTPException, UploadFile, File
        from fastapi.responses import PlainTextResponse, JSONResponse
        import uvicorn
    except ImportError:
        raise ImportError("请安装 fastapi uvicorn: pip install fastapi uvicorn[standard]")

    if config is None:
        config = InferenceConfig()

    engine = RKNNInferenceEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("启动推理服务...")
        success = engine.load_model()
        if not success:
            logger.error("模型加载失败！")
        yield
        engine.release()
        logger.info("推理服务已关闭")

    app = FastAPI(
        title="VisionOps Edge Inference",
        description="RK3588 NPU推理服务",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok" if engine.is_loaded else "model_not_loaded",
            "model_path": config.model_path,
            "platform": config.target_platform,
        }

    @app.post("/infer")
    async def infer_endpoint(file: UploadFile = File(...)):
        import cv2
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "无法解码图像")

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
        """热重载模型（部署新版本时调用）"""
        if model_path:
            config.model_path = model_path
        engine.release()
        success = engine.load_model()
        return {"success": success, "model_path": config.model_path}

    return app


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="VisionOps RK3588推理服务")
    parser.add_argument("--model", default="/opt/visionops/models/current.rknn")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--npu-core", default="auto",
                        choices=["auto", "core_0", "core_1", "core_2", "core_0_1", "core_0_1_2"])
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    cfg = InferenceConfig(
        model_path=args.model,
        npu_core=args.npu_core,
        num_classes=args.num_classes,
    )
    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
