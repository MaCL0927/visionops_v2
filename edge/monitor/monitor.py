"""
边缘端性能监控模块
- 定期采集推理性能指标
- 检测精度漂移（对比ground truth或置信度分布）
- 回传指标到服务器（触发再训练告警）
"""
import os
import time
import json
import logging
import requests
import statistics
from datetime import datetime, timedelta
from collections import deque
from typing import Optional

logger = logging.getLogger("visionops.monitor")


class DriftDetector:
    """
    数据漂移检测
    通过监控置信度分布变化来检测数据分布偏移
    """
    def __init__(self, window_size: int = 500, drift_threshold: float = 0.1):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.baseline_confidences: Optional[list] = None
        self.current_confidences = deque(maxlen=window_size)

    def set_baseline(self, confidences: list):
        """设置基准置信度分布"""
        self.baseline_confidences = confidences
        logger.info(f"漂移检测基准已设置，样本数: {len(confidences)}")

    def add_observation(self, confidence: float):
        """添加新观测"""
        self.current_confidences.append(confidence)

    def check_drift(self) -> dict:
        """检查是否发生数据漂移（PSI指标）"""
        if self.baseline_confidences is None or len(self.current_confidences) < 100:
            return {"drift_detected": False, "reason": "数据不足"}

        baseline_mean = statistics.mean(self.baseline_confidences)
        current_mean = statistics.mean(self.current_confidences)
        baseline_std = statistics.stdev(self.baseline_confidences)
        current_std = statistics.stdev(self.current_confidences)

        # 简化版：均值偏移检测
        mean_shift = abs(current_mean - baseline_mean)
        relative_shift = mean_shift / (baseline_mean + 1e-8)

        drift_detected = relative_shift > self.drift_threshold

        return {
            "drift_detected": drift_detected,
            "baseline_mean": round(baseline_mean, 4),
            "current_mean": round(current_mean, 4),
            "mean_shift": round(mean_shift, 4),
            "relative_shift": round(relative_shift, 4),
            "threshold": self.drift_threshold,
            "sample_size": len(self.current_confidences),
        }


class EdgeMonitor:
    """边缘设备监控主类"""

    def __init__(self,
                 device_id: str,
                 inference_service_url: str = "http://localhost:8080",
                 server_url: str = "http://192.168.1.1:8000",
                 report_interval: int = 60):  # 每60秒上报一次
        self.device_id = device_id
        self.inference_url = inference_service_url
        self.server_url = server_url
        self.report_interval = report_interval
        self.drift_detector = DriftDetector()
        self._running = False

    def collect_metrics(self) -> dict:
        """从推理服务采集指标"""
        try:
            resp = requests.get(f"{self.inference_url}/stats", timeout=5)
            stats = resp.json()
        except Exception as e:
            logger.warning(f"采集推理指标失败: {e}")
            stats = {}

        # 系统指标
        system_metrics = self._collect_system_metrics()

        return {
            "device_id": self.device_id,
            "timestamp": datetime.now().isoformat(),
            "inference": stats,
            "system": system_metrics,
        }

    def _collect_system_metrics(self) -> dict:
        """采集系统资源使用率"""
        metrics = {}
        try:
            import psutil
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            metrics["memory_percent"] = psutil.virtual_memory().percent
            metrics["disk_percent"] = psutil.disk_usage("/").percent

            # RK3588温度（通过sysfs）
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    temp_milli = int(f.read().strip())
                    metrics["cpu_temp_c"] = round(temp_milli / 1000.0, 1)
            except Exception:
                pass

        except ImportError:
            pass

        return metrics

    def report_to_server(self, metrics: dict) -> bool:
        """上报指标到服务器"""
        try:
            resp = requests.post(
                f"{self.server_url}/api/v1/edge/metrics",
                json=metrics,
                timeout=10,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"上报失败: {e}")
            return False

    def check_and_alert_drift(self):
        """检查数据漂移并告警"""
        drift_result = self.drift_detector.check_drift()
        if drift_result.get("drift_detected"):
            logger.warning(f"检测到数据漂移! {json.dumps(drift_result, ensure_ascii=False)}")
            try:
                requests.post(
                    f"{self.server_url}/api/v1/alerts/drift",
                    json={
                        "device_id": self.device_id,
                        "drift_info": drift_result,
                        "timestamp": datetime.now().isoformat(),
                    },
                    timeout=5,
                )
            except Exception:
                pass

    def run(self):
        """主监控循环"""
        self._running = True
        logger.info(f"边缘监控启动 | 设备: {self.device_id} | 上报间隔: {self.report_interval}s")

        while self._running:
            try:
                metrics = self.collect_metrics()
                self.report_to_server(metrics)
                self.check_and_alert_drift()

                # 本地日志
                lat = metrics.get("inference", {}).get("latency_ms", {})
                if lat:
                    logger.info(f"推理延迟 P50={lat.get('p50')}ms P95={lat.get('p95')}ms | "
                                f"FPS={metrics['inference'].get('throughput_fps', 'N/A')}")

            except Exception as e:
                logger.error(f"监控循环异常: {e}")

            time.sleep(self.report_interval)

    def stop(self):
        self._running = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", "rk3588-001"))
    parser.add_argument("--server-url", default=os.environ.get("SERVER_URL", "http://192.168.1.1:8000"))
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    monitor = EdgeMonitor(
        device_id=args.device_id,
        server_url=args.server_url,
        report_interval=args.interval,
    )
    monitor.run()
