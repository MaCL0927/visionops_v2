"""
MLOps 自动再训练调度器
- 定期检查：精度漂移 / 新数据积累 / 定时触发
- 满足条件时调用 DVC pipeline 执行完整重训练
- 通过 MLflow API 获取当前 Production 模型指标
- 支持 Slack / 邮件告警通知
"""
import os
import json
import time
import logging
import subprocess
import smtplib
import threading
from datetime import datetime
from pathlib import Path
from email.mime.text import MIMEText
from typing import Optional

import yaml
import requests

logger = logging.getLogger("visionops.retrain_scheduler")


# ──────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────
def load_config(path: str = "pipeline/configs/mlops.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────
# MLflow 指标查询
# ──────────────────────────────────────────────────────────
class MLflowMetricsReader:
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient()
            self._available = True
        except ImportError:
            logger.warning("mlflow 未安装，指标查询将使用本地文件 fallback")
            self._available = False

    def get_production_metrics(self, model_name: str) -> Optional[dict]:
        """获取 Production 阶段模型的最新指标"""
        if not self._available:
            return self._read_local_metrics()

        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                logger.info(f"模型 {model_name} 没有 Production 版本")
                return None
            latest = versions[0]
            run = self.client.get_run(latest.run_id)
            return dict(run.data.metrics)
        except Exception as e:
            logger.error(f"获取 MLflow 指标失败: {e}")
            return self._read_local_metrics()

    def _read_local_metrics(self) -> Optional[dict]:
        """Fallback：从本地 eval_metrics.json 读取"""
        path = "models/metrics/eval_metrics.json"
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None


# ──────────────────────────────────────────────────────────
# 新数据统计
# ──────────────────────────────────────────────────────────
def count_new_data(raw_dir: str = "data/raw/", marker_file: str = ".last_train_count") -> int:
    """
    统计自上次训练以来新增的原始数据文件数量。
    marker_file 记录上次训练时的文件数量。
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    current_count = sum(
        1 for p in Path(raw_dir).rglob("*")
        if p.suffix.lower() in extensions
    ) if Path(raw_dir).exists() else 0

    last_count = 0
    if os.path.exists(marker_file):
        try:
            with open(marker_file) as f:
                last_count = int(f.read().strip())
        except ValueError:
            pass

    return max(0, current_count - last_count)


def update_data_marker(raw_dir: str = "data/raw/", marker_file: str = ".last_train_count"):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = sum(
        1 for p in Path(raw_dir).rglob("*")
        if p.suffix.lower() in extensions
    ) if Path(raw_dir).exists() else 0
    with open(marker_file, "w") as f:
        f.write(str(count))


# ──────────────────────────────────────────────────────────
# 通知
# ──────────────────────────────────────────────────────────
def send_slack_notification(webhook_url: str, message: str):
    if not webhook_url:
        return
    try:
        requests.post(webhook_url, json={"text": message}, timeout=10)
        logger.info("Slack 通知已发送")
    except Exception as e:
        logger.warning(f"Slack 通知失败: {e}")


def send_email_notification(to_email: str, subject: str, body: str):
    if not to_email:
        return
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    if not smtp_host or not smtp_user:
        logger.warning("SMTP 未配置，跳过邮件通知")
        return
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info(f"邮件通知已发送至 {to_email}")
    except Exception as e:
        logger.warning(f"邮件发送失败: {e}")


# ──────────────────────────────────────────────────────────
# 训练触发
# ──────────────────────────────────────────────────────────
def trigger_dvc_pipeline(force: bool = False, log_dir: str = "logs/retrain") -> bool:
    """
    调用 DVC repro 执行完整重训练流水线
    返回 True 表示成功
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"retrain_{timestamp}.log")

    cmd = ["dvc", "repro"]
    if force:
        cmd.append("--force")

    logger.info(f"触发 DVC 重训练: {' '.join(cmd)}")
    logger.info(f"日志路径: {log_path}")

    with open(log_path, "w") as log_file:
        log_file.write(f"=== 重训练开始: {timestamp} ===\n")
        log_file.write(f"命令: {' '.join(cmd)}\n\n")

        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        log_file.write(f"\n=== 退出码: {result.returncode} ===\n")

    success = result.returncode == 0
    logger.info(f"重训练{'成功' if success else '失败'}，退出码: {result.returncode}")
    return success


# ──────────────────────────────────────────────────────────
# 主调度器
# ──────────────────────────────────────────────────────────
class RetrainScheduler:
    """
    自动再训练调度器

    触发条件（任意一条满足即触发）：
    1. 精度下降超过 accuracy_drop 阈值
    2. 累积新数据超过 new_data_size 条
    3. 检测到数据漂移告警（由 API /alerts/drift 接口写入标志文件）
    """

    DRIFT_FLAG_FILE = ".drift_alert_pending"

    def __init__(self, config_path: str = "pipeline/configs/mlops.yaml"):
        self.cfg = load_config(config_path)
        retrain_cfg = self.cfg.get("retraining", {})
        alerts_cfg = self.cfg.get("alerts", {})
        registry_cfg = self.cfg.get("registry", {})

        self.enabled = retrain_cfg.get("enabled", True)
        triggers = retrain_cfg.get("triggers", {})
        self.accuracy_drop_threshold = triggers.get("accuracy_drop", 0.05)
        self.new_data_threshold = triggers.get("new_data_size", 500)
        self.drift_enabled = True

        self.model_name = registry_cfg.get("model_name", "visionops-detector")
        self.baseline_accuracy: Optional[float] = None

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.mlflow_reader = MLflowMetricsReader(tracking_uri)

        self.slack_webhook = alerts_cfg.get("slack_webhook", "")
        self.alert_email = alerts_cfg.get("email", "")

        self._running = False
        self._check_interval = 300  # 每5分钟检查一次条件（cron另行控制是否真正触发）

    def _notify(self, subject: str, message: str):
        full_msg = f"[VisionOps] {subject}\n{message}\n时间: {datetime.now().isoformat()}"
        logger.info(full_msg)
        send_slack_notification(self.slack_webhook, full_msg)
        send_email_notification(self.alert_email, f"[VisionOps] {subject}", message)

    def _check_accuracy_drop(self) -> tuple[bool, str]:
        """检查精度是否下降超过阈值"""
        metrics = self.mlflow_reader.get_production_metrics(self.model_name)
        if metrics is None:
            return False, "无法获取生产模型指标"

        current_acc = metrics.get("accuracy", metrics.get("val_acc", metrics.get("mAP50")))
        if current_acc is None:
            return False, "指标中没有 accuracy/val_acc/mAP50"

        if self.baseline_accuracy is None:
            self.baseline_accuracy = current_acc
            logger.info(f"初始化精度基准: {current_acc:.4f}")
            return False, "已初始化基准"

        drop = self.baseline_accuracy - current_acc
        if drop >= self.accuracy_drop_threshold:
            return True, f"精度下降 {drop:.4f} >= 阈值 {self.accuracy_drop_threshold}"

        return False, f"精度正常 (当前={current_acc:.4f}, 基准={self.baseline_accuracy:.4f})"

    def _check_new_data(self) -> tuple[bool, str]:
        """检查新数据积累量"""
        new_count = count_new_data()
        if new_count >= self.new_data_threshold:
            return True, f"新增数据 {new_count} >= 阈值 {self.new_data_threshold}"
        return False, f"新数据 {new_count} / {self.new_data_threshold}"

    def _check_drift_alert(self) -> tuple[bool, str]:
        """检查边缘端数据漂移告警"""
        if os.path.exists(self.DRIFT_FLAG_FILE):
            with open(self.DRIFT_FLAG_FILE) as f:
                content = f.read()
            return True, f"收到数据漂移告警: {content}"
        return False, "无漂移告警"

    def _clear_drift_flag(self):
        if os.path.exists(self.DRIFT_FLAG_FILE):
            os.remove(self.DRIFT_FLAG_FILE)

    def check_triggers(self) -> tuple[bool, str]:
        """检查所有触发条件，返回 (should_retrain, reason)"""
        if not self.enabled:
            return False, "自动再训练已禁用"

        # 漂移告警优先级最高
        triggered, reason = self._check_drift_alert()
        if triggered:
            return True, reason

        triggered, reason = self._check_accuracy_drop()
        if triggered:
            return True, reason

        triggered, reason = self._check_new_data()
        if triggered:
            return True, reason

        return False, "所有指标正常，无需重训练"

    def run_once(self) -> bool:
        """执行一次检查+可能触发的重训练"""
        should_retrain, reason = self.check_triggers()
        logger.info(f"再训练检查: {reason}")

        if not should_retrain:
            return False

        self._notify(
            "触发自动再训练",
            f"触发原因: {reason}\n即将执行 DVC pipeline 完整重训练..."
        )

        success = trigger_dvc_pipeline()

        if success:
            # 更新基准
            metrics = self.mlflow_reader.get_production_metrics(self.model_name)
            if metrics:
                new_acc = metrics.get("accuracy", metrics.get("val_acc", metrics.get("mAP50")))
                if new_acc:
                    self.baseline_accuracy = new_acc
            update_data_marker()
            self._clear_drift_flag()
            self._notify(
                "自动再训练完成",
                f"原因: {reason}\n训练成功，模型已更新到 MLflow Registry"
            )
        else:
            self._notify(
                "自动再训练失败",
                f"原因: {reason}\nDVC pipeline 执行失败，请查看 logs/retrain/ 日志"
            )

        return success

    def run_loop(self, check_interval: int = 300):
        """
        持续运行监控循环（后台线程模式）
        check_interval: 检查间隔秒数
        """
        self._running = True
        logger.info(f"再训练调度器启动，检查间隔: {check_interval}s")

        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"调度器异常: {e}", exc_info=True)
            time.sleep(check_interval)

    def stop(self):
        self._running = False

    def start_background(self, check_interval: int = 300) -> threading.Thread:
        """在后台线程启动调度器"""
        t = threading.Thread(
            target=self.run_loop,
            args=(check_interval,),
            daemon=True,
            name="retrain-scheduler"
        )
        t.start()
        logger.info("再训练调度器已在后台线程启动")
        return t


# ──────────────────────────────────────────────────────────
# CLI 入口（供 cron / GitHub Actions 调用）
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="VisionOps 再训练调度器")
    parser.add_argument("--config", default="pipeline/configs/mlops.yaml")
    parser.add_argument(
        "--mode",
        choices=["once", "loop"],
        default="once",
        help="once=单次检查并退出；loop=持续监控",
    )
    parser.add_argument("--interval", type=int, default=300, help="loop模式检查间隔(秒)")
    parser.add_argument("--force", action="store_true", help="强制重训练（跳过条件检查）")
    args = parser.parse_args()

    scheduler = RetrainScheduler(config_path=args.config)

    if args.force:
        logger.info("强制触发重训练...")
        trigger_dvc_pipeline(force=True)
    elif args.mode == "once":
        scheduler.run_once()
    else:
        scheduler.run_loop(check_interval=args.interval)
