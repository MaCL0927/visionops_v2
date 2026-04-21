"""
VisionOps FastAPI 后端 API Server
- 实验管理接口
- 模型仓库接口
- 边缘设备管理接口
- 触发训练/部署
"""
import os
import json
import subprocess
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient


# ── 初始化 ──
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

app = FastAPI(
    title="VisionOps API",
    description="视觉AI平台管理接口",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 内存存储（生产环境应替换为数据库） ──
edge_metrics_store: List[dict] = []
active_training_jobs: dict = {}


# ════════════════════════════════════════════
# 健康检查
# ════════════════════════════════════════════
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ════════════════════════════════════════════
# 实验管理
# ════════════════════════════════════════════
@app.get("/api/v1/experiments")
async def list_experiments():
    """列出所有MLflow实验"""
    try:
        experiments = client.search_experiments()
        return [{
            "id": e.experiment_id,
            "name": e.name,
            "lifecycle_stage": e.lifecycle_stage,
            "artifact_location": e.artifact_location,
        } for e in experiments]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/experiments/{experiment_id}/runs")
async def list_runs(experiment_id: str, max_results: int = 20):
    """列出实验中的运行记录"""
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        return [{
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "metrics": r.data.metrics,
            "params": r.data.params,
            "tags": r.data.tags,
        } for r in runs]
    except Exception as e:
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════
# 模型仓库
# ════════════════════════════════════════════
@app.get("/api/v1/models")
async def list_models():
    """列出所有注册模型"""
    try:
        models = client.search_registered_models()
        result = []
        for m in models:
            latest_versions = client.get_latest_versions(m.name)
            result.append({
                "name": m.name,
                "description": m.description,
                "latest_versions": [{
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "creation_timestamp": v.creation_timestamp,
                } for v in latest_versions],
            })
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/models/{model_name}/versions")
async def model_versions(model_name: str):
    """获取模型所有版本"""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return [{
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "creation_timestamp": v.creation_timestamp,
            "tags": v.tags,
            "description": v.description,
        } for v in versions]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/models/{model_name}/versions/{version}/promote")
async def promote_model(model_name: str, version: str, stage: str = "Production"):
    """手动晋升模型版本"""
    valid_stages = ["Staging", "Production", "Archived"]
    if stage not in valid_stages:
        raise HTTPException(400, f"无效stage，可选: {valid_stages}")
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        return {"success": True, "model": model_name, "version": version, "stage": stage}
    except Exception as e:
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════
# 训练触发
# ════════════════════════════════════════════
class TrainingRequest(BaseModel):
    experiment_name: Optional[str] = None
    config_overrides: Optional[dict] = None  # 覆盖训练配置
    force: bool = False


@app.post("/api/v1/training/trigger")
async def trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """触发训练流水线"""
    job_id = f"job-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def run_pipeline():
        active_training_jobs[job_id] = {"status": "running", "started_at": datetime.now().isoformat()}
        try:
            cmd = ["dvc", "repro"]
            if request.force:
                cmd.append("--force")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")
            active_training_jobs[job_id].update({
                "status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "finished_at": datetime.now().isoformat(),
                "stdout": result.stdout[-5000:],  # 限制长度
                "stderr": result.stderr[-2000:],
            })
        except Exception as e:
            active_training_jobs[job_id].update({"status": "error", "error": str(e)})

    background_tasks.add_task(run_pipeline)
    active_training_jobs[job_id] = {"status": "queued", "queued_at": datetime.now().isoformat()}

    return {"job_id": job_id, "status": "queued", "message": "训练任务已加入队列"}


@app.get("/api/v1/training/jobs")
async def list_training_jobs():
    return active_training_jobs


@app.get("/api/v1/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    if job_id not in active_training_jobs:
        raise HTTPException(404, "任务不存在")
    return active_training_jobs[job_id]


# ════════════════════════════════════════════
# 边缘设备管理
# ════════════════════════════════════════════
@app.get("/api/v1/edge/devices")
async def list_edge_devices():
    """从配置中读取边缘设备列表"""
    import yaml
    try:
        with open("pipeline/configs/mlops.yaml") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("edge_devices", [])
    except FileNotFoundError:
        return []


@app.post("/api/v1/edge/metrics")
async def receive_edge_metrics(metrics: dict):
    """接收边缘设备上报的指标"""
    metrics["received_at"] = datetime.now().isoformat()
    edge_metrics_store.append(metrics)
    # 保留最近500条
    if len(edge_metrics_store) > 500:
        edge_metrics_store.pop(0)
    return {"status": "ok"}


@app.get("/api/v1/edge/metrics")
async def get_edge_metrics(device_id: Optional[str] = None, limit: int = 50):
    """查询边缘设备指标"""
    data = edge_metrics_store
    if device_id:
        data = [m for m in data if m.get("device_id") == device_id]
    return data[-limit:]


@app.post("/api/v1/alerts/drift")
async def receive_drift_alert(alert: dict):
    """接收数据漂移告警"""
    alert["received_at"] = datetime.now().isoformat()
    # TODO: 集成Slack/Email告警
    # TODO: 触发自动再训练
    print(f"⚠️ 数据漂移告警: {json.dumps(alert, ensure_ascii=False)}")
    return {"status": "ok", "message": "告警已记录，将触发再训练检查"}


# ════════════════════════════════════════════
# 部署触发
# ════════════════════════════════════════════
class DeployRequest(BaseModel):
    model_name: str = "visionops-detector"
    version: Optional[str] = None  # None = Production版本
    device_id: Optional[str] = None  # None = 所有设备


@app.post("/api/v1/deploy")
async def trigger_deploy(request: DeployRequest, background_tasks: BackgroundTasks):
    """触发模型部署到边缘设备"""
    def run_deploy():
        cmd = ["bash", "edge/deploy/push.sh", "models/export/model.rknn"]
        if request.device_id:
            cmd.append(request.device_id)
        subprocess.run(cmd, cwd="/app")

    background_tasks.add_task(run_deploy)
    return {"status": "queued", "message": f"部署任务已触发，设备: {request.device_id or '所有'}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
