#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from server.annotation.annotation_app import router as annotation_router

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_CONTEXT_PATH = DATA_DIR / "model_context" / "manifest.json"
WORKFLOW_RUNS_DIR = DATA_DIR / "workflow_runs"
TASK_YAML_PATH = PROJECT_ROOT / "pipeline" / "configs" / "task.yaml"
TASK_YAML_DISPLAY_PATH = str(TASK_YAML_PATH.relative_to(PROJECT_ROOT))
PRESETS_DIR = PROJECT_ROOT / "pipeline" / "configs" / "presets"
INCOMING_DIR = DATA_DIR / "incoming"
RAW_COLLECTED_DIR = DATA_DIR / "raw_collected"
COLLECTED_INDEX_PATH = DATA_DIR / "collected_batches" / "index.json"

# X-AnyLabeling 启动命令。
# 你的实际命令是 xanylabeling，且安装在 x-anylabeling-cu12 环境中。
X_ANYLABELING_CMD = os.environ.get("X_ANYLABELING_CMD", "xanylabeling")
X_ANYLABELING_ENV = os.environ.get("X_ANYLABELING_ENV", "x-anylabeling-cu12")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_DEFAULT_EXPERIMENT = os.environ.get("VISIONOPS_MLFLOW_EXPERIMENT", "visionops-detection")


app = FastAPI(title="VisionOps Workflow Control Panel")
app.include_router(annotation_router)

def build_x_anylabeling_cmd(images_dir: Path) -> list[str]:
    if X_ANYLABELING_ENV:
        return [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            X_ANYLABELING_ENV,
            X_ANYLABELING_CMD,
            "--filename",
            str(images_dir),
        ]

    return [X_ANYLABELING_CMD, "--filename", str(images_dir)]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_incoming_packages() -> list[dict[str, Any]]:
    """列出 data/incoming 下待处理的 tar.gz 上传包，按修改时间倒序。"""
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)

    packages: list[dict[str, Any]] = []
    for p in sorted(INCOMING_DIR.glob("*.tar.gz"), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_file():
            continue
        stat = p.stat()
        packages.append({
            "name": p.name,
            "path": str(p.relative_to(PROJECT_ROOT)),
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })

    return packages


def yaml_module():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"缺少 PyYAML，请先安装 pyyaml: {exc}")


def validate_yaml_content(content: str) -> None:
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=400, detail="task.yaml 内容不能为空")

    try:
        yaml = yaml_module()
        parsed = yaml.safe_load(content)
        if not isinstance(parsed, dict):
            raise ValueError("task.yaml 顶层必须是字典")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"YAML 格式错误: {exc}")


def get_current_batch_dir() -> Path:
    manifest = read_json(MODEL_CONTEXT_PATH, default={}) or {}

    source_manifest = manifest.get("source_manifest")
    if source_manifest:
        p = Path(source_manifest)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.parent

    package_name = manifest.get("package_name") or manifest.get("batch_id")
    if package_name:
        return PROJECT_ROOT / "data" / "raw_collected" / package_name

    raise RuntimeError(
        "未找到当前 batch。请先点击“处理上传数据集”，生成 data/model_context/manifest.json。"
    )


def get_dataset_yaml_path(task_type: str) -> Path:
    if task_type in {"classification", "cls"}:
        return PROJECT_ROOT / "data" / "raw_classification" / "data.yaml"
    if task_type in {"obb", "obb_detection"}:
        return PROJECT_ROOT / "data" / "raw_obb" / "data.yaml"
    return PROJECT_ROOT / "data" / "raw_detection" / "data.yaml"


def read_dataset_classes(task_type: str) -> tuple[list[str], Path]:
    """从确认审核完成后生成的 data.yaml 中读取 names/nc。"""
    yaml = yaml_module()
    data_yaml = get_dataset_yaml_path(task_type)

    if not data_yaml.exists():
        raise HTTPException(
            status_code=404,
            detail=f"未找到 {data_yaml.relative_to(PROJECT_ROOT)}。请先点击“确认审核完成”。",
        )

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names")

    if isinstance(names, dict):
        # YOLO 常见形式：names: {0: box, 1: phone}
        names = [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
    elif isinstance(names, list):
        names = [str(x) for x in names]
    else:
        raise HTTPException(status_code=400, detail=f"{data_yaml.relative_to(PROJECT_ROOT)} 中没有有效 names 字段")

    if not names:
        raise HTTPException(status_code=400, detail=f"{data_yaml.relative_to(PROJECT_ROOT)} 中类别为空")

    return names, data_yaml


def update_task_classes(cfg: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """
    将 data.yaml 中的类别同步到 task.yaml：
      classes.names
      classes.num_classes

    同时递归同步已有的 num_classes/nc/class_names 字段，尽量兼容旧配置。
    """
    cfg.setdefault("classes", {})
    if not isinstance(cfg["classes"], dict):
        cfg["classes"] = {}
    cfg["classes"]["names"] = names
    cfg["classes"]["num_classes"] = len(names)

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if key == "num_classes":
                    obj[key] = len(names)
                elif key == "nc":
                    obj[key] = len(names)
                elif key == "class_names":
                    obj[key] = names
                elif key == "names" and isinstance(value, (list, dict)):
                    # 只替换明显用于类别名的 names，路径字段不会是 list/dict。
                    obj[key] = names
                else:
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(cfg)
    return cfg


def format_duration_ms(ms: int | float | None) -> str:
    if ms is None:
        return "-"
    try:
        seconds = max(0.0, float(ms) / 1000.0)
    except Exception:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    return f"{hours:.1f}h"


def format_time_ms(ms: int | float | None) -> str:
    if ms is None:
        return "-"
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def pick_metric(metrics: dict[str, Any], candidates: list[str]) -> float | None:
    for key in candidates:
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                continue
    return None


def pick_param(params: dict[str, Any], candidates: list[str]) -> str:
    for key in candidates:
        value = params.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return "-"


def discover_local_model_artifacts() -> dict[str, bool | str]:
    candidates = {
        "best_pt": [
            PROJECT_ROOT / "models" / "checkpoints_detection" / "best.pt",
            PROJECT_ROOT / "models" / "checkpoints" / "best.pt",
        ],
        "onnx": [
            PROJECT_ROOT / "models" / "export_detection" / "model.onnx",
            PROJECT_ROOT / "models" / "export" / "model.onnx",
        ],
        "rknn": [
            PROJECT_ROOT / "models" / "export_detection" / "model.rknn",
            PROJECT_ROOT / "models" / "export" / "model.rknn",
        ],
    }
    result: dict[str, bool | str] = {}
    for key, paths in candidates.items():
        existing = next((path for path in paths if path.exists()), None)
        result[key] = bool(existing)
        result[f"{key}_path"] = str(existing.relative_to(PROJECT_ROOT)) if existing else ""
    return result


def make_run_url(experiment_id: str, run_id: str) -> str:
    return f"{MLFLOW_TRACKING_URI.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"


def run_to_summary(run: Any, experiment_id: str) -> dict[str, Any]:
    info = run.info
    data = run.data
    metrics = dict(data.metrics or {})
    params = dict(data.params or {})
    start_ms = getattr(info, "start_time", None)
    end_ms = getattr(info, "end_time", None)
    duration_ms = (end_ms - start_ms) if start_ms and end_ms else None
    run_name = getattr(info, "run_name", None) or params.get("run_name") or getattr(info, "run_id", "")
    metric_map = {
        "best_map50": pick_metric(metrics, ["best_map50", "map50", "metrics/mAP50(B)", "mAP50", "box_map50"]),
        "best_map50_95": pick_metric(metrics, ["best_map50_95", "map50_95", "metrics/mAP50-95(B)", "mAP50-95", "box_map"]),
        "precision": pick_metric(metrics, ["precision", "metrics/precision(B)", "box_precision"]),
        "recall": pick_metric(metrics, ["recall", "metrics/recall(B)", "box_recall"]),
    }
    return {
        "run_id": getattr(info, "run_id", ""),
        "run_name": run_name,
        "status": getattr(info, "status", ""),
        "created_at": format_time_ms(start_ms),
        "duration": format_duration_ms(duration_ms),
        "metrics": metric_map,
        "params": {
            "batch_size": pick_param(params, ["batch_size", "batch"]),
            "epochs": pick_param(params, ["epochs"]),
            "imgsz": pick_param(params, ["imgsz", "input_size"]),
            "model": pick_param(params, ["model", "base_model"]),
        },
        "url": make_run_url(str(experiment_id), getattr(info, "run_id", "")),
    }


def get_mlflow_client() -> Any:
    try:
        from mlflow.tracking import MlflowClient  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"MLflow Python 包不可用: {exc}") from exc
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def fetch_mlflow_runs(experiment_name: str, limit: int = 20) -> dict[str, Any]:
    local_artifacts = discover_local_model_artifacts()
    try:
        client = get_mlflow_client()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return {
                "ok": False,
                "available": False,
                "message": f"未找到 MLflow 实验: {experiment_name}",
                "tracking_uri": MLFLOW_TRACKING_URI,
                "experiment_name": experiment_name,
                "runs": [],
                "local_artifacts": local_artifacts,
            }
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=max(1, min(int(limit), 50)),
        )
        summaries = [run_to_summary(run, exp.experiment_id) for run in runs]
        if summaries:
            summaries[0]["artifacts"] = local_artifacts
            summaries[0]["deployable"] = bool(local_artifacts.get("rknn")) and summaries[0].get("status") == "FINISHED"
        return {
            "ok": True,
            "available": True,
            "message": "ok",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": experiment_name,
            "experiment_id": exp.experiment_id,
            "runs": summaries,
            "latest_run": summaries[0] if summaries else None,
            "local_artifacts": local_artifacts,
        }
    except Exception as exc:
        return {
            "ok": False,
            "available": False,
            "message": f"MLflow 暂不可用，请确认 {MLFLOW_TRACKING_URI} 是否启动。错误: {exc}",
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": experiment_name,
            "runs": [],
            "local_artifacts": local_artifacts,
        }


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def start(self, name: str, cmd: list[str]) -> str:
        job_id = uuid.uuid4().hex[:12]
        job_dir = WORKFLOW_RUNS_DIR / job_id
        log_path = job_dir / "run.log"
        status_path = job_dir / "status.json"
        job_dir.mkdir(parents=True, exist_ok=True)

        status = {
            "job_id": job_id,
            "name": name,
            "cmd": cmd,
            "status": "running",
            "started_at": now_str(),
            "finished_at": "",
            "returncode": None,
            "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        }
        write_json(status_path, status)

        with self.lock:
            self.jobs[job_id] = status

        def runner() -> None:
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"[INFO] job_id={job_id}\n")
                log_file.write(f"[INFO] name={name}\n")
                log_file.write(f"[INFO] cmd={' '.join(cmd)}\n")
                log_file.write(f"[INFO] cwd={PROJECT_ROOT}\n")
                log_file.write("=" * 80 + "\n")
                log_file.flush()

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy(),
                )

                assert proc.stdout is not None
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()

                returncode = proc.wait()

            status["status"] = "success" if returncode == 0 else "failed"
            status["finished_at"] = now_str()
            status["returncode"] = returncode
            write_json(status_path, status)

            with self.lock:
                self.jobs[job_id] = status

        t = threading.Thread(target=runner, daemon=True)
        t.start()

        return job_id

    def get(self, job_id: str) -> dict[str, Any]:
        status_path = WORKFLOW_RUNS_DIR / job_id / "status.json"
        status = read_json(status_path, default=None)
        if not status:
            raise KeyError(job_id)
        return status

    def logs(self, job_id: str) -> str:
        log_path = WORKFLOW_RUNS_DIR / job_id / "run.log"
        if not log_path.exists():
            return ""
        return log_path.read_text(encoding="utf-8", errors="ignore")


jobs = JobManager()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    manifest = read_json(MODEL_CONTEXT_PATH, default={}) or {}

    current_batch_dir = ""
    all_images_dir = ""
    all_images_count = 0
    labels_dir = ""
    labels_count = 0

    try:
        batch_dir = get_current_batch_dir()
        current_batch_dir = str(batch_dir.relative_to(PROJECT_ROOT))
        all_images = batch_dir / "all_images"
        labels = batch_dir / "labels"
        all_images_dir = str(all_images.relative_to(PROJECT_ROOT))
        labels_dir = str(labels.relative_to(PROJECT_ROOT))
        if all_images.exists():
            all_images_count = len([
                p for p in all_images.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ])
        if labels.exists():
            labels_count = len([p for p in labels.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    except Exception:
        pass

    package_items = list_incoming_packages()

    return {
        "project_root": str(PROJECT_ROOT),
        "incoming_dir": "data/incoming",
        "pending_packages": [x["path"] for x in package_items],
        "pending_package_items": package_items,
        "pending_package_count": len(package_items),
        "model_context_exists": MODEL_CONTEXT_PATH.exists(),
        "model_context": manifest,
        "current_batch_dir": current_batch_dir,
        "all_images_dir": all_images_dir,
        "all_images_count": all_images_count,
        "labels_dir": labels_dir,
        "labels_count": labels_count,
        "x_anylabeling_cmd": X_ANYLABELING_CMD,
        "x_anylabeling_env": X_ANYLABELING_ENV,
        "task_yaml_exists": TASK_YAML_PATH.exists(),
    }


@app.get("/api/mlflow/latest-model-status")
def api_mlflow_latest_model_status(experiment: str = MLFLOW_DEFAULT_EXPERIMENT) -> dict[str, Any]:
    return fetch_mlflow_runs(experiment_name=experiment, limit=20)


@app.get("/api/mlflow/runs")
def api_mlflow_runs(experiment: str = MLFLOW_DEFAULT_EXPERIMENT, limit: int = 20) -> dict[str, Any]:
    return fetch_mlflow_runs(experiment_name=experiment, limit=limit)


@app.get("/api/mlflow/config")
def api_mlflow_config() -> dict[str, Any]:
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "default_experiment": MLFLOW_DEFAULT_EXPERIMENT,
    }


@app.get("/api/incoming-packages")
def api_incoming_packages() -> dict[str, Any]:
    packages = list_incoming_packages()
    return {
        "incoming_dir": str(INCOMING_DIR.relative_to(PROJECT_ROOT)),
        "count": len(packages),
        "packages": packages,
    }


@app.post("/api/ingest")
def api_ingest(payload: dict[str, Any]) -> dict[str, Any]:
    selected_packages = payload.get("packages") or []
    if not isinstance(selected_packages, list):
        raise HTTPException(status_code=400, detail="packages 必须是数组")

    selected_packages = [str(x).strip() for x in selected_packages if str(x).strip()]
    if not selected_packages:
        raise HTTPException(status_code=400, detail="请先选择要处理的上传压缩包")

    package_paths: list[Path] = []
    incoming_root = INCOMING_DIR.resolve()

    for rel in selected_packages:
        p = (PROJECT_ROOT / rel).resolve()
        try:
            p.relative_to(incoming_root)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"非法压缩包路径: {rel}")

        if not p.exists():
            raise HTTPException(status_code=404, detail=f"压缩包不存在: {rel}")

        if not p.name.endswith(".tar.gz"):
            raise HTTPException(status_code=400, detail=f"只支持 .tar.gz 压缩包: {rel}")

        package_paths.append(p)

    cmd = [
        "python",
        "server/data_ingest/ingest_uploaded_package.py",
        "--incoming-dir",
        "data/incoming",
        "--raw-collected-dir",
        "data/raw_collected",
        "--index-path",
        "data/collected_batches/index.json",
        "--overwrite",
    ]

    rel_paths = [str(p.relative_to(PROJECT_ROOT)) for p in package_paths]
    if len(package_paths) == 1:
        cmd.extend(["--package", rel_paths[0]])
        name = "ingest-selected-package"
        message = f"已开始处理选中的上传包: {package_paths[0].name}"
    else:
        cmd.append("--packages")
        cmd.extend(rel_paths)
        name = "ingest-selected-packages"
        message = f"已开始合并处理 {len(package_paths)} 个上传包"

    job_id = jobs.start(name, cmd)

    return {
        "job_id": job_id,
        "message": message,
        "cmd": " ".join(cmd),
    }


@app.post("/api/open-annotator")
def api_open_annotator() -> dict[str, Any]:
    try:
        batch_dir = get_current_batch_dir()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    all_images_dir = batch_dir / "all_images"
    if not all_images_dir.exists():
        raise HTTPException(status_code=404, detail=f"未找到图片目录: {all_images_dir}")

    try:
        cmd = build_x_anylabeling_cmd(all_images_dir)
        subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=os.environ.copy())
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=(
                f"找不到 X-AnyLabeling 启动命令: {X_ANYLABELING_CMD}。"
                "请先安装 X-AnyLabeling，或设置环境变量 X_ANYLABELING_CMD。"
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "message": "已尝试打开 X-AnyLabeling",
        "images_dir": str(all_images_dir.relative_to(PROJECT_ROOT)),
        "cmd": " ".join(cmd),
        "tip": "请在 X-AnyLabeling 中导出 YOLO 格式标注到当前 batch 的 labels 目录，完成后关闭工具。",
    }


@app.post("/api/accept-reviewed")
def api_accept_reviewed() -> dict[str, Any]:
    job_id = jobs.start(
        "accept-reviewed-detection",
        ["python", "server/workflow/accept_reviewed_detection.py", "--move"],
    )
    return {"job_id": job_id, "message": "已开始确认审核完成并转换到 data/raw_detection"}


@app.get("/api/task-yaml")
def api_get_task_yaml() -> dict[str, Any]:
    if not TASK_YAML_PATH.exists():
        return {"exists": False, "path": TASK_YAML_DISPLAY_PATH, "content": ""}

    return {
        "exists": True,
        "path": "task.yaml",
        "content": TASK_YAML_PATH.read_text(encoding="utf-8"),
    }


@app.post("/api/task-yaml")
def api_save_task_yaml(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload.get("content", "")
    validate_yaml_content(content)
    TASK_YAML_PATH.write_text(content, encoding="utf-8")
    return {"message": "pipeline/configs/task.yaml 已保存", "path": TASK_YAML_DISPLAY_PATH}


@app.post("/api/prepare-task-yaml")
def api_prepare_task_yaml(payload: dict[str, Any]) -> dict[str, Any]:
    """选择任务类型后，从 preset 生成 task.yaml，并自动同步 data.yaml 中的类别信息。"""
    task_type = str(payload.get("task_type") or "").strip()
    preset_map = {
        "classification": "classification.yaml",
        "detection": "detection.yaml",
        "yolo_detection": "detection.yaml",
        "obb": "obb.yaml",
        "obb_detection": "obb.yaml",
    }

    filename = preset_map.get(task_type)
    if not filename:
        raise HTTPException(status_code=400, detail=f"未知任务类型: {task_type}")

    preset_path = PRESETS_DIR / filename
    if not preset_path.exists():
        raise HTTPException(status_code=404, detail=f"未找到 preset: {preset_path}")

    yaml = yaml_module()
    cfg = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail=f"preset 顶层不是字典: {preset_path}")

    names, data_yaml = read_dataset_classes(task_type)
    cfg = update_task_classes(cfg, names)

    content = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    TASK_YAML_PATH.write_text(content, encoding="utf-8")

    return {
        "message": "已从模板生成 task.yaml，并自动同步类别信息",
        "task_type": task_type,
        "preset": str(preset_path.relative_to(PROJECT_ROOT)),
        "data_yaml": str(data_yaml.relative_to(PROJECT_ROOT)),
        "num_classes": len(names),
        "names": names,
        "content": content,
    }


@app.post("/api/pipeline-confirmed")
def api_pipeline_confirmed(payload: dict[str, Any]) -> dict[str, Any]:
    content = payload.get("content", "")
    validate_yaml_content(content)
    TASK_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    TASK_YAML_PATH.write_text(content, encoding="utf-8")

    job_id = jobs.start("pipeline", ["make", "pipeline"])
    return {"job_id": job_id, "message": "task.yaml 已保存，训练流水线已开始"}


@app.post("/api/pipeline")
def api_pipeline() -> dict[str, Any]:
    job_id = jobs.start("pipeline", ["make", "pipeline"])
    return {"job_id": job_id, "message": "已开始训练流水线"}


@app.post("/api/deploy")
def api_deploy() -> dict[str, Any]:
    job_id = jobs.start("deploy", ["make", "deploy"])
    return {"job_id": job_id, "message": "已开始部署模型"}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str) -> dict[str, Any]:
    try:
        return jobs.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"未知任务: {job_id}")


@app.get("/api/jobs/{job_id}/logs")
def api_job_logs(job_id: str) -> JSONResponse:
    try:
        status = jobs.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"未知任务: {job_id}")

    return JSONResponse({"status": status, "logs": jobs.logs(job_id)})


HTML_PAGE = r'''
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>VisionOps 数据闭环控制台</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; background: #f5f7fb; color: #172033; }
    .page { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 6px 0; font-size: 28px; }
    .subtitle { color: #64748b; margin-bottom: 16px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card, .status { background: white; border-radius: 16px; padding: 18px; box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08); border: 1px solid #e5e7eb; }
    .card h2, .status h2 { font-size: 18px; margin: 0 0 8px 0; }
    .card p, .status p { color: #64748b; line-height: 1.5; margin: 6px 0 12px; }
    button { border: 0; border-radius: 12px; padding: 11px 16px; font-size: 15px; cursor: pointer; background: #2563eb; color: white; font-weight: 600; }
    button.secondary { background: #0f172a; }
    button.warning { background: #d97706; }
    button.success { background: #059669; }
    button.ghost { background: #e5e7eb; color: #111827; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status { margin-bottom: 16px; }
    .kv { display: grid; grid-template-columns: 180px 1fr; gap: 7px 12px; font-size: 14px; }
    .key { color: #64748b; }
    .value { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; word-break: break-all; }
    .log { background: #0b1020; color: #d1e7ff; border-radius: 14px; padding: 14px; min-height: 180px; max-height: 300px; overflow: auto; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; line-height: 1.45; }
    .modal-mask { display:none; position:fixed; inset:0; background:rgba(15,23,42,0.46); z-index:1000; align-items:center; justify-content:center; padding:20px; }
    .modal { width:min(920px, 96vw); max-height:92vh; overflow:auto; background:white; border-radius:18px; box-shadow:0 24px 80px rgba(15,23,42,0.35); padding:22px; }
    .modal.narrow { width:min(620px, 96vw); }
    .modal h2 { margin: 0 0 10px; font-size: 20px; }
    .modal p { color:#64748b; margin:6px 0 14px; line-height:1.55; }
    .btn-row { display:flex; gap:10px; flex-wrap:wrap; margin: 12px 0; }
    textarea { width:100%; height:440px; border:1px solid #d1d5db; border-radius:12px; padding:14px; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size:13px; line-height:1.5; box-sizing:border-box; }
    input.text-input { width:100%; border:1px solid #d1d5db; border-radius:12px; padding:11px 12px; font-size:14px; box-sizing:border-box; margin:6px 0 10px; }
    label.field { display:block; color:#334155; font-size:14px; font-weight:700; margin-top:8px; }
    .hint { background:#f8fafc; border:1px solid #e5e7eb; border-radius:12px; padding:12px; color:#475569; font-size:14px; line-height:1.55; }
    .ingest-card, .deploy-card { grid-column: 1 / -1; }
    .ingest-layout { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; align-items: start; }
    .package-list { border: 1px solid #e5e7eb; border-radius: 12px; background: #f8fafc; max-height: 260px; overflow: auto; padding: 8px; }
    .package-item { display: grid; grid-template-columns: 28px 1fr auto; gap: 8px; align-items: center; padding: 9px 8px; border-bottom: 1px solid #e5e7eb; font-size: 13px; cursor: pointer; }
    .package-item:last-child { border-bottom: none; }
    .package-name { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; word-break: break-all; color: #0f172a; }
    .package-meta { color: #64748b; font-size: 12px; margin-top: 3px; }
    .badge { display:inline-block; background:#dbeafe; color:#1d4ed8; border-radius:999px; padding:2px 8px; font-size:12px; font-weight:700; }
    .empty { color:#64748b; padding:12px; }
    .selected-summary { margin-top: 10px; color:#334155; font-size:14px; }
    .step-note { color:#64748b; font-size:13px; margin-top:8px; }
    .status-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .pill { background:#f1f5f9; color:#334155; border:1px solid #e2e8f0; border-radius:999px; padding:6px 10px; font-size:13px; }
    .metric-grid { display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:12px; }
    .metric { background:#f8fafc; border:1px solid #e5e7eb; border-radius:12px; padding:12px; }
    .metric .label { color:#64748b; font-size:12px; margin-bottom:6px; }
    .metric .num { color:#0f172a; font-size:18px; font-weight:800; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .deploy-layout { display:grid; grid-template-columns: 1fr 1fr; gap:16px; align-items:start; }
    .mlflow-head { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; flex-wrap:wrap; }
    .mlflow-status { display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:5px 10px; font-size:13px; font-weight:800; }
    .mlflow-status.ok { background:#dcfce7; color:#166534; }
    .mlflow-status.warn { background:#fff7ed; color:#9a3412; }
    .mlflow-status.err { background:#fee2e2; color:#991b1b; }
    .artifact-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
    .artifact { border-radius:999px; padding:5px 9px; font-size:12px; font-weight:800; border:1px solid #e5e7eb; }
    .artifact.ok { background:#dcfce7; color:#166534; border-color:#bbf7d0; }
    .artifact.no { background:#f1f5f9; color:#64748b; }
    table.run-table { width:100%; border-collapse:collapse; margin-top:12px; font-size:13px; }
    table.run-table th, table.run-table td { border-bottom:1px solid #e5e7eb; padding:8px 7px; text-align:left; vertical-align:top; }
    table.run-table th { color:#475569; background:#f8fafc; position:sticky; top:0; }
    .run-name { max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .model-details-toggle { margin-top: 10px; }
    .model-details { display:none; margin-top: 10px; }
    .model-details.open { display:block; }
    .model-summary-line { margin-top: 8px; color:#334155; font-size:14px; line-height:1.65; }
    .model-mini-metrics { display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:10px; }
    @media (max-width: 860px) { .grid, .ingest-layout, .deploy-layout, .metric-grid, .model-mini-metrics { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="page">
    <h1>VisionOps 数据闭环控制台</h1>
    <div class="subtitle">接收数据 → 标注与审核 → 训练与模型状态 → 模型部署</div>

    <div class="status">
      <h2>运行日志</h2>
      <div id="log" class="log">暂无任务日志。</div>
    </div>

    <div class="grid">
      <div class="card ingest-card">
        <div class="ingest-layout">
          <div>
            <h2>1. 接收并处理上传包</h2>
            <p>展示 data/incoming 下尚未处理的 tar.gz。勾选 1 个包时按单包处理；勾选多个包时自动合并为一个 batch，并只生成一个 manifest。</p>
            <div class="btn-row"><button class="ghost" onclick="refreshIncomingPackages()">刷新上传包</button></div>
            <div id="incomingPackages" class="package-list">正在加载...</div>
          </div>
          <div>
            <h2>当前数据上下文</h2>
            <div id="currentContext" class="metric-grid">
              <div class="metric"><div class="label">当前 batch</div><div class="num">-</div></div>
              <div class="metric"><div class="label">图片 / 标签</div><div class="num">-</div></div>
            </div>
            <div class="btn-row"><button onclick="startIngestSelected()">处理选中上传包</button></div>
            <div id="selectedSummary" class="selected-summary">当前未选择上传包。</div>
            <div class="hint" style="margin-top:12px;">建议只合并同一设备、同一客户、同一类别体系的数据包；不同任务类型的数据不要合并。</div>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>2. 标注与审核</h2>
        <p>打开 VisionOps 标注器进行半自动标注、人工审核和快速学习。审核完成按钮已移动到标注器右下角。</p>
        <div class="btn-row">
          <button class="secondary" onclick="openVisionOpsAnnotator()">打开 VisionOps 标注器</button>
          <button class="secondary" onclick="openAnnotator()">打开 X-AnyLabeling</button>
        </div>
        <div class="step-note">完成标注后，在标注器页面右下角点击“确认审核完成”。</div>
      </div>

      <div class="card">
        <div class="mlflow-head">
          <div>
            <h2>3. 训练与模型状态</h2>
          </div>
          <span id="mlflowConnBadge" class="mlflow-status warn">MLflow: 未刷新</span>
        </div>
        <div class="btn-row">
          <button class="success" onclick="openPipelineModal()">开始训练流水线</button>
          <button class="ghost" onclick="refreshModelStatus()">刷新模型状态</button>
          <button class="secondary" onclick="openMlflow()">打开 MLflow</button>
          <button class="secondary" onclick="openRunHistoryModal()">查看历史模型</button>
        </div>
        <div class="model-details-toggle">
          <button id="modelDetailsToggleBtn" class="ghost" onclick="toggleModelDetails()">展开详情</button>
        </div>
        <div id="modelDetailsPanel" class="model-details">
          <div class="status-pills">
            <span id="taskYamlPill" class="pill">task.yaml: -</span>
            <span id="modelContextPill" class="pill">manifest: -</span>
            <span id="mlflowExperimentPill" class="pill">experiment: visionops-detection</span>
          </div>
          <div id="modelStatusBox" class="hint" style="margin-top:12px;">等待刷新 MLflow 状态。</div>
        </div>
      </div>

      <div class="card deploy-card">
        <h2>4. 模型部署</h2>
        <div class="deploy-layout">
          <div>
            <p>部署当前设备会执行 make deploy，沿用当前项目配置中的设备信息、模型路径和健康检查逻辑。</p>
            <div class="btn-row">
              <button onclick="startJob('/api/deploy')">部署当前设备</button>
              <button class="secondary" onclick="openDeployOtherModal()">部署其他设备</button>
            </div>
          </div>
          <div class="hint">
            当前部署链路通常由 make deploy 调用 edge/deploy/push.sh 完成，通过 SSH/SCP 把同一个 RKNN 模型、类别配置和运行时上下文推送到边缘设备，再重启服务并健康检查。<br><br>
            “部署其他设备”先作为界面入口预留，下一步可接入设备 IP、SSH 用户、端口、部署目录等参数。
          </div>
        </div>
      </div>
    </div>

    <div class="status" style="margin-top:16px;">
      <h2>状态信息</h2>
      <div id="status" class="kv"></div>
    </div>
  </div>

  <div id="pipelineModal" class="modal-mask">
    <div class="modal">
      <h2>开始训练流水线</h2>
      <div id="taskTypeStep">
        <p>请选择任务类型。系统会从 pipeline/configs/presets 复制对应模板到 task.yaml，并自动从 data.yaml 同步 classes.names 和 classes.num_classes。</p>
        <div class="btn-row">
          <button onclick="prepareTaskYaml('classification')">分类任务</button>
          <button onclick="prepareTaskYaml('detection')">检测任务</button>
          <button onclick="prepareTaskYaml('obb')">OBB 任务</button>
          <button class="ghost" onclick="closePipelineModal()">取消</button>
        </div>
      </div>
      <div id="taskEditStep" style="display:none;">
        <div class="hint" id="taskHint"></div>
        <p>下面是已自动生成的 task.yaml。通常只需要确认类别、数据路径和训练参数是否正确；确认后点击“保存并开始训练”。</p>
        <textarea id="taskYamlText"></textarea>
        <div class="btn-row">
          <button class="success" onclick="saveAndStartPipeline()">保存并开始训练</button>
          <button class="secondary" onclick="backToTaskTypeStep()">重新选择任务类型</button>
          <button class="ghost" onclick="closePipelineModal()">取消</button>
        </div>
      </div>
    </div>
  </div>

  <div id="runHistoryModal" class="modal-mask">
    <div class="modal">
      <h2>历史模型记录</h2>
      <p>展示 MLflow 实验 visionops-detection 最近 20 个 run，便于快速比较训练结果。</p>
      <div class="btn-row">
        <button class="ghost" onclick="refreshRunHistory()">刷新历史记录</button>
        <button class="secondary" onclick="openMlflow()">打开 MLflow</button>
        <button class="ghost" onclick="closeRunHistoryModal()">关闭</button>
      </div>
      <div id="runHistoryBox" class="hint">等待加载。</div>
    </div>
  </div>

  <div id="deployOtherModal" class="modal-mask">
    <div class="modal narrow">
      <h2>部署其他设备</h2>
      <p>这里先完成界面入口预留。下一步把这些字段接入 push.sh 参数或环境变量后，即可把当前模型部署到指定设备。</p>
      <label class="field">设备 ID</label><input id="otherDeviceId" class="text-input" placeholder="例如 rk3588-002" />
      <label class="field">设备 IP / Host</label><input id="otherDeviceHost" class="text-input" placeholder="例如 192.168.1.202" />
      <label class="field">SSH 用户</label><input id="otherDeviceUser" class="text-input" value="ubuntu" />
      <label class="field">SSH 端口</label><input id="otherDevicePort" class="text-input" value="22" />
      <label class="field">部署目录</label><input id="otherDeviceRoot" class="text-input" value="/opt/visionops" />
      <div class="btn-row">
        <button class="secondary" onclick="previewDeployOther()">生成部署参数预览</button>
        <button class="ghost" onclick="closeDeployOtherModal()">关闭</button>
      </div>
      <div id="deployOtherPreview" class="hint">尚未生成预览。</div>
    </div>
  </div>

<script>
let currentJobId = null;
let timer = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderStatus(data) {
  const rows = [
    ["项目根目录", data.project_root],
    ["上传包目录", data.incoming_dir],
    ["待处理上传包", `${data.pending_package_count || 0} 个`],
    ["manifest 是否存在", data.model_context_exists ? "是" : "否"],
    ["当前 batch 目录", data.current_batch_dir || "未生成"],
    ["图片目录", data.all_images_dir || "未生成"],
    ["图片数量", String(data.all_images_count)],
    ["导出标签目录", data.labels_dir || "未生成"],
    ["导出标签数量", String(data.labels_count)],
    ["task.yaml 是否存在", data.task_yaml_exists ? "是" : "否"],
    ["X-AnyLabeling 命令", data.x_anylabeling_cmd],
    ["X-AnyLabeling 环境", data.x_anylabeling_env || "当前环境"],
    ["device_id", data.model_context.device_id || ""],
    ["customer_id", data.model_context.customer_id || ""],
    ["package_name", data.model_context.package_name || ""],
  ];
  document.getElementById("status").innerHTML = rows.map(([k, v]) => `
    <div class="key">${escapeHtml(k)}</div><div class="value">${escapeHtml(v)}</div>
  `).join("");
  renderCurrentContext(data);
  renderModelStatus(data);
}

function renderCurrentContext(data) {
  const batch = data.current_batch_dir || "未生成";
  const imgCount = Number(data.all_images_count || 0);
  const labelCount = Number(data.labels_count || 0);
  document.getElementById("currentContext").innerHTML = `
    <div class="metric"><div class="label">当前 batch</div><div class="num" title="${escapeHtml(batch)}">${escapeHtml(data.model_context.package_name || data.model_context.batch_id || "-")}</div></div>
    <div class="metric"><div class="label">图片 / 标签</div><div class="num">${imgCount} / ${labelCount}</div></div>`;
}

let mlflowState = null;
let modelDetailsOpen = false;

function toggleModelDetails(forceOpen) {
  if (typeof forceOpen === "boolean") modelDetailsOpen = forceOpen;
  else modelDetailsOpen = !modelDetailsOpen;
  const panel = document.getElementById("modelDetailsPanel");
  const btn = document.getElementById("modelDetailsToggleBtn");
  if (!panel || !btn) return;
  panel.className = modelDetailsOpen ? "model-details open" : "model-details";
  btn.textContent = modelDetailsOpen ? "收起详情" : "展开详情";
}

function metricText(v) {
  if (v === null || v === undefined || v === "") return "-";
  const n = Number(v);
  if (!Number.isFinite(n)) return escapeHtml(v);
  return n.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function artifactBadge(label, ok) {
  return `<span class="artifact ${ok ? 'ok' : 'no'}">${escapeHtml(label)} ${ok ? '✓' : '—'}</span>`;
}

function statusBadgeClass(status) {
  if (status === "FINISHED" || status === "success") return "ok";
  if (status === "FAILED" || status === "KILLED" || status === "failed") return "err";
  return "warn";
}

function renderModelStatus(data) {
  document.getElementById("taskYamlPill").textContent = "task.yaml: " + (data.task_yaml_exists ? "已生成" : "未生成");
  document.getElementById("modelContextPill").textContent = "manifest: " + (data.model_context_exists ? "已生成" : "未生成");
}

function renderMlflowStatus(state) {
  mlflowState = state;
  const badge = document.getElementById("mlflowConnBadge");
  const box = document.getElementById("modelStatusBox");
  const expPill = document.getElementById("mlflowExperimentPill");
  expPill.textContent = "experiment: " + escapeHtml(state.experiment_name || "visionops-detection");

  if (!state.available || !state.ok) {
    badge.className = "mlflow-status err";
    badge.textContent = "MLflow: 不可用";
    const a = state.local_artifacts || {};
    box.innerHTML = `${escapeHtml(state.message || "MLflow 暂不可用，请确认 http://localhost:5000 是否启动。")}
      <div class="artifact-row">
        ${artifactBadge('best.pt', a.best_pt)}
        ${artifactBadge('ONNX', a.onnx)}
        ${artifactBadge('RKNN', a.rknn)}
      </div>`;
    return;
  }

  badge.className = "mlflow-status ok";
  badge.textContent = "MLflow: 已连接";
  const run = state.latest_run;
  if (!run) {
    box.innerHTML = "已连接 MLflow，但当前实验还没有 run。";
    return;
  }

  const m = run.metrics || {};
  const p = run.params || {};
  const a = run.artifacts || state.local_artifacts || {};
  const deployText = run.deployable ? "可部署" : "暂不建议部署";

  box.innerHTML = `
    <div><b>最新 Run：</b>${escapeHtml(run.run_name || run.run_id || "-")}</div>
    <div class="model-summary-line">
      <b>状态：</b><span class="mlflow-status ${statusBadgeClass(run.status)}">${escapeHtml(run.status || "-")}</span>
      <b>创建：</b>${escapeHtml(run.created_at || "-")}
      <b>耗时：</b>${escapeHtml(run.duration || "-")}
    </div>
    <div class="model-mini-metrics">
      <div class="metric"><div class="label">mAP50</div><div class="num">${metricText(m.best_map50)}</div></div>
      <div class="metric"><div class="label">mAP50-95</div><div class="num">${metricText(m.best_map50_95)}</div></div>
    </div>
    <div class="status-pills">
      <span class="pill">batch_size: ${escapeHtml(p.batch_size || "-")}</span>
      <span class="pill">epochs: ${escapeHtml(p.epochs || "-")}</span>
      <span class="pill">imgsz: ${escapeHtml(p.imgsz || "-")}</span>
      <span class="pill">部署建议: ${deployText}</span>
    </div>
    <div class="artifact-row">
      ${artifactBadge('best.pt', a.best_pt)}
      ${artifactBadge('ONNX', a.onnx)}
      ${artifactBadge('RKNN', a.rknn)}
    </div>
    <div class="model-summary-line">
      <b>Precision：</b>${metricText(m.precision)}　
      <b>Recall：</b>${metricText(m.recall)}　
      <b>Run ID：</b>${escapeHtml(run.run_id || "-")}　
      <b>模型：</b>${escapeHtml(p.model || "-")}
    </div>
    <div class="btn-row"><button class="ghost" onclick="openLatestRun()">打开最新 Run</button></div>`;
}
async function refreshModelStatus() {
  const res = await fetch("/api/mlflow/latest-model-status?experiment=visionops-detection");
  const data = await res.json();
  renderMlflowStatus(data);
}

function openMlflow() {
  window.open((mlflowState && mlflowState.tracking_uri) || "http://localhost:5000", "_blank");
}

function openLatestRun() {
  const run = mlflowState && mlflowState.latest_run;
  if (run && run.url) window.open(run.url, "_blank");
  else openMlflow();
}

function openRunHistoryModal() {
  document.getElementById("runHistoryModal").style.display = "flex";
  refreshRunHistory();
}

function closeRunHistoryModal() {
  document.getElementById("runHistoryModal").style.display = "none";
}

function renderRunHistory(state) {
  const box = document.getElementById("runHistoryBox");
  if (!state.available || !state.ok) {
    box.innerHTML = escapeHtml(state.message || "MLflow 暂不可用，请确认 http://localhost:5000 是否启动。");
    return;
  }

  const runs = state.runs || [];
  if (!runs.length) {
    box.innerHTML = "当前实验还没有 run。";
    return;
  }

  box.innerHTML = `<table class="run-table">
    <thead><tr><th>时间</th><th>Run Name</th><th>状态</th><th>mAP50</th><th>mAP50-95</th><th>batch</th><th>耗时</th><th>操作</th></tr></thead>
    <tbody>` + runs.map(r => {
      const m = r.metrics || {};
      const p = r.params || {};
      return `<tr>
        <td>${escapeHtml(r.created_at || "-")}</td>
        <td><div class="run-name" title="${escapeHtml(r.run_name || r.run_id)}">${escapeHtml(r.run_name || r.run_id)}</div></td>
        <td><span class="mlflow-status ${statusBadgeClass(r.status)}">${escapeHtml(r.status || "-")}</span></td>
        <td>${metricText(m.best_map50)}</td>
        <td>${metricText(m.best_map50_95)}</td>
        <td>${escapeHtml(p.batch_size || "-")}</td>
        <td>${escapeHtml(r.duration || "-")}</td>
        <td><button class="ghost" onclick="window.open('${escapeHtml(r.url || "")}', '_blank')">查看</button></td>
      </tr>`;
    }).join("") + `</tbody></table>`;
}

async function refreshRunHistory() {
  const res = await fetch("/api/mlflow/runs?experiment=visionops-detection&limit=20");
  const data = await res.json();
  mlflowState = data;
  renderRunHistory(data);
  renderMlflowStatus(data);
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  renderStatus(data);
}

async function startJob(url) {
  const res = await fetch(url, { method: "POST" });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || "启动失败"); return; }
  currentJobId = data.job_id;
  document.getElementById("log").textContent = data.message + "\njob_id=" + currentJobId + "\n";
  if (timer) clearInterval(timer);
  timer = setInterval(fetchLogs, 1000);
}

async function fetchLogs() {
  if (!currentJobId) return;
  const res = await fetch(`/api/jobs/${currentJobId}/logs`);
  const data = await res.json();
  document.getElementById("log").textContent = data.logs || "";
  const logDiv = document.getElementById("log");
  logDiv.scrollTop = logDiv.scrollHeight;
  if (data.status && (data.status.status === "success" || data.status.status === "failed")) {
    clearInterval(timer); timer = null; refreshStatus(); refreshIncomingPackages();
  }
}

let incomingPackagesCache = [];
async function refreshIncomingPackages() {
  const res = await fetch("/api/incoming-packages");
  const data = await res.json();
  if (!res.ok) {
    document.getElementById("incomingPackages").innerHTML = `<div class="empty">${escapeHtml(data.detail || "读取上传包失败")}</div>`;
    return;
  }
  incomingPackagesCache = data.packages || [];
  renderIncomingPackages(incomingPackagesCache);
}

function renderIncomingPackages(packages) {
  const box = document.getElementById("incomingPackages");
  if (!packages.length) {
    box.innerHTML = `<div class="empty">当前没有待处理上传包。目录：data/incoming</div>`;
    updateSelectedSummary(); return;
  }
  box.innerHTML = packages.map((p, idx) => `
    <label class="package-item">
      <input type="checkbox" class="pkg-check" value="${escapeHtml(p.path)}" onchange="updateSelectedSummary()" />
      <div><div class="package-name">${escapeHtml(p.name)}</div><div class="package-meta">${escapeHtml(p.path)} ｜ ${escapeHtml(p.size_mb)} MB ｜ ${escapeHtml(p.mtime)}</div></div>
      ${idx === 0 ? '<span class="badge">最新</span>' : ''}
    </label>`).join("");
  updateSelectedSummary();
}

function getSelectedPackages() { return Array.from(document.querySelectorAll(".pkg-check:checked")).map(el => el.value); }
function updateSelectedSummary() {
  const selected = getSelectedPackages();
  const box = document.getElementById("selectedSummary");
  if (!box) return;
  if (!selected.length) box.textContent = "当前未选择上传包。";
  else if (selected.length === 1) box.textContent = `当前选择 1 个上传包，将按单包处理：${selected[0]}`;
  else box.textContent = `当前选择 ${selected.length} 个上传包，将自动合并处理。`;
}

async function startIngestSelected() {
  const packages = getSelectedPackages();
  if (!packages.length) { alert("请先选择至少一个上传压缩包"); return; }
  const confirmText = packages.length === 1 ? `确认处理这个上传包？
${packages[0]}` : `确认合并处理 ${packages.length} 个上传包？合并后只会生成一个 manifest。`;
  if (!confirm(confirmText)) return;
  const res = await fetch("/api/ingest", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({packages}) });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || "启动失败"); return; }
  currentJobId = data.job_id;
  document.getElementById("log").textContent = data.message + "\njob_id=" + currentJobId + "\n" + (data.cmd || "");
  if (timer) clearInterval(timer);
  timer = setInterval(fetchLogs, 1000);
}

function openVisionOpsAnnotator() { window.open('/annotator', '_blank'); }
async function openAnnotator() {
  const res = await fetch("/api/open-annotator", { method: "POST" });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || "打开失败"); return; }
  document.getElementById("log").textContent = data.message + "\n图片目录：" + data.images_dir + "\n命令：" + (data.cmd || "") + "\n" + data.tip;
}

function openPipelineModal() { document.getElementById("pipelineModal").style.display = "flex"; document.getElementById("taskTypeStep").style.display = "block"; document.getElementById("taskEditStep").style.display = "none"; }
function closePipelineModal() { document.getElementById("pipelineModal").style.display = "none"; }
function backToTaskTypeStep() { document.getElementById("taskTypeStep").style.display = "block"; document.getElementById("taskEditStep").style.display = "none"; }
async function prepareTaskYaml(taskType) {
  const res = await fetch("/api/prepare-task-yaml", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({task_type: taskType}) });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || "生成 task.yaml 失败"); return; }
  document.getElementById("taskYamlText").value = data.content || "";
  document.getElementById("taskHint").textContent = `任务类型：${data.task_type}；模板：${data.preset}；类别数：${data.num_classes}；类别：${(data.names || []).join(", ")}；来源：${data.data_yaml}`;
  document.getElementById("taskTypeStep").style.display = "none";
  document.getElementById("taskEditStep").style.display = "block";
  document.getElementById("log").textContent = data.message;
  refreshStatus();
}
async function saveAndStartPipeline() {
  const content = document.getElementById("taskYamlText").value;
  if (!confirm("确认保存当前 task.yaml 并开始训练流水线？")) return;
  const res = await fetch("/api/pipeline-confirmed", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({content}) });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || "启动失败"); return; }
  closePipelineModal(); currentJobId = data.job_id;
  document.getElementById("log").textContent = data.message + "\njob_id=" + currentJobId + "\n";
  if (timer) clearInterval(timer); timer = setInterval(fetchLogs, 1000);
}

function openDeployOtherModal() { document.getElementById("deployOtherModal").style.display = "flex"; }
function closeDeployOtherModal() { document.getElementById("deployOtherModal").style.display = "none"; }
function previewDeployOther() {
  const deviceId = document.getElementById("otherDeviceId").value.trim() || "rk3588-xxx";
  const host = document.getElementById("otherDeviceHost").value.trim() || "192.168.1.xxx";
  const user = document.getElementById("otherDeviceUser").value.trim() || "ubuntu";
  const port = document.getElementById("otherDevicePort").value.trim() || "22";
  const root = document.getElementById("otherDeviceRoot").value.trim() || "/opt/visionops";
  document.getElementById("deployOtherPreview").innerHTML = `设备参数预览：<br>device_id=${escapeHtml(deviceId)}<br>target=${escapeHtml(user)}@${escapeHtml(host)}:${escapeHtml(port)}<br>deploy_root=${escapeHtml(root)}<br><br>下一步建议把这些字段传给 edge/deploy/push.sh，例如通过命令行参数或环境变量完成多设备部署。`;
}

refreshStatus();
refreshIncomingPackages();
refreshModelStatus();
</script>
</body>
</html>
'''
