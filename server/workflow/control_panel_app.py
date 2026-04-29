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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_CONTEXT_PATH = DATA_DIR / "model_context" / "manifest.json"
WORKFLOW_RUNS_DIR = DATA_DIR / "workflow_runs"
TASK_YAML_PATH = PROJECT_ROOT / "pipeline" / "configs" / "task.yaml"
PRESETS_DIR = PROJECT_ROOT / "pipeline" / "configs" / "presets"

# X-AnyLabeling 启动命令。
# 你的实际命令是 xanylabeling，且安装在 x-anylabeling-cu12 环境中。
X_ANYLABELING_CMD = os.environ.get("X_ANYLABELING_CMD", "xanylabeling")
X_ANYLABELING_ENV = os.environ.get("X_ANYLABELING_ENV", "x-anylabeling-cu12")


app = FastAPI(title="VisionOps Workflow Control Panel")


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

    incoming = DATA_DIR / "incoming"
    packages = []
    if incoming.exists():
        packages = [str(p.relative_to(PROJECT_ROOT)) for p in sorted(incoming.glob("*.tar.gz"))]

    return {
        "project_root": str(PROJECT_ROOT),
        "incoming_dir": "data/incoming",
        "pending_packages": packages,
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


@app.post("/api/ingest")
def api_ingest() -> dict[str, Any]:
    job_id = jobs.start("ingest-collected-force", ["make", "ingest-collected-force"])
    return {"job_id": job_id, "message": "已开始强制处理上传数据集"}


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
    .modal h2 { margin: 0 0 10px; font-size: 20px; }
    .modal p { color:#64748b; margin:6px 0 14px; line-height:1.55; }
    .btn-row { display:flex; gap:10px; flex-wrap:wrap; margin: 12px 0; }
    textarea { width:100%; height:440px; border:1px solid #d1d5db; border-radius:12px; padding:14px; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size:13px; line-height:1.5; box-sizing:border-box; }
    .hint { background:#f8fafc; border:1px solid #e5e7eb; border-radius:12px; padding:12px; color:#475569; font-size:14px; }
  </style>
</head>
<body>
  <div class="page">
    <h1>VisionOps 数据闭环控制台</h1>
    <div class="subtitle">接收数据 → 半自动标注 → 审核入库 → 训练流水线 → 部署模型</div>

    <div class="status">
      <h2>运行日志</h2>
      <div id="log" class="log">暂无任务日志。</div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>1. 处理上传数据集</h2>
        <p>扫描 data/incoming 下的 tar.gz，强制解压到 data/raw_collected/&lt;batch_id&gt;，并同步 manifest。</p>
        <button onclick="startJob('/api/ingest')">处理上传数据集</button>
      </div>

      <div class="card">
        <h2>2. 打开半自动标注</h2>
        <p>自动定位当前 batch，并用 X-AnyLabeling 打开 all_images 目录。</p>
        <button class="secondary" onclick="openAnnotator()">打开半自动标注</button>
      </div>

      <div class="card">
        <h2>3. 确认审核完成</h2>
        <p>读取 all_images 图片和 labels 目录下的 YOLO txt，整理到 raw_detection 或 raw_obb。</p>
        <button class="warning" onclick="startJob('/api/accept-reviewed')">确认审核完成</button>
      </div>

      <div class="card">
        <h2>4. 开始训练流水线</h2>
        <p>先选择任务类型，系统从 preset 生成 task.yaml 并自动同步 data.yaml 中的类别。</p>
        <button class="success" onclick="openPipelineModal()">开始训练流水线</button>
      </div>

      <div class="card">
        <h2>5. 部署模型</h2>
        <p>执行 make deploy，把最新模型部署到边缘端设备，并进行健康检查。</p>
        <button onclick="startJob('/api/deploy')">部署模型</button>
      </div>

      <div class="card">
        <h2>当前状态</h2>
        <p>显示当前批次、待处理上传包、图片数量和标签数量。</p>
        <button onclick="refreshStatus()">刷新状态</button>
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

<script>
let currentJobId = null;
let timer = null;

function renderStatus(data) {
  const rows = [
    ["项目根目录", data.project_root],
    ["上传包目录", data.incoming_dir],
    ["待处理上传包", data.pending_packages.join(", ") || "无"],
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
    <div class="key">${k}</div>
    <div class="value">${v}</div>
  `).join("");
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  renderStatus(data);
}

async function startJob(url) {
  const res = await fetch(url, { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "启动失败");
    return;
  }
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
    clearInterval(timer);
    timer = null;
    refreshStatus();
  }
}

async function openAnnotator() {
  const res = await fetch("/api/open-annotator", { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "打开失败");
    return;
  }
  document.getElementById("log").textContent =
    data.message + "\n图片目录：" + data.images_dir + "\n命令：" + (data.cmd || "") + "\n" + data.tip;
}

function openPipelineModal() {
  document.getElementById("pipelineModal").style.display = "flex";
  document.getElementById("taskTypeStep").style.display = "block";
  document.getElementById("taskEditStep").style.display = "none";
}

function closePipelineModal() {
  document.getElementById("pipelineModal").style.display = "none";
}

function backToTaskTypeStep() {
  document.getElementById("taskTypeStep").style.display = "block";
  document.getElementById("taskEditStep").style.display = "none";
}

async function prepareTaskYaml(taskType) {
  const res = await fetch("/api/prepare-task-yaml", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({task_type: taskType})
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "生成 task.yaml 失败");
    return;
  }

  document.getElementById("taskYamlText").value = data.content || "";
  document.getElementById("taskHint").textContent =
    `任务类型：${data.task_type}；模板：${data.preset}；类别数：${data.num_classes}；类别：${(data.names || []).join(", ")}；来源：${data.data_yaml}`;
  document.getElementById("taskTypeStep").style.display = "none";
  document.getElementById("taskEditStep").style.display = "block";
  document.getElementById("log").textContent = data.message;
  refreshStatus();
}

async function saveAndStartPipeline() {
  const content = document.getElementById("taskYamlText").value;
  const ok = confirm("确认保存当前 task.yaml 并开始训练流水线？");
  if (!ok) return;

  const res = await fetch("/api/pipeline-confirmed", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({content})
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "启动失败");
    return;
  }

  closePipelineModal();
  currentJobId = data.job_id;
  document.getElementById("log").textContent = data.message + "\njob_id=" + currentJobId + "\n";
  if (timer) clearInterval(timer);
  timer = setInterval(fetchLogs, 1000);
}

refreshStatus();
</script>
</body>
</html>
'''
