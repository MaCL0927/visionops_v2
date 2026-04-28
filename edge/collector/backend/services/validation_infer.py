#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from backend.config import (
    MODELS_DIR,
    VALIDATION_ENGINE_PATH,
    VALIDATION_INFER_HOST,
    VALIDATION_INFER_PORT,
    VALIDATION_INFER_TIMEOUT_SEC,
    VALIDATION_NPU_CORE,
    VALIDATION_TOPK,
    VALIDATION_WARMUP_RUNS,
)

logger = logging.getLogger("visionops.collector.validation")

_validation_process: Optional[subprocess.Popen] = None
_validation_log_file = None
_loaded_model_name: str = ""
_loaded_model_path: str = ""
_loaded_meta_path: str = ""
_loaded_task: str = ""


# ────────────────────────────────────────────────
# 模型与 YAML 读取
# ────────────────────────────────────────────────
def _list_from_names(value: Any) -> list:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        def key_fn(k: Any):
            s = str(k)
            return (0, int(s)) if s.isdigit() else (1, s)
        return [str(value[k]) for k in sorted(value.keys(), key=key_fn)]
    return []


def _safe_model_path(model_name: str) -> Path:
    name = Path(model_name or "").name
    if not name or name in {".", ".."}:
        raise ValueError("请先选择模型")
    if not name.endswith(".rknn"):
        raise ValueError("只能选择 .rknn 模型")

    models_dir = MODELS_DIR.resolve()
    path = (models_dir / name).resolve()
    if models_dir not in path.parents and path != models_dir:
        raise ValueError("非法模型路径")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"模型不存在: {name}")
    return path


def _meta_path_for_model(model_path: Path) -> Path:
    meta_path = model_path.with_suffix(".yaml")
    if not meta_path.exists() or not meta_path.is_file():
        raise FileNotFoundError(f"缺少同名模型配置文件: {meta_path.name}")
    return meta_path


def _read_model_meta(model_path: Path) -> Dict[str, Any]:
    meta_path = _meta_path_for_model(model_path)
    try:
        data = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise ValueError(f"读取模型配置失败: {meta_path.name}, {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"模型配置格式错误: {meta_path.name}")

    model_meta = data.get("model") if isinstance(data.get("model"), dict) else {}
    task = str(data.get("task") or model_meta.get("task") or "").strip().lower()
    if task in {"obb", "oriented_detection", "rotated_detection", "yolo_obb", "yolov8_obb"}:
        task = "obb_detection"
    if task not in {"classification", "detection", "obb_detection"}:
        raise ValueError(f"模型配置缺少有效 task: {meta_path.name}")

    input_size = data.get("input_size") or model_meta.get("input_size")
    if input_size is None:
        input_size = [224, 224] if task == "classification" else [640, 640]
    try:
        input_size = [int(input_size[0]), int(input_size[1])]
    except Exception as e:
        raise ValueError(f"模型配置 input_size 无效: {meta_path.name}") from e

    class_names = _list_from_names(data.get("class_names") or model_meta.get("class_names"))
    try:
        num_classes = int(data.get("num_classes") or model_meta.get("num_classes") or len(class_names))
    except Exception as e:
        raise ValueError(f"模型配置 num_classes 无效: {meta_path.name}") from e
    if num_classes <= 0:
        raise ValueError(f"模型配置 num_classes 必须大于 0: {meta_path.name}")
    if not class_names:
        class_names = [str(i) for i in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(f"class_names 数量与 num_classes 不一致: {meta_path.name}")

    topk = int(data.get("topk") or model_meta.get("topk") or min(max(num_classes, 1), int(VALIDATION_TOPK)))
    conf_threshold = float(data.get("conf_threshold") or model_meta.get("conf_threshold") or 0.25)
    nms_threshold = float(data.get("nms_threshold") or model_meta.get("nms_threshold") or 0.45)

    return {
        "path": str(meta_path.resolve()),
        "raw": data,
        "task": task,
        "input_size": input_size,
        "num_classes": num_classes,
        "class_names": class_names,
        "topk": topk,
        "conf_threshold": conf_threshold,
        "nms_threshold": nms_threshold,
    }


def _resolve_engine_path() -> Path:
    candidates = [
        Path(VALIDATION_ENGINE_PATH).expanduser(),
        Path("/opt/visionops/edge/inference/engine.py"),
        Path(__file__).resolve().parents[3] / "inference" / "engine.py",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0]


# ────────────────────────────────────────────────
# 验证推理服务管理
# ────────────────────────────────────────────────
def _validation_port() -> int:
    return int(VALIDATION_INFER_PORT)


def _health_url() -> str:
    return f"http://{VALIDATION_INFER_HOST}:{_validation_port()}/health"


def _infer_url() -> str:
    return f"http://{VALIDATION_INFER_HOST}:{_validation_port()}/infer"


def _json_get(url: str, timeout: float = 2.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_abs_path(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(Path(str(value)).expanduser().resolve())
    except Exception:
        return str(value)


def _health_matches_model(health: Dict[str, Any], model_path: Path, meta_path: str, task: str) -> bool:
    """严格判断当前端口上的 engine 是否就是目标模型。"""
    if not isinstance(health, dict):
        return False
    if health.get("status") != "ok":
        return False
    if str(health.get("task", "")).lower() != str(task).lower():
        return False

    health_model = _normalize_abs_path(health.get("model_path"))
    target_model = _normalize_abs_path(model_path)
    if health_model != target_model:
        return False

    health_meta = _normalize_abs_path(health.get("class_names_file"))
    target_meta = _normalize_abs_path(meta_path)
    if health_meta != target_meta:
        return False

    return True


def _is_process_alive() -> bool:
    return _validation_process is not None and _validation_process.poll() is None


def _close_log_file() -> None:
    global _validation_log_file
    if _validation_log_file is not None:
        try:
            _validation_log_file.close()
        except Exception:
            pass
    _validation_log_file = None


def _stop_validation_process() -> None:
    """停止当前 collector 进程持有的验证 engine。"""
    global _validation_process, _loaded_model_name, _loaded_model_path, _loaded_meta_path, _loaded_task
    if _validation_process is not None and _validation_process.poll() is None:
        _validation_process.terminate()
        try:
            _validation_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _validation_process.kill()
            _validation_process.wait(timeout=3)
    _validation_process = None
    _loaded_model_name = ""
    _loaded_model_path = ""
    _loaded_meta_path = ""
    _loaded_task = ""
    _close_log_file()


def _pid_cmdline(pid: str) -> str:
    try:
        return subprocess.check_output(["ps", "-p", str(pid), "-o", "args="], text=True).strip()
    except Exception:
        return ""


def _find_pids_on_validation_port() -> List[int]:
    """查找动态验证端口上的监听进程 PID。端口来自 VALIDATION_INFER_PORT，不写死 8082。"""
    port = _validation_port()
    pids: List[int] = []
    commands = [
        ["bash", "-lc", f"ss -lntp 2>/dev/null | grep ':{port} ' || true"],
        ["bash", "-lc", f"netstat -lntp 2>/dev/null | grep ':{port} ' || true"],
    ]
    for cmd in commands:
        try:
            out = subprocess.check_output(cmd, text=True)
        except Exception:
            out = ""
        for line in out.splitlines():
            # ss: users:(("python",pid=123,fd=11))
            for m in re.finditer(r"pid=(\d+)", line):
                pid = int(m.group(1))
                if pid not in pids:
                    pids.append(pid)
            # netstat: 0.0.0.0:8082 ... LISTEN 123/python
            for m in re.finditer(r"\s(\d+)/(?:python|python3|[^\s]+)", line):
                pid = int(m.group(1))
                if pid not in pids:
                    pids.append(pid)
        if pids:
            break
    return pids


def _is_validation_engine_cmdline(cmdline: str) -> bool:
    if not cmdline:
        return False
    port = str(_validation_port())
    # 只清理验证 engine.py，避免误杀其他服务。
    if "engine.py" not in cmdline:
        return False
    if f"--port {port}" in cmdline or f"--port={port}" in cmdline:
        return True
    # 兼容极少数通过环境变量 PORT 启动的情况。
    if f"PORT={port}" in cmdline:
        return True
    return False


def _kill_validation_engine_on_port() -> None:
    """清理动态验证端口上的旧 engine.py。

    这个函数只会杀 command line 中包含 engine.py 且使用 VALIDATION_INFER_PORT 的进程，
    不会因为端口号写死而误杀 8082，也不会误杀生产 8080。
    """
    port = _validation_port()
    killed = []
    for pid in _find_pids_on_validation_port():
        # 不要重复 kill 当前 _validation_process；交给 _stop_validation_process 处理也可以，但这里兼容残留进程。
        cmdline = _pid_cmdline(str(pid))
        if not _is_validation_engine_cmdline(cmdline):
            logger.warning("验证端口 %s 被非验证 engine 进程占用，pid=%s, cmd=%s", port, pid, cmdline)
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            logger.warning("无权限停止验证端口上的旧进程，pid=%s", pid)

    if killed:
        deadline = time.time() + 3.0
        while time.time() < deadline:
            alive = [pid for pid in killed if Path(f"/proc/{pid}").exists()]
            if not alive:
                break
            time.sleep(0.2)
        for pid in killed:
            if Path(f"/proc/{pid}").exists():
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass


def _port_has_unrelated_process() -> Optional[str]:
    port = _validation_port()
    pids = _find_pids_on_validation_port()
    for pid in pids:
        cmdline = _pid_cmdline(str(pid))
        if not _is_validation_engine_cmdline(cmdline):
            return f"验证端口 {port} 被其他进程占用: pid={pid}, cmd={cmdline}"
    return None


def _wait_validation_port_free(timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _find_pids_on_validation_port():
            return True
        time.sleep(0.2)
    return not _find_pids_on_validation_port()


def _read_last_log_lines(max_lines: int = 80) -> str:
    path = Path(f"/tmp/visionops_validation_engine_{_validation_port()}.log")
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception:
        return ""



def _resolve_switch_model_script() -> Path:
    candidates = [
        Path("/opt/visionops/edge/deploy/switch_model.sh"),
        Path(__file__).resolve().parents[3] / "deploy" / "switch_model.sh",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0]


def _switch_model_via_system_service(model_path: Path, meta_path: str) -> Dict[str, Any]:
    """通过统一脚本切换当前 visionops-inference 服务，避免多个 engine.py 抢占同一端口。"""
    script = _resolve_switch_model_script()
    if not script.exists():
        raise FileNotFoundError(f"未找到模型切换脚本: {script}")

    cmd = [
        "bash",
        str(script),
        str(model_path),
        str(meta_path),
        str(_validation_port()),
        "visionops-inference",
    ]
    logger.info("切换验证模型: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(30.0, float(VALIDATION_INFER_TIMEOUT_SEC) + 20.0),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "切换模型失败。请确认板端当前用户具备免密 sudo，且 stop_inference.sh/switch_model.sh 已同步。\n"
            f"命令: {' '.join(cmd)}\n"
            f"输出:\n{result.stdout}"
        )
    health = _json_get(_health_url(), timeout=2.0)
    return {"health": health, "stdout": result.stdout}

def ensure_validation_engine(model_name: str) -> Dict[str, Any]:
    global _validation_process, _validation_log_file, _loaded_model_name, _loaded_model_path, _loaded_meta_path, _loaded_task

    model_path = _safe_model_path(model_name)
    meta = _read_model_meta(model_path)
    meta_path = meta["path"]
    task = meta["task"]

    # 1) 如果当前端口已经是目标模型，则直接复用。
    try:
        health = _json_get(_health_url(), timeout=1.0)
        if _health_matches_model(health, model_path, meta_path, task):
            _loaded_model_name = model_path.name
            _loaded_model_path = str(model_path)
            _loaded_meta_path = meta_path
            _loaded_task = task
            return {"reused": True, "health": health, "model_path": str(model_path), "meta": meta}
        logger.info(
            "验证端口已有服务但不是目标模型，将通过 switch_model.sh 切换。current_model=%s, current_task=%s, target_model=%s, target_task=%s",
            health.get("model_path"), health.get("task"), str(model_path), task,
        )
    except Exception:
        # 端口未启动服务或健康检查失败，继续通过 switch_model.sh 切换。
        pass

    # 2) 统一通过 systemd 服务切换模型，不再由 collector 直接启动第二个 engine.py。
    #    这可以避免检测/分类来回切换时多个 engine.py 抢占同一端口。
    _stop_validation_process()
    switched = _switch_model_via_system_service(model_path, meta_path)
    health = switched.get("health", {})
    if not _health_matches_model(health, model_path, meta_path, task):
        raise RuntimeError(
            "切换后健康检查与目标模型不一致："
            f"health.model_path={health.get('model_path')}, "
            f"health.class_names_file={health.get('class_names_file')}, "
            f"health.task={health.get('task')}, "
            f"target_model={model_path}, target_meta={meta_path}, target_task={task}"
        )

    _loaded_model_name = model_path.name
    _loaded_model_path = str(model_path)
    _loaded_meta_path = meta_path
    _loaded_task = task
    _validation_process = None
    return {"reused": False, "health": health, "model_path": str(model_path), "meta": meta}


# ────────────────────────────────────────────────
# 推理请求与结果包装
# ────────────────────────────────────────────────
def _post_multipart_image(image_path: Path) -> Dict[str, Any]:
    boundary = f"----VisionOpsBoundary{int(time.time() * 1000)}"
    content = image_path.read_bytes()
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + content + footer

    req = urllib.request.Request(
        _infer_url(),
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=float(VALIDATION_INFER_TIMEOUT_SEC)) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"推理服务返回错误: {e.code} {detail}") from e


def _normalize_classification_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    pred = raw.get("prediction") or raw.get("top1")
    if pred is None and isinstance(raw.get("predictions"), list) and raw["predictions"]:
        pred = raw["predictions"][0]
    if pred is None and isinstance(raw.get("topk"), list) and raw["topk"]:
        pred = raw["topk"][0]

    if pred is None:
        return {"class_id": None, "class_name": "未识别", "confidence": None, "confidence_percent": "--"}

    cls_id = pred.get("class_id")
    cls_name = pred.get("class_name") or pred.get("class") or pred.get("label") or str(cls_id)
    conf = pred.get("confidence")
    if conf is None:
        conf = pred.get("score")
    try:
        conf_float = float(conf)
    except Exception:
        conf_float = None

    return {
        "class_id": cls_id,
        "class_name": str(cls_name),
        "confidence": conf_float,
        "confidence_percent": f"{conf_float * 100:.1f}%" if conf_float is not None else "--",
    }


def _summarize_detection(raw: Dict[str, Any]) -> Dict[str, Any]:
    preds = raw.get("predictions") or []
    counts: Dict[str, int] = {}
    for p in preds:
        name = str(p.get("class_name") or p.get("class") or p.get("class_id") or "未知")
        counts[name] = counts.get(name, 0) + 1
    return {"count": len(preds), "class_counts": counts}


def infer_image_with_model(model_name: str, image_path: Path) -> Dict[str, Any]:
    engine_info = ensure_validation_engine(model_name)
    meta = engine_info.get("meta", {})
    raw = _post_multipart_image(image_path)
    task = raw.get("task") or meta.get("task") or "unknown"

    base = {
        "ok": True,
        "task": task,
        "model_name": Path(model_name).name,
        "image_name": image_path.name,
        "latency_ms": raw.get("latency_ms"),
        "engine": {
            "port": _validation_port(),
            "reused": engine_info.get("reused", False),
            "model_path": engine_info.get("model_path"),
            "meta_path": meta.get("path"),
        },
        "model_meta": {
            "task": meta.get("task"),
            "input_size": meta.get("input_size"),
            "num_classes": meta.get("num_classes"),
            "class_names": meta.get("class_names"),
        },
        "raw": raw,
    }

    if task == "classification":
        base["result"] = _normalize_classification_result(raw)
        base["topk"] = raw.get("topk", [])
    elif task in {"detection", "obb_detection"}:
        base["predictions"] = raw.get("predictions", [])
        base["detection"] = _summarize_detection(raw)
        task_text = "旋转框检测" if task == "obb_detection" else "检测"
        base["result"] = {
            "class_name": f"{task_text}到 {base['detection']['count']} 个目标",
            "confidence": None,
            "confidence_percent": "--",
        }
        base["topk"] = []
    else:
        base["result"] = {"class_name": "未知任务", "confidence": None, "confidence_percent": "--"}
        base["topk"] = []
    return base


# 保留旧函数名，兼容 collector.py 现有调用。
def classify_image_with_model(model_name: str, image_path: Path) -> Dict[str, Any]:
    return infer_image_with_model(model_name, image_path)
