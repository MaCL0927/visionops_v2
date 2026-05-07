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

from backend.services.settings_store import (
    get_algorithm_effective_config,
    get_effective_models_dir,
    write_runtime_algorithm_env,
)

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
_loaded_algorithm_signature: str = ""
_loaded_pipeline_config: str = ""
_force_algorithm_reload: bool = False


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

    models_dir = get_effective_models_dir().resolve()
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


def _safe_pipeline_config(model_name: str) -> Path:
    """校验并返回 ROI Classification bundle 的 pipeline.yaml。"""
    name = Path(model_name or "").name
    if not name or name in {".", ".."}:
        raise ValueError("请先选择模型")
    if name.endswith(".rknn"):
        raise ValueError("当前名称是单模型 RKNN，不是 ROI 双模型 bundle")

    models_dir = get_effective_models_dir().resolve()
    bundle_dir = (models_dir / name).resolve()
    if models_dir not in bundle_dir.parents and bundle_dir != models_dir:
        raise ValueError("非法模型路径")
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise FileNotFoundError(f"ROI 双模型 bundle 不存在: {name}")
    pipeline_config = bundle_dir / "pipeline.yaml"
    if not pipeline_config.exists() or not pipeline_config.is_file():
        raise FileNotFoundError(f"缺少 ROI 双模型配置文件: {name}/pipeline.yaml")
    return pipeline_config


def _resolve_bundle_path(bundle_dir: Path, value: Any) -> Path:
    p = Path(str(value or ""))
    if p.is_absolute():
        return p.resolve()
    return (bundle_dir / p).resolve()


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise ValueError(f"读取 YAML 失败: {path}, {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"YAML 格式错误: {path}")
    return data


def _read_roi_pipeline_meta(pipeline_config: Path) -> Dict[str, Any]:
    data = _read_yaml(pipeline_config)
    pipeline_type = str(data.get("pipeline_type") or data.get("task") or "").strip().lower()
    if pipeline_type != "roi_classification":
        raise ValueError(f"不是 ROI 分类双模型配置: {pipeline_config}")

    bundle_dir = pipeline_config.parent
    stage1 = data.get("stage1") if isinstance(data.get("stage1"), dict) else {}
    stage2 = data.get("stage2") if isinstance(data.get("stage2"), dict) else {}
    roi_cfg = data.get("roi") if isinstance(data.get("roi"), dict) else {}

    detector_path = _resolve_bundle_path(bundle_dir, stage1.get("model_path") or "detector.rknn")
    classifier_path = _resolve_bundle_path(bundle_dir, stage2.get("model_path") or "classifier.rknn")
    if not detector_path.exists():
        raise FileNotFoundError(f"ROI bundle 缺少检测模型: {detector_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"ROI bundle 缺少分类模型: {classifier_path}")

    detector_names = _list_from_names(stage1.get("class_names") or stage1.get("names"))
    classifier_names = _list_from_names(stage2.get("class_names") or stage2.get("names"))
    try:
        detector_num = int(stage1.get("num_classes") or len(detector_names))
    except Exception:
        detector_num = len(detector_names)
    try:
        classifier_num = int(stage2.get("num_classes") or len(classifier_names))
    except Exception:
        classifier_num = len(classifier_names)
    if not detector_names and detector_num > 0:
        detector_names = [str(i) for i in range(detector_num)]
    if not classifier_names and classifier_num > 0:
        classifier_names = [str(i) for i in range(classifier_num)]

    return {
        "path": str(pipeline_config.resolve()),
        "raw": data,
        "task": "roi_classification",
        "pipeline_config": str(pipeline_config.resolve()),
        "pipeline_name": str(data.get("pipeline_name") or bundle_dir.name),
        "bundle_dir": str(bundle_dir.resolve()),
        "input_size": stage2.get("input_size") or [224, 224],
        "num_classes": classifier_num,
        "class_names": classifier_names,
        "algorithm": {},
        "algorithm_signature": "roi_classification:" + str(pipeline_config.resolve()),
        "detector": {
            "model_path": str(detector_path),
            "input_size": stage1.get("input_size") or [640, 640],
            "num_classes": detector_num,
            "class_names": detector_names,
            "conf_threshold": stage1.get("conf_threshold"),
            "nms_threshold": stage1.get("nms_threshold"),
            "target_class_id": stage1.get("target_class_id"),
            "target_class_name": stage1.get("target_class_name"),
        },
        "classifier": {
            "model_path": str(classifier_path),
            "input_size": stage2.get("input_size") or [224, 224],
            "num_classes": classifier_num,
            "class_names": classifier_names,
            "topk": stage2.get("topk"),
        },
        "roi": roi_cfg,
    }


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
    if task in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg", "yolov8_seg", "mask_segmentation"}:
        task = "segmentation"
    if task not in {"classification", "detection", "obb_detection", "segmentation"}:
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

    # v2.2：模型 YAML 给出默认值，runtime_overrides.yaml 中的算法设置作为现场覆盖。
    meta_defaults = dict(data)
    meta_defaults["task"] = task
    runtime_algo = get_algorithm_effective_config(task=task, model_meta=meta_defaults)

    topk = int(runtime_algo.get("topk") or data.get("topk") or model_meta.get("topk") or min(max(num_classes, 1), int(VALIDATION_TOPK)))
    conf_threshold = float(runtime_algo.get("conf_threshold") if runtime_algo.get("conf_threshold") is not None else (data.get("conf_threshold") or model_meta.get("conf_threshold") or 0.25))
    nms_threshold = float(runtime_algo.get("nms_threshold") if runtime_algo.get("nms_threshold") is not None else (data.get("nms_threshold") or model_meta.get("nms_threshold") or 0.45))
    mask_threshold = float(runtime_algo.get("mask_threshold") if runtime_algo.get("mask_threshold") is not None else (data.get("mask_threshold") or model_meta.get("mask_threshold") or 0.5))

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
        "mask_threshold": mask_threshold,
        "algorithm": runtime_algo,
        "algorithm_signature": runtime_algo.get("signature", ""),
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


def _health_matches_model(health: Dict[str, Any], model_path: Path, meta_path: str, meta: Dict[str, Any], ignore_force: bool = False) -> bool:
    """严格判断当前端口上的 engine 是否就是目标模型和目标算法参数。"""
    global _force_algorithm_reload
    if _force_algorithm_reload and not ignore_force:
        return False
    if not isinstance(health, dict):
        return False
    if health.get("status") != "ok":
        return False
    task = str(meta.get("task") or "").lower()
    if str(health.get("task", "")).lower() != task:
        return False

    health_model = _normalize_abs_path(health.get("model_path"))
    target_model = _normalize_abs_path(model_path)
    if health_model != target_model:
        return False

    health_meta = _normalize_abs_path(health.get("class_names_file"))
    target_meta = _normalize_abs_path(meta_path)
    if health_meta != target_meta:
        return False

    expected = meta.get("algorithm", {}) if isinstance(meta.get("algorithm"), dict) else {}

    def _same_float(key: str, expected_value: Any) -> bool:
        if key not in health or expected_value is None:
            return True
        try:
            return abs(float(health.get(key)) - float(expected_value)) < 1e-6
        except Exception:
            return True

    def _same_int(key: str, expected_value: Any) -> bool:
        if key not in health or expected_value is None:
            return True
        try:
            return int(health.get(key)) == int(expected_value)
        except Exception:
            return True

    if not _same_float("conf_threshold", expected.get("conf_threshold")):
        return False
    if not _same_float("nms_threshold", expected.get("nms_threshold")):
        return False
    if not _same_float("mask_threshold", expected.get("mask_threshold")):
        return False
    if not _same_int("topk", expected.get("topk")):
        return False

    # 如果 collector 本进程已经加载过 engine，则参数签名必须一致。
    expected_sig = str(meta.get("algorithm_signature") or "")
    if expected_sig and _loaded_algorithm_signature and expected_sig != _loaded_algorithm_signature:
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


def invalidate_algorithm_runtime(reason: str = "") -> None:
    """设置界面修改算法参数后，通知下次推理不要复用旧参数。"""
    global _loaded_algorithm_signature, _force_algorithm_reload
    _loaded_algorithm_signature = ""
    _force_algorithm_reload = True
    logger.info("算法运行时参数已失效，下次推理将重新检查/加载: %s", reason)


def _stop_validation_process() -> None:
    """停止当前 collector 进程持有的验证 engine。"""
    global _validation_process, _loaded_model_name, _loaded_model_path, _loaded_meta_path, _loaded_task, _loaded_algorithm_signature, _loaded_pipeline_config, _force_algorithm_reload
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
    _loaded_algorithm_signature = ""
    _loaded_pipeline_config = ""
    _force_algorithm_reload = False
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
    # 只清理验证 engine.py / pipeline_engine.py，避免误杀其他服务。
    if "engine.py" not in cmdline and "pipeline_engine.py" not in cmdline:
        return False
    if f"--port {port}" in cmdline or f"--port={port}" in cmdline:
        return True
    # 兼容 systemd 环境变量 PORT=8082 启动。
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


def _switch_model_via_system_service(model_path: Path, meta_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """通过统一脚本切换当前 visionops-inference 服务，避免多个 engine.py 抢占同一端口。"""
    script = _resolve_switch_model_script()
    if not script.exists():
        raise FileNotFoundError(f"未找到模型切换脚本: {script}")

    # v2.2：先写 runtime_algorithm.env，switch_model.sh 会按模型 task 选择对应阈值/TopK。
    try:
        write_runtime_algorithm_env()
    except Exception as exc:
        logger.warning("写 runtime_algorithm.env 失败，继续使用脚本默认参数: %s", exc)

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

def _resolve_pipeline_engine_path() -> Path:
    candidates = [
        Path("/opt/visionops/edge/inference/pipeline_engine.py"),
        Path(__file__).resolve().parents[3] / "inference" / "pipeline_engine.py",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0]


def _install_text_with_sudo(content: str, target: str, mode: str = "644") -> None:
    tmp = Path(f"/tmp/visionops_collector_{Path(target).name}_{int(time.time()*1000)}")
    tmp.write_text(content, encoding="utf-8")
    try:
        subprocess.run(["sudo", "-n", "install", "-m", mode, str(tmp), target], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def _health_matches_roi_pipeline(health: Dict[str, Any], pipeline_config: Path, ignore_force: bool = False) -> bool:
    global _force_algorithm_reload
    if _force_algorithm_reload and not ignore_force:
        return False
    if not isinstance(health, dict) or health.get("status") != "ok":
        return False
    if str(health.get("task", "")).lower() != "roi_classification":
        return False
    health_pipeline = _normalize_abs_path(health.get("pipeline_config"))
    target_pipeline = _normalize_abs_path(pipeline_config)
    return bool(health_pipeline and health_pipeline == target_pipeline)


def _switch_roi_pipeline_via_system_service(pipeline_config: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """切换到 ROI Classification pipeline_engine.py。"""
    pipeline_engine = _resolve_pipeline_engine_path()
    if not pipeline_engine.exists():
        raise FileNotFoundError(f"未找到 ROI 双模型推理入口: {pipeline_engine}")

    port = _validation_port()
    metrics_port = int(os.getenv("METRICS_PORT", "9091"))
    npu_core = str(VALIDATION_NPU_CORE or "auto")

    # 明确清理 8082 上旧 engine.py / pipeline_engine.py，避免模型切换时端口被旧进程占用。
    _stop_validation_process()
    subprocess.run(["sudo", "-n", "systemctl", "stop", "visionops-inference"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _kill_validation_engine_on_port()
    if not _wait_validation_port_free(timeout=5.0):
        raise RuntimeError(f"验证端口 {port} 仍被占用，无法启动 ROI 双模型推理服务")

    env_content = f"""# Auto generated by collector validation_infer.py for roi_classification
TASK=roi_classification
VISIONOPS_TASK=roi_classification
PIPELINE_CONFIG={pipeline_config}
INFERENCE_URL=http://localhost:{port}
NPU_CORE={npu_core}
PORT={port}
METRICS_PORT={metrics_port}
WARMUP_RUNS={int(VALIDATION_WARMUP_RUNS)}
"""
    service_content = f"""[Unit]
Description=VisionOps RK3588 ROI Classification Pipeline Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/visionops
EnvironmentFile=/opt/visionops/.env
ExecStart=/opt/visionops/venv/bin/python {pipeline_engine} --pipeline-config ${{PIPELINE_CONFIG}} --host 0.0.0.0 --port ${{PORT}} --metrics-port ${{METRICS_PORT}}
Restart=always
RestartSec=5
TimeoutStartSec=30
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-inference

[Install]
WantedBy=multi-user.target
"""
    subprocess.run(["sudo", "-n", "mkdir", "-p", "/opt/visionops/edge/runtime"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _install_text_with_sudo(env_content, "/opt/visionops/.env")
    _install_text_with_sudo(env_content, "/opt/visionops/edge/runtime/edge.env")
    _install_text_with_sudo(service_content, "/etc/systemd/system/visionops-inference.service")

    for cmd in [
        ["sudo", "-n", "systemctl", "daemon-reload"],
        ["sudo", "-n", "systemctl", "enable", "visionops-inference"],
        ["sudo", "-n", "systemctl", "restart", "visionops-inference"],
    ]:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    deadline = time.time() + max(20.0, float(VALIDATION_INFER_TIMEOUT_SEC) + 5.0)
    last_err = ""
    while time.time() < deadline:
        try:
            health = _json_get(_health_url(), timeout=2.0)
            if _health_matches_roi_pipeline(health, pipeline_config, ignore_force=True):
                return {"health": health, "stdout": "systemd roi_classification service restarted"}
            last_err = json.dumps(health, ensure_ascii=False)
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.5)

    logs = ""
    try:
        logs = subprocess.check_output(["sudo", "-n", "journalctl", "-u", "visionops-inference", "-n", "80", "--no-pager"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        logs = _read_last_log_lines()
    raise RuntimeError(f"ROI 双模型服务启动后健康检查失败: {last_err}\n最近日志:\n{logs}")

def ensure_validation_engine(model_name: str) -> Dict[str, Any]:
    global _validation_process, _validation_log_file, _loaded_model_name, _loaded_model_path, _loaded_meta_path, _loaded_task, _loaded_algorithm_signature, _loaded_pipeline_config, _force_algorithm_reload

    name = Path(model_name or "").name

    # ROI Classification 双模型 bundle：模型选择界面只传 bundle 目录名。
    if name and not name.endswith(".rknn"):
        pipeline_config = _safe_pipeline_config(name)
        meta = _read_roi_pipeline_meta(pipeline_config)
        task = "roi_classification"

        try:
            health = _json_get(_health_url(), timeout=1.0)
            if _health_matches_roi_pipeline(health, pipeline_config):
                _loaded_model_name = name
                _loaded_model_path = str(pipeline_config.parent)
                _loaded_meta_path = str(pipeline_config)
                _loaded_task = task
                _loaded_algorithm_signature = str(meta.get("algorithm_signature") or "")
                _loaded_pipeline_config = str(pipeline_config)
                _force_algorithm_reload = False
                return {"reused": True, "health": health, "model_path": str(pipeline_config.parent), "meta": meta}
            logger.info(
                "验证端口已有服务但不是目标 ROI pipeline，将重启 pipeline_engine。current_task=%s, current_pipeline=%s, target_pipeline=%s",
                health.get("task"), health.get("pipeline_config"), str(pipeline_config),
            )
        except Exception:
            pass

        switched = _switch_roi_pipeline_via_system_service(pipeline_config, meta)
        health = switched.get("health", {})
        if not _health_matches_roi_pipeline(health, pipeline_config, ignore_force=True):
            raise RuntimeError(
                "切换后健康检查与目标 ROI pipeline 不一致："
                f"health.task={health.get('task')}, health.pipeline_config={health.get('pipeline_config')}, "
                f"target_pipeline={pipeline_config}"
            )

        _loaded_model_name = name
        _loaded_model_path = str(pipeline_config.parent)
        _loaded_meta_path = str(pipeline_config)
        _loaded_task = task
        _loaded_algorithm_signature = str(meta.get("algorithm_signature") or "")
        _loaded_pipeline_config = str(pipeline_config)
        _force_algorithm_reload = False
        _validation_process = None
        return {"reused": False, "health": health, "model_path": str(pipeline_config.parent), "meta": meta}

    # 单 RKNN 模型：保持原有逻辑。
    model_path = _safe_model_path(model_name)
    meta = _read_model_meta(model_path)
    meta_path = meta["path"]
    task = meta["task"]

    # 1) 如果当前端口已经是目标模型，则直接复用。
    try:
        health = _json_get(_health_url(), timeout=1.0)
        if _health_matches_model(health, model_path, meta_path, meta):
            _loaded_model_name = model_path.name
            _loaded_model_path = str(model_path)
            _loaded_meta_path = meta_path
            _loaded_task = task
            _loaded_algorithm_signature = str(meta.get("algorithm_signature") or "")
            _loaded_pipeline_config = ""
            return {"reused": True, "health": health, "model_path": str(model_path), "meta": meta}
        logger.info(
            "验证端口已有服务但不是目标模型，将通过 switch_model.sh 切换。current_model=%s, current_task=%s, target_model=%s, target_task=%s",
            health.get("model_path"), health.get("task"), str(model_path), task,
        )
    except Exception:
        # 端口未启动服务或健康检查失败，继续通过 switch_model.sh 切换。
        pass

    # 2) 统一通过 systemd 服务切换模型，不再由 collector 直接启动第二个 engine.py。
    #    这可以避免检测/分类/ROI双模型来回切换时多个进程抢占同一端口。
    _stop_validation_process()
    _kill_validation_engine_on_port()
    switched = _switch_model_via_system_service(model_path, meta_path, meta)
    health = switched.get("health", {})
    if not _health_matches_model(health, model_path, meta_path, meta, ignore_force=True):
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
    _loaded_algorithm_signature = str(meta.get("algorithm_signature") or "")
    _loaded_pipeline_config = ""
    _force_algorithm_reload = False
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


def _ensure_detection_centers(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """兼容旧版 engine：如果检测结果没有 center 字段，则根据 bbox 补算中心点。"""
    fixed: List[Dict[str, Any]] = []
    for pred in predictions or []:
        item = dict(pred)
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = [float(x) for x in bbox[:4]]
                cx = round((x1 + x2) / 2.0, 2)
                cy = round((y1 + y2) / 2.0, 2)
                item["bbox"] = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                item.setdefault("center", [cx, cy])
                item.setdefault("center_x", cx)
                item.setdefault("center_y", cy)
            except Exception:
                pass
        fixed.append(item)
    return fixed


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
        "updated_at": time.strftime("%H:%M:%S"),
        "engine": {
            "port": _validation_port(),
            "reused": engine_info.get("reused", False),
            "model_path": engine_info.get("model_path"),
            "meta_path": meta.get("path"),
            "algorithm_signature": meta.get("algorithm_signature"),
        },
        "model_meta": {
            "task": meta.get("task"),
            "input_size": meta.get("input_size"),
            "num_classes": meta.get("num_classes"),
            "class_names": meta.get("class_names"),
            "algorithm": meta.get("algorithm"),
            "algorithm_signature": meta.get("algorithm_signature"),
        },
        "raw": raw,
    }

    if task == "classification":
        result = _normalize_classification_result(raw)
        algo = meta.get("algorithm", {}) if isinstance(meta.get("algorithm"), dict) else {}
        score_threshold = float(algo.get("score_threshold") or 0.0)
        low_policy = str(algo.get("low_confidence_policy") or "review")
        confidence = result.get("confidence")
        if confidence is not None and float(confidence) < score_threshold:
            result["low_confidence"] = True
            result["score_threshold"] = score_threshold
            if low_policy == "unknown":
                result["class_name"] = "未知"
            elif low_policy == "review":
                result["class_name"] = f"{result.get('class_name') or '未识别'}（建议复核）"
        base["result"] = result
        topk = raw.get("topk", [])
        try:
            topk = topk[:int(algo.get("topk") or len(topk))]
        except Exception:
            pass
        base["topk"] = topk
    elif task == "roi_classification":
        predictions = _ensure_detection_centers(raw.get("predictions", []))
        base["predictions"] = predictions
        base["detection"] = _summarize_detection({"predictions": predictions})
        base["roi"] = raw.get("roi")
        base["detector"] = raw.get("detector")
        base["classifier"] = raw.get("classifier")
        final_label = raw.get("final_label") or raw.get("final_decision")
        final_conf = raw.get("final_confidence")
        if final_conf is None and predictions:
            final_conf = predictions[0].get("confidence")
        try:
            final_conf_float = float(final_conf) if final_conf is not None else None
        except Exception:
            final_conf_float = None
        base["result"] = {
            "class_id": predictions[0].get("class_id") if predictions else None,
            "class_name": str(final_label or "未识别"),
            "confidence": final_conf_float,
            "confidence_percent": f"{final_conf_float * 100:.1f}%" if final_conf_float is not None else "--",
        }
        classifier_obj = raw.get("classifier") if isinstance(raw.get("classifier"), dict) else {}
        base["topk"] = classifier_obj.get("topk", []) if isinstance(classifier_obj.get("topk"), list) else []
    elif task in {"detection", "obb_detection", "segmentation"}:
        predictions = _ensure_detection_centers(raw.get("predictions", []))
        algo = meta.get("algorithm", {}) if isinstance(meta.get("algorithm"), dict) else {}
        try:
            predictions = predictions[:max(1, int(algo.get("max_results") or len(predictions)))]
        except Exception:
            pass
        base["predictions"] = predictions
        base["detection"] = _summarize_detection({"predictions": predictions})
        task_text = "旋转框检测" if task == "obb_detection" else ("实例分割" if task == "segmentation" else "检测")
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
