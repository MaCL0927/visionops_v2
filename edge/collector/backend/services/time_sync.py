#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime as _dt
import re
import subprocess
from typing import Any, Dict, List

from backend.services.settings_store import get_time_sync_runtime_config, load_settings


def _run_cmd(args: List[str], timeout: int = 5) -> Dict[str, Any]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "cmd": args,
        }
    except FileNotFoundError:
        return {"ok": False, "returncode": 127, "stdout": "", "stderr": f"命令不存在: {args[0]}", "cmd": args}
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": 124, "stdout": "", "stderr": f"命令超时: {' '.join(args)}", "cmd": args}
    except Exception as exc:
        return {"ok": False, "returncode": 1, "stdout": "", "stderr": str(exc), "cmd": args}


def _parse_tracking(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        result[key] = value.strip()
    # 提取 Reference ID 中的 IP
    ref = result.get("reference_id", "")
    m = re.search(r"\(([^)]+)\)", ref)
    if m:
        result["reference_source"] = m.group(1)
    # 提取常用数值
    for src_key, dst_key in [
        ("stratum", "stratum"),
        ("last_offset", "last_offset_sec"),
        ("rms_offset", "rms_offset_sec"),
        ("system_time", "system_time_offset_sec"),
        ("root_delay", "root_delay_sec"),
        ("root_dispersion", "root_dispersion_sec"),
        ("update_interval", "update_interval_sec"),
    ]:
        raw = result.get(src_key)
        if raw is None:
            continue
        m = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
        if not m:
            continue
        try:
            val = float(m.group(0))
            result[dst_key] = int(val) if dst_key == "stratum" else val
        except Exception:
            pass
    return result


def _parse_sources(text: str) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith(".") or s.startswith("/") or s.startswith("|") or s.startswith("=") or s.startswith("MS "):
            continue
        if len(s) < 2 or s[0] not in "^=#" or s[1] not in "*+-?x~":
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        entry: Dict[str, Any] = {
            "mode": s[0],
            "state": s[1],
            "selected": s[1] == "*",
            "combined": s[1] == "+",
            "reachable_state": s[1] not in {"?", "x", "~"},
            "name": parts[1],
            "raw": s,
        }
        if len(parts) > 2:
            try:
                entry["stratum"] = int(parts[2])
            except Exception:
                pass
        if len(parts) > 3:
            try:
                entry["poll"] = int(parts[3])
            except Exception:
                pass
        if len(parts) > 4:
            entry["reach"] = parts[4]
        if len(parts) > 5:
            entry["last_rx"] = parts[5]
        sources.append(entry)
    return sources


def get_time_sync_status() -> Dict[str, Any]:
    config = get_time_sync_runtime_config(load_settings())
    sources_res = _run_cmd(["chronyc", "sources", "-v"], timeout=5)
    tracking_res = _run_cmd(["chronyc", "tracking"], timeout=5)
    timedate_res = _run_cmd(["timedatectl", "status", "--no-pager"], timeout=5)

    sources = _parse_sources(sources_res.get("stdout", "")) if sources_res["ok"] else []
    tracking = _parse_tracking(tracking_res.get("stdout", "")) if tracking_res["ok"] else {}
    selected = next((s for s in sources if s.get("selected")), None)
    configured_server = config.get("ntp_server", "")
    configured_source = next((s for s in sources if configured_server and s.get("name") == configured_server), None)
    leap = tracking.get("leap_status", "")
    synced = bool(selected and str(leap).lower() == "normal")

    return {
        "ok": True,
        "config": config,
        "status": {
            "synced": synced,
            "selected_source": selected.get("name") if selected else "",
            "configured_source_seen": bool(configured_source),
            "configured_source_selected": bool(configured_source and configured_source.get("selected")),
            "reference_source": tracking.get("reference_source", ""),
            "leap_status": leap,
            "stratum": tracking.get("stratum"),
            "system_time_offset_sec": tracking.get("system_time_offset_sec"),
            "last_offset_sec": tracking.get("last_offset_sec"),
            "rms_offset_sec": tracking.get("rms_offset_sec"),
            "root_delay_sec": tracking.get("root_delay_sec"),
            "root_dispersion_sec": tracking.get("root_dispersion_sec"),
            "update_interval_sec": tracking.get("update_interval_sec"),
            "checked_at": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        "sources": sources,
        "raw": {
            "sources_ok": sources_res["ok"],
            "sources_stdout": sources_res.get("stdout", ""),
            "sources_stderr": sources_res.get("stderr", ""),
            "tracking_ok": tracking_res["ok"],
            "tracking_stdout": tracking_res.get("stdout", ""),
            "tracking_stderr": tracking_res.get("stderr", ""),
            "timedate_ok": timedate_res["ok"],
            "timedate_stdout": timedate_res.get("stdout", ""),
            "timedate_stderr": timedate_res.get("stderr", ""),
        },
        "message": "时间同步状态已读取" if (sources_res["ok"] or tracking_res["ok"]) else "无法读取 chrony 状态，请确认 chrony 已安装并运行",
    }


def test_time_sync() -> Dict[str, Any]:
    status = get_time_sync_status()
    config = status.get("config", {})
    st = status.get("status", {})
    server = config.get("ntp_server") or ""
    if not server:
        status["ok"] = False
        status["message"] = "未配置上位机 NTP 地址"
    elif st.get("configured_source_selected"):
        status["ok"] = True
        status["message"] = f"NTP 同步正常，当前已选中上位机 {server}"
    elif st.get("configured_source_seen"):
        status["ok"] = True
        status["message"] = f"已检测到上位机 {server}，但当前未被选为 ^*，请稍等或检查 chrony 选择状态"
    else:
        status["ok"] = False
        status["message"] = f"未在 chrony sources 中看到上位机 {server}，请检查 RK3588 chrony 配置和网络"
    return status
