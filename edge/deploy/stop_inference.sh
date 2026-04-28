#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8082}"
SERVICE_NAME="${2:-visionops-inference}"

log() { echo "[stop_inference] $*"; }
err() { echo "[stop_inference][ERROR] $*" >&2; }

if ! command -v sudo >/dev/null 2>&1; then
  err "sudo 不存在，无法停止 systemd 服务或释放 root 进程"
  exit 1
fi

if ! sudo -n true 2>/dev/null; then
  err "当前用户没有免密 sudo。请在板端配置 NOPASSWD，否则 Web 客户端/部署脚本无法自动切换模型。"
  exit 11
fi

log "stopping service: ${SERVICE_NAME}"
sudo -n systemctl stop "${SERVICE_NAME}" 2>/dev/null || true

log "killing old engine.py processes"
sudo -n pkill -9 -f "/opt/visionops/edge/inference/engine.py" 2>/dev/null || true
sudo -n pkill -9 -f "edge/inference/engine.py" 2>/dev/null || true

log "freeing port: ${PORT}"
if command -v fuser >/dev/null 2>&1; then
  sudo -n fuser -k "${PORT}/tcp" 2>/dev/null || true
fi

sleep 1

if command -v ss >/dev/null 2>&1 && ss -lntp 2>/dev/null | grep -q ":${PORT}"; then
  err "port ${PORT} is still occupied:"
  ss -lntp 2>/dev/null | grep ":${PORT}" || true
  exit 12
fi

log "inference stopped and port ${PORT} is free"
