#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VisionOps Edge Model Deploy Script (Detection / RK3588)
# - 默认部署 detection RKNN 模型
# - 支持 ubuntu + sudo
# - 上传到 /tmp 后再原子替换 current.rknn
# - 失败自动回滚
# ============================================================

MODEL_PATH="${1:-models/export_detection/model.rknn}"
TARGET_DEVICE="${2:-}"
CONFIG_FILE="pipeline/configs/mlops.yaml"

REMOTE_MODEL_DIR="/opt/visionops/models"
REMOTE_CURRENT_MODEL="${REMOTE_MODEL_DIR}/current.rknn"
REMOTE_TMP_DIR="/tmp"
SERVICE_NAME_DEFAULT="visionops-inference"
HEALTH_URL_DEFAULT="http://localhost:8080/health"
RELOAD_URL_DEFAULT="http://localhost:8080/reload"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

log_info()  { echo "[INFO] $*"; }
log_ok()    { echo "[OK] $*"; }
log_warn()  { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log_error "文件不存在: $path"
    exit 1
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log_error "缺少命令: $cmd"
    exit 1
  fi
}

get_model_size() {
  local path="$1"
  du -h "$path" | awk '{print $1}'
}

get_model_md5() {
  local path="$1"
  md5sum "$path" | awk '{print $1}'
}

parse_device_from_yaml() {
  local config_file="$1"
  local device_id="$2"

  python3 - <<'PY' "$config_file" "$device_id"
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    print("YAML_IMPORT_ERROR")
    sys.exit(0)

cfg_path = Path(sys.argv[1])
device_id = sys.argv[2]

if not cfg_path.exists():
    print("CONFIG_NOT_FOUND")
    sys.exit(0)

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

devices = cfg.get("edge_devices", [])
for d in devices:
    if str(d.get("id")) == device_id:
        host = d.get("host", "")
        port = d.get("port", 22)
        user = d.get("user", "ubuntu")
        deploy_path = d.get("deploy_path", "/opt/visionops/models/")
        service_name = d.get("service_name", "visionops-inference")
        health_url = d.get("health_url", "http://localhost:8080/health")
        reload_url = d.get("reload_url", "http://localhost:8080/reload")
        print(f"{host}|{port}|{user}|{deploy_path}|{service_name}|{health_url}|{reload_url}")
        sys.exit(0)

print("DEVICE_NOT_FOUND")
PY
}

remote_run() {
  local user="$1"
  local host="$2"
  local port="$3"
  local cmd="$4"
  ssh ${SSH_OPTS} -p "$port" "${user}@${host}" "$cmd"
}

remote_sudo_cmd() {
  local user="$1"
  local host="$2"
  local port="$3"
  shift 3
  ssh ${SSH_OPTS} -p "$port" "${user}@${host}" sudo -n "$@"
}

do_health_check() {
  local user="$1"
  local host="$2"
  local port="$3"
  local health_url="$4"
  remote_run "$user" "$host" "$port" "curl -sf '$health_url' >/dev/null"
}

main() {
  require_cmd ssh
  require_cmd scp
  require_cmd md5sum
  require_cmd python3

  require_file "$MODEL_PATH"

  local model_size model_md5
  model_size="$(get_model_size "$MODEL_PATH")"
  model_md5="$(get_model_md5 "$MODEL_PATH")"

  log_info "准备部署模型: $MODEL_PATH (大小: ${model_size}, MD5: ${model_md5})"
  log_info "─────────────────────────────────────────"

  local host="" port="22" user="ubuntu"
  local deploy_path="${REMOTE_MODEL_DIR}/"
  local service_name="${SERVICE_NAME_DEFAULT}"
  local health_url="${HEALTH_URL_DEFAULT}"
  local reload_url="${RELOAD_URL_DEFAULT}"

  if [[ -n "$TARGET_DEVICE" ]]; then
    local parsed
    parsed="$(parse_device_from_yaml "$CONFIG_FILE" "$TARGET_DEVICE")"

    if [[ "$parsed" == "YAML_IMPORT_ERROR" ]]; then
      log_error "python3 缺少 pyyaml，无法解析 $CONFIG_FILE"
      exit 1
    elif [[ "$parsed" == "CONFIG_NOT_FOUND" ]]; then
      log_error "配置文件不存在: $CONFIG_FILE"
      exit 1
    elif [[ "$parsed" == "DEVICE_NOT_FOUND" ]]; then
      log_error "未在 $CONFIG_FILE 中找到设备: $TARGET_DEVICE"
      exit 1
    fi

    IFS='|' read -r host port user deploy_path service_name health_url reload_url <<< "$parsed"
  else
    log_error "必须指定目标设备 ID，例如: make deploy-detection DEVICE=rk3588-001"
    exit 1
  fi

  if [[ -z "$host" ]]; then
    log_error "设备 $TARGET_DEVICE 的 host 为空"
    exit 1
  fi

  # 统一使用正式目录
  REMOTE_MODEL_DIR="${deploy_path%/}"
  REMOTE_CURRENT_MODEL="${REMOTE_MODEL_DIR}/current.rknn"

  local ts remote_tmp_model remote_backup_model
  ts="$(date +%Y%m%d_%H%M%S)"
  remote_tmp_model="${REMOTE_TMP_DIR}/visionops_model_${ts}.rknn"
  remote_backup_model="${REMOTE_MODEL_DIR}/backup_${ts}.rknn"

  log_info "部署到设备: ${TARGET_DEVICE} (${user}@${host}:${port})"

  # 1) 连通性检查
  if ! remote_run "$user" "$host" "$port" "echo connected >/dev/null"; then
    log_error "无法连接到设备 ${TARGET_DEVICE} (${host}:${port})"
    exit 1
  fi
  log_ok "设备连通性验证通过"

  # 2) 创建目录（需要 sudo）
  remote_sudo_cmd "$user" "$host" "$port" mkdir -p "${REMOTE_MODEL_DIR}"

  # 3) 如有现模型，先备份
  if remote_run "$user" "$host" "$port" "[ -f '${REMOTE_CURRENT_MODEL}' ]"; then
    remote_sudo_cmd "$user" "$host" "$port" cp "${REMOTE_CURRENT_MODEL}" "${remote_backup_model}"
    log_ok "已备份当前模型 -> ${remote_backup_model}"
  else
    log_warn "远端不存在 current.rknn，将作为首次部署"
  fi

  # 4) 上传到 /tmp
  log_info "上传模型文件到远端临时目录..."
  scp ${SSH_OPTS} -P "$port" "$MODEL_PATH" "${user}@${host}:${remote_tmp_model}"
  log_ok "模型上传完成 -> ${remote_tmp_model}"

  # 5) 校验远端临时文件 MD5
  local remote_md5
  remote_md5="$(remote_run "$user" "$host" "$port" "md5sum '${remote_tmp_model}' | awk '{print \$1}'")"
  if [[ "$remote_md5" != "$model_md5" ]]; then
    log_error "MD5 校验失败: 本地=${model_md5}, 远端=${remote_md5}"
    remote_run "$user" "$host" "$port" "rm -f '${remote_tmp_model}'" || true
    exit 1
  fi
  log_ok "MD5 校验通过"

  # 6) 原子替换 current.rknn
  remote_sudo_cmd "$user" "$host" "$port" chown "${user}:${user}" "${remote_tmp_model}" || true
  remote_sudo_cmd "$user" "$host" "$port" mv "${remote_tmp_model}" "${REMOTE_CURRENT_MODEL}"
  remote_sudo_cmd "$user" "$host" "$port" chmod 644 "${REMOTE_CURRENT_MODEL}"
  log_ok "已切换正式模型 -> ${REMOTE_CURRENT_MODEL}"

  # 7) 优先尝试热重载
  if remote_run "$user" "$host" "$port" "curl -sf -X POST '${reload_url}?model_path=${REMOTE_CURRENT_MODEL}' >/dev/null"; then
    log_ok "API 热重载成功"
  else
    log_warn "API 热重载不可用，尝试重启 systemd 服务..."
    if ! rremote_sudo_cmd "$user" "$host" "$port" systemctl restart "${service_name}"; then
      log_error "服务重启失败，准备回滚..."
      if remote_run "$user" "$host" "$port" "[ -f '${remote_backup_model}' ]"; then
        remote_sudo_run "$user" "$host" "$port" "
          cp '${remote_backup_model}' '${REMOTE_CURRENT_MODEL}'
          systemctl restart '${service_name}' || true
        "
        log_warn "已回滚到备份模型"
      fi
      exit 1
    fi
    log_ok "systemd 服务重启成功"
  fi

  # 8) 健康检查
  sleep 2
  if do_health_check "$user" "$host" "$port" "$health_url"; then
    log_ok "服务健康检查通过"
  else
    log_error "服务健康检查失败，回滚中..."
    if remote_run "$user" "$host" "$port" "[ -f '${remote_backup_model}' ]"; then
      remote_sudo_cmd "$user" "$host" "$port" cp "${remote_backup_model}" "${REMOTE_CURRENT_MODEL}"
      remote_sudo_cmd "$user" "$host" "$port" systemctl restart "${service_name}" || true
      log_warn "已回滚到备份模型"
    fi
    exit 1
  fi

  log_ok "─────────────────────────────────────────"
  log_ok "模型部署完成"
  log_ok "  设备: ${TARGET_DEVICE}"
  log_ok "  模型: ${REMOTE_CURRENT_MODEL}"
  log_ok "  服务: ${service_name}"
  log_ok "─────────────────────────────────────────"
}

main "$@"