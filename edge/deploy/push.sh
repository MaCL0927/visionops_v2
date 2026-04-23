#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VisionOps Edge Deploy Script (Detection / RK3588)
# 功能：
# 1. 默认同步三件套：
#    - model.rknn
#    - edge/runtime/class_names.yaml
#    - edge/runtime/edge.env
# 2. 可选同步整个 edge/ 代码：
#    --sync-edge-code
# 3. 远端备份 current.rknn
# 4. 上传到 /tmp 后再原子替换
# 5. 重启服务并做健康检查
# 6. 失败自动回滚模型
# ============================================================

MODEL_PATH="${1:-models/export_detection/model.rknn}"
TARGET_DEVICE="${2:-}"
THIRD_ARG="${3:-}"

SYNC_EDGE_CODE="false"
if [[ "${THIRD_ARG}" == "--sync-edge-code" ]]; then
  SYNC_EDGE_CODE="true"
elif [[ -n "${THIRD_ARG}" ]]; then
  echo "[ERROR] 不支持的参数: ${THIRD_ARG}" >&2
  echo "用法: bash edge/deploy/push.sh models/export_detection/model.rknn rk3588-001 [--sync-edge-code]" >&2
  exit 1
fi

CONFIG_FILE="pipeline/configs/mlops.yaml"

LOCAL_CLASS_NAMES="edge/runtime/class_names.yaml"
LOCAL_EDGE_ENV="edge/runtime/edge.env"
LOCAL_EDGE_DIR="edge"

REMOTE_INSTALL_DIR="/opt/visionops"
REMOTE_MODEL_DIR="${REMOTE_INSTALL_DIR}/models"
REMOTE_EDGE_DIR="${REMOTE_INSTALL_DIR}/edge"
REMOTE_RUNTIME_DIR="${REMOTE_EDGE_DIR}/runtime"

REMOTE_CURRENT_MODEL="${REMOTE_MODEL_DIR}/current.rknn"
REMOTE_CLASS_NAMES="${REMOTE_RUNTIME_DIR}/class_names.yaml"
REMOTE_EDGE_ENV="${REMOTE_RUNTIME_DIR}/edge.env"

REMOTE_TMP_DIR="/tmp"
SERVICE_NAME_DEFAULT="visionops-inference"
HEALTH_URL_DEFAULT="http://localhost:8080/health"
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

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    log_error "目录不存在: $path"
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

get_file_size() {
  local path="$1"
  du -h "$path" | awk '{print $1}'
}

get_file_md5() {
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
        print(f"{host}|{port}|{user}|{deploy_path}|{service_name}|{health_url}")
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
  remote_run "$user" "$host" "$port" "curl -sf '${health_url}'"
}

main() {
  require_cmd ssh
  require_cmd scp
  require_cmd rsync
  require_cmd md5sum
  require_cmd python3

  require_file "$MODEL_PATH"
  require_file "$LOCAL_CLASS_NAMES"
  require_file "$LOCAL_EDGE_ENV"

  if [[ "$SYNC_EDGE_CODE" == "true" ]]; then
    require_dir "$LOCAL_EDGE_DIR"
  fi

  if [[ -z "$TARGET_DEVICE" ]]; then
    log_error "必须指定目标设备 ID，例如: bash edge/deploy/push.sh models/export_detection/model.rknn rk3588-001 [--sync-edge-code]"
    exit 1
  fi

  local model_size model_md5
  model_size="$(get_file_size "$MODEL_PATH")"
  model_md5="$(get_file_md5 "$MODEL_PATH")"

  log_info "准备部署到设备: ${TARGET_DEVICE}"
  log_info "模型: $MODEL_PATH (大小: ${model_size}, MD5: ${model_md5})"
  log_info "同步 runtime: ${LOCAL_CLASS_NAMES}, ${LOCAL_EDGE_ENV}"
  log_info "同步 edge 代码: ${SYNC_EDGE_CODE}"
  log_info "─────────────────────────────────────────"

  local host="" port="22" user="ubuntu"
  local deploy_path="${REMOTE_MODEL_DIR}/"
  local service_name="${SERVICE_NAME_DEFAULT}"
  local health_url="${HEALTH_URL_DEFAULT}"

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

  IFS='|' read -r host port user deploy_path service_name health_url <<< "$parsed"

  if [[ -z "$host" ]]; then
    log_error "设备 $TARGET_DEVICE 的 host 为空"
    exit 1
  fi

  # 如果 mlops.yaml 配了 deploy_path，就继续兼容，但我们统一模型正式目录仍按 deploy_path
  REMOTE_MODEL_DIR="${deploy_path%/}"
  REMOTE_CURRENT_MODEL="${REMOTE_MODEL_DIR}/current.rknn"

  local ts
  ts="$(date +%Y%m%d_%H%M%S)"

  local remote_tmp_model="${REMOTE_TMP_DIR}/visionops_model_${ts}.rknn"
  local remote_tmp_class_names="${REMOTE_TMP_DIR}/visionops_class_names_${ts}.yaml"
  local remote_tmp_edge_env="${REMOTE_TMP_DIR}/visionops_edge_${ts}.env"
  local remote_backup_model="${REMOTE_MODEL_DIR}/backup_${ts}.rknn"

  log_info "目标设备: ${user}@${host}:${port}"

  # 1) 连通性检查
  if ! remote_run "$user" "$host" "$port" "echo connected >/dev/null"; then
    log_error "无法连接到设备 ${TARGET_DEVICE} (${host}:${port})"
    exit 1
  fi
  log_ok "设备连通性验证通过"

  # 2) 创建目录
  remote_sudo_cmd "$user" "$host" "$port" mkdir -p "${REMOTE_MODEL_DIR}" "${REMOTE_RUNTIME_DIR}"

  # 3) 备份 current.rknn
  if remote_run "$user" "$host" "$port" "[ -f '${REMOTE_CURRENT_MODEL}' ]"; then
    remote_sudo_cmd "$user" "$host" "$port" cp "${REMOTE_CURRENT_MODEL}" "${remote_backup_model}"
    log_ok "已备份当前模型 -> ${remote_backup_model}"
  else
    log_warn "远端不存在 current.rknn，将作为首次部署"
  fi

  # 4) 上传三件套到 /tmp
  log_info "上传模型到远端临时目录..."
  scp ${SSH_OPTS} -P "$port" "$MODEL_PATH" "${user}@${host}:${remote_tmp_model}"

  log_info "上传 class_names.yaml 到远端临时目录..."
  scp ${SSH_OPTS} -P "$port" "$LOCAL_CLASS_NAMES" "${user}@${host}:${remote_tmp_class_names}"

  log_info "上传 edge.env 到远端临时目录..."
  scp ${SSH_OPTS} -P "$port" "$LOCAL_EDGE_ENV" "${user}@${host}:${remote_tmp_edge_env}"

  log_ok "三件套上传完成"

  # 5) 校验模型 MD5
  local remote_md5
  remote_md5="$(remote_run "$user" "$host" "$port" "md5sum '${remote_tmp_model}' | awk '{print \$1}'")"
  if [[ "$remote_md5" != "$model_md5" ]]; then
    log_error "模型 MD5 校验失败: 本地=${model_md5}, 远端=${remote_md5}"
    remote_run "$user" "$host" "$port" "rm -f '${remote_tmp_model}' '${remote_tmp_class_names}' '${remote_tmp_edge_env}'" || true
    exit 1
  fi
  log_ok "模型 MD5 校验通过"

  # 6) 可选同步整个 edge 代码
  if [[ "$SYNC_EDGE_CODE" == "true" ]]; then
    log_info "同步整个 edge/ 目录..."
    rsync -az --delete -e "ssh ${SSH_OPTS} -p ${port}" "${LOCAL_EDGE_DIR}/" "${user}@${host}:${REMOTE_EDGE_DIR}/"
    log_ok "edge/ 代码同步完成"
  fi

  # 7) 原子替换模型 + 覆盖 runtime 配置
  remote_sudo_cmd "$user" "$host" "$port" chown "${user}:${user}" "${remote_tmp_model}" "${remote_tmp_class_names}" "${remote_tmp_edge_env}" || true

  remote_sudo_cmd "$user" "$host" "$port" mv "${remote_tmp_model}" "${REMOTE_CURRENT_MODEL}"
  remote_sudo_cmd "$user" "$host" "$port" chmod 644 "${REMOTE_CURRENT_MODEL}"

  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_class_names}" "${REMOTE_CLASS_NAMES}"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_edge_env}" "${REMOTE_EDGE_ENV}"

  remote_run "$user" "$host" "$port" "rm -f '${remote_tmp_class_names}' '${remote_tmp_edge_env}'" || true

  log_ok "已更新模型与 runtime 配置"
  log_ok "  模型 -> ${REMOTE_CURRENT_MODEL}"
  log_ok "  类别 -> ${REMOTE_CLASS_NAMES}"
  log_ok "  环境 -> ${REMOTE_EDGE_ENV}"

  # 8) 若同步了 edge 代码，先 daemon-reload
  if [[ "$SYNC_EDGE_CODE" == "true" ]]; then
    log_info "检测到同步 edge 代码，执行 daemon-reload..."
    remote_sudo_cmd "$user" "$host" "$port" systemctl daemon-reload
  fi

  # 9) 统一重启服务
  log_info "重启服务: ${service_name}"
  if ! remote_sudo_cmd "$user" "$host" "$port" systemctl restart "${service_name}"; then
    log_error "服务重启失败，准备回滚模型..."
    if remote_run "$user" "$host" "$port" "[ -f '${remote_backup_model}' ]"; then
      remote_sudo_cmd "$user" "$host" "$port" cp "${remote_backup_model}" "${REMOTE_CURRENT_MODEL}"
      remote_sudo_cmd "$user" "$host" "$port" systemctl restart "${service_name}" || true
      log_warn "已回滚到备份模型"
    fi
    exit 1
  fi
  log_ok "服务重启成功"

  # 10) 健康检查
  sleep 2
  if do_health_check "$user" "$host" "$port" "$health_url" >/dev/null; then
    log_ok "服务健康检查通过"
  else
    log_error "服务健康检查失败，回滚模型中..."
    if remote_run "$user" "$host" "$port" "[ -f '${remote_backup_model}' ]"; then
      remote_sudo_cmd "$user" "$host" "$port" cp "${remote_backup_model}" "${REMOTE_CURRENT_MODEL}"
      remote_sudo_cmd "$user" "$host" "$port" systemctl restart "${service_name}" || true
      log_warn "已回滚到备份模型"
    fi
    exit 1
  fi

  # 11) 打印健康检查结果
  log_info "当前健康检查结果："
  remote_run "$user" "$host" "$port" "curl -sf '${health_url}'" || true

  log_ok "─────────────────────────────────────────"
  log_ok "部署完成"
  log_ok "  设备: ${TARGET_DEVICE}"
  log_ok "  模型: ${REMOTE_CURRENT_MODEL}"
  log_ok "  类别配置: ${REMOTE_CLASS_NAMES}"
  log_ok "  环境配置: ${REMOTE_EDGE_ENV}"
  log_ok "  服务: ${service_name}"
  log_ok "  同步 edge 代码: ${SYNC_EDGE_CODE}"
  log_ok "─────────────────────────────────────────"
}

main "$@"