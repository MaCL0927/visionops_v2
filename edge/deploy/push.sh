#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VisionOps Edge Deploy Script (Detection / Classification / RK3588)
# 统一原则：
# 1. 部署时由 --task 控制 detection / classification
# 2. 分类类别文件写入 class_names_classification.yaml，不覆盖检测 class_names.yaml
# 3. 上传 /opt/visionops/.env，INPUT_SIZE 使用逗号格式：224,224 / 640,640
# 4. 自动安装/刷新 visionops-inference.service，后续不用手动改板端 service
# 5. systemd 不传 --input-size，由 engine.py 读取 INPUT_SIZE 环境变量
# ============================================================

MODEL_PATH="${1:-models/export_detection/model.rknn}"
TARGET_DEVICE="${2:-}"
shift $(( $# >= 1 ? 1 : 0 )) || true
shift $(( $# >= 1 ? 1 : 0 )) || true

CONFIG_FILE="pipeline/configs/mlops.yaml"
LOCAL_EDGE_DIR="edge"
LOCAL_EDGE_ENV_DEFAULT="edge/runtime/edge.env"
LOCAL_EDGE_ENV="${LOCAL_EDGE_ENV_DEFAULT}"

SYNC_EDGE_CODE="false"
TASK=""
LOCAL_CLASS_NAMES_OVERRIDE=""
TOPK_OVERRIDE=""
INPUT_SIZE_OVERRIDE=""

REMOTE_INSTALL_DIR="/opt/visionops"
REMOTE_MODEL_DIR="${REMOTE_INSTALL_DIR}/models"
REMOTE_EDGE_DIR="${REMOTE_INSTALL_DIR}/edge"
REMOTE_RUNTIME_DIR="${REMOTE_EDGE_DIR}/runtime"
REMOTE_CURRENT_MODEL="${REMOTE_MODEL_DIR}/current.rknn"
REMOTE_EDGE_ENV="${REMOTE_RUNTIME_DIR}/edge.env"
REMOTE_ROOT_ENV="${REMOTE_INSTALL_DIR}/.env"
REMOTE_TMP_DIR="/tmp"
REMOTE_INFERENCE_SERVICE="/etc/systemd/system/visionops-inference.service"

SERVICE_NAME_DEFAULT="visionops-inference"
HEALTH_URL_DEFAULT="http://localhost:8080/health"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

usage() {
  cat <<USAGE
用法：
  bash edge/deploy/push.sh <model.rknn> <device-id> [options]

示例：
  bash edge/deploy/push.sh models/export_detection/model.rknn rk3588-001 --task detection --sync-edge-code
  bash edge/deploy/push.sh models/export/model.rknn rk3588-001 --task classification --sync-edge-code

参数：
  --task detection|classification
  --class-names <path>
  --edge-env <path>
  --input-size "H,W" 或 "H W"
  --topk <N>
  --sync-edge-code
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync-edge-code) SYNC_EDGE_CODE="true"; shift ;;
    --task) TASK="${2:-}"; shift 2 ;;
    --class-names) LOCAL_CLASS_NAMES_OVERRIDE="${2:-}"; shift 2 ;;
    --edge-env) LOCAL_EDGE_ENV="${2:-}"; shift 2 ;;
    --input-size) INPUT_SIZE_OVERRIDE="${2:-}"; shift 2 ;;
    --topk) TOPK_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 不支持的参数: $1" >&2; usage; exit 1 ;;
  esac
done

log_info()  { echo "[INFO] $*"; }
log_ok()    { echo "[OK] $*"; }
log_warn()  { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

require_file() { [[ -f "$1" ]] || { log_error "文件不存在: $1"; exit 1; }; }
require_dir()  { [[ -d "$1" ]] || { log_error "目录不存在: $1"; exit 1; }; }
require_cmd()  { command -v "$1" >/dev/null 2>&1 || { log_error "缺少命令: $1"; exit 1; }; }
get_file_size() { du -h "$1" | awk '{print $1}'; }
get_file_md5() { md5sum "$1" | awk '{print $1}'; }

read_env_value() {
  local key="$1" file="$2"
  [[ -f "$file" ]] || return 0
  grep -E "^${key}=" "$file" | tail -n 1 | cut -d= -f2- || true
}

normalize_task() {
  local raw="$1"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
  if [[ "$raw" != "detection" && "$raw" != "classification" ]]; then
    log_error "TASK 只能是 detection 或 classification，当前: ${raw}"
    exit 1
  fi
  echo "$raw"
}

infer_task() {
  if [[ -n "$TASK" ]]; then normalize_task "$TASK"; return; fi
  local env_task
  env_task="$(read_env_value TASK "$LOCAL_EDGE_ENV")"
  if [[ -n "$env_task" ]]; then normalize_task "$env_task"; return; fi
  if [[ "$MODEL_PATH" == *"export_detection"* || "$MODEL_PATH" == *"_detection"* ]]; then echo "detection"; else echo "classification"; fi
}

normalize_input_size_comma() {
  local value="$1"
  value="${value//,/ }"
  value="$(echo "$value" | xargs || true)"
  local h="" w="" extra=""
  read -r h w extra <<< "$value"
  if [[ -z "${h}" || -z "${w}" || -n "${extra}" ]]; then
    log_error "INPUT_SIZE 格式错误，应为 '640,640'、'640 640'、'224,224' 或 '224 224'，当前: ${value}"
    exit 1
  fi
  echo "${h},${w}"
}

parse_class_names_info() {
  local class_file="$1"
  python3 - <<'PY' "$class_file"
import sys
from pathlib import Path
try:
    import yaml
except Exception:
    print("|0")
    sys.exit(0)
p = Path(sys.argv[1])
if not p.exists():
    print("|0")
    sys.exit(0)
with p.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
class_names = data.get("class_names") or []
task = str(data.get("task") or "")
num_classes = data.get("num_classes")
if not isinstance(class_names, list):
    class_names = []
if num_classes is None:
    num_classes = len(class_names)
try:
    num_classes = int(num_classes)
except Exception:
    num_classes = len(class_names)
print(f"{task}|{num_classes}")
PY
}

resolve_local_class_names() {
  local task="$1"
  if [[ -n "$LOCAL_CLASS_NAMES_OVERRIDE" ]]; then echo "$LOCAL_CLASS_NAMES_OVERRIDE"; return; fi
  if [[ "$task" == "classification" ]]; then
    if [[ -f "edge/runtime/class_names_classification.yaml" ]]; then echo "edge/runtime/class_names_classification.yaml";
    elif [[ -f "data/processed/class_names.yaml" ]]; then echo "data/processed/class_names.yaml";
    else echo "edge/runtime/class_names_classification.yaml"; fi
  else
    echo "edge/runtime/class_names.yaml"
  fi
}

resolve_remote_class_names() {
  local task="$1"
  if [[ "$task" == "classification" ]]; then echo "${REMOTE_RUNTIME_DIR}/class_names_classification.yaml"; else echo "${REMOTE_RUNTIME_DIR}/class_names.yaml"; fi
}

parse_device_from_yaml() {
  python3 - <<'PY' "$1" "$2"
import sys
from pathlib import Path
try:
    import yaml
except Exception:
    print("YAML_IMPORT_ERROR"); sys.exit(0)
cfg_path = Path(sys.argv[1]); device_id = sys.argv[2]
if not cfg_path.exists():
    print("CONFIG_NOT_FOUND"); sys.exit(0)
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
for d in cfg.get("edge_devices", []):
    if str(d.get("id")) == device_id:
        print(f"{d.get('host','')}|{d.get('port',22)}|{d.get('user','ubuntu')}|{d.get('deploy_path','/opt/visionops/models/')}|{d.get('service_name','visionops-inference')}|{d.get('health_url','http://localhost:8080/health')}")
        sys.exit(0)
print("DEVICE_NOT_FOUND")
PY
}

remote_run() { local user="$1" host="$2" port="$3" cmd="$4"; ssh ${SSH_OPTS} -p "$port" "${user}@${host}" "$cmd"; }
remote_sudo_cmd() { local user="$1" host="$2" port="$3"; shift 3; ssh ${SSH_OPTS} -p "$port" "${user}@${host}" sudo -n "$@"; }
do_health_check() { remote_run "$1" "$2" "$3" "curl -sf '$4'"; }

build_deploy_env() {
  local task="$1" local_class_names="$2" remote_class_names="$3" out_file="$4"
  local class_task class_num_classes
  IFS='|' read -r class_task class_num_classes <<< "$(parse_class_names_info "$local_class_names")"

  local env_num_classes env_input_size env_topk env_npu_core env_port env_metrics_port env_conf env_nms env_warmup env_report_interval
  env_num_classes="$(read_env_value NUM_CLASSES "$LOCAL_EDGE_ENV")"
  if [[ -n "${class_num_classes:-}" && "${class_num_classes}" != "0" ]]; then env_num_classes="${class_num_classes}"; fi
  [[ -n "$env_num_classes" ]] || env_num_classes=$([[ "$task" == "classification" ]] && echo "2" || echo "6")

  if [[ -n "$INPUT_SIZE_OVERRIDE" ]]; then
    env_input_size="$(normalize_input_size_for_env "$INPUT_SIZE_OVERRIDE")"
  else
    # 关键修改：
    # 只要命令行明确传了 --task classification，就默认使用 224,224，
    # 不再被 edge/runtime/edge.env 里的 INPUT_SIZE=640,640 影响。
    if [[ "$task" == "classification" ]]; then
      env_input_size="224,224"
    else
      env_input_size="640,640"
    fi
  fi

  if [[ -n "$TOPK_OVERRIDE" ]]; then env_topk="$TOPK_OVERRIDE"; else env_topk="$(read_env_value TOPK "$LOCAL_EDGE_ENV")"; [[ -n "$env_topk" ]] || env_topk=$([[ "$task" == "classification" ]] && echo "2" || echo "5"); fi
  env_npu_core="$(read_env_value NPU_CORE "$LOCAL_EDGE_ENV")"; [[ -n "$env_npu_core" ]] || env_npu_core="auto"
  env_port="$(read_env_value PORT "$LOCAL_EDGE_ENV")"; [[ -n "$env_port" ]] || env_port="8080"
  env_metrics_port="$(read_env_value METRICS_PORT "$LOCAL_EDGE_ENV")"; [[ -n "$env_metrics_port" ]] || env_metrics_port="9091"
  env_conf="$(read_env_value CONF_THRESHOLD "$LOCAL_EDGE_ENV")"; [[ -n "$env_conf" ]] || env_conf="0.25"
  env_nms="$(read_env_value NMS_THRESHOLD "$LOCAL_EDGE_ENV")"; [[ -n "$env_nms" ]] || env_nms="0.45"
  env_warmup="$(read_env_value WARMUP_RUNS "$LOCAL_EDGE_ENV")"; [[ -n "$env_warmup" ]] || env_warmup="3"
  env_report_interval="$(read_env_value REPORT_INTERVAL "$LOCAL_EDGE_ENV")"; [[ -n "$env_report_interval" ]] || env_report_interval="60"

  cat > "$out_file" <<EOF_ENV
# Auto generated by edge/deploy/push.sh
DEVICE_ID=${TARGET_DEVICE}
MODEL_PATH=${REMOTE_CURRENT_MODEL}
INFERENCE_URL=http://localhost:${env_port}
REPORT_INTERVAL=${env_report_interval}
TASK=${task}
NPU_CORE=${env_npu_core}
NUM_CLASSES=${env_num_classes}
INPUT_SIZE=${env_input_size}
CLASS_NAMES_FILE=${remote_class_names}
PORT=${env_port}
METRICS_PORT=${env_metrics_port}
CONF_THRESHOLD=${env_conf}
NMS_THRESHOLD=${env_nms}
TOPK=${env_topk}
WARMUP_RUNS=${env_warmup}
EOF_ENV
}

build_inference_service_file() {
  local out_file="$1"
  cat > "$out_file" <<'EOF_SERVICE'
[Unit]
Description=VisionOps RK3588 Edge Inference Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/visionops
EnvironmentFile=/opt/visionops/.env

ExecStart=/opt/visionops/venv/bin/python /opt/visionops/edge/inference/engine.py \
    --model ${MODEL_PATH} \
    --task ${TASK} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --npu-core ${NPU_CORE} \
    --num-classes ${NUM_CLASSES} \
    --class-names-file ${CLASS_NAMES_FILE} \
    --metrics-port ${METRICS_PORT} \
    --conf-threshold ${CONF_THRESHOLD} \
    --nms-threshold ${NMS_THRESHOLD} \
    --topk ${TOPK} \
    --warmup-runs ${WARMUP_RUNS}

Restart=always
RestartSec=5
TimeoutStartSec=30
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-inference

[Install]
WantedBy=multi-user.target
EOF_SERVICE
}

main() {
  require_cmd ssh; require_cmd scp; require_cmd rsync; require_cmd md5sum; require_cmd python3
  [[ -n "$TARGET_DEVICE" ]] || { log_error "必须指定目标设备 ID"; usage; exit 1; }
  require_file "$MODEL_PATH"; require_file "$LOCAL_EDGE_ENV"
  [[ "$SYNC_EDGE_CODE" == "true" ]] && require_dir "$LOCAL_EDGE_DIR"

  local task local_class_names remote_class_names
  task="$(infer_task)"
  local_class_names="$(resolve_local_class_names "$task")"; require_file "$local_class_names"
  remote_class_names="$(resolve_remote_class_names "$task")"

  local model_size model_md5
  model_size="$(get_file_size "$MODEL_PATH")"; model_md5="$(get_file_md5 "$MODEL_PATH")"

  log_info "准备部署到设备: ${TARGET_DEVICE}"
  log_info "任务类型: ${task}"
  log_info "模型: $MODEL_PATH (大小: ${model_size}, MD5: ${model_md5})"
  log_info "类别文件: ${local_class_names} -> ${remote_class_names}"
  log_info "edge.env 模板: ${LOCAL_EDGE_ENV}"

  local parsed host port user deploy_path service_name health_url
  parsed="$(parse_device_from_yaml "$CONFIG_FILE" "$TARGET_DEVICE")"
  case "$parsed" in YAML_IMPORT_ERROR|CONFIG_NOT_FOUND|DEVICE_NOT_FOUND) log_error "设备配置解析失败: $parsed"; exit 1 ;; esac
  IFS='|' read -r host port user deploy_path service_name health_url <<< "$parsed"
  [[ -n "$host" ]] || { log_error "设备 host 为空"; exit 1; }

  REMOTE_MODEL_DIR="${deploy_path%/}"
  REMOTE_CURRENT_MODEL="${REMOTE_MODEL_DIR}/current.rknn"
  remote_class_names="$(resolve_remote_class_names "$task")"

  local ts remote_tmp_model remote_tmp_class_names remote_tmp_edge_env remote_tmp_service remote_backup_model local_deploy_env local_service_file
  ts="$(date +%Y%m%d_%H%M%S)"
  remote_tmp_model="${REMOTE_TMP_DIR}/visionops_model_${ts}.rknn"
  remote_tmp_class_names="${REMOTE_TMP_DIR}/visionops_class_names_${task}_${ts}.yaml"
  remote_tmp_edge_env="${REMOTE_TMP_DIR}/visionops_edge_${task}_${ts}.env"
  remote_tmp_service="${REMOTE_TMP_DIR}/visionops-inference_${ts}.service"
  remote_backup_model="${REMOTE_MODEL_DIR}/backup_${ts}.rknn"
  local_deploy_env="$(mktemp /tmp/visionops_edge_env_${task}_XXXXXX.env)"
  local_service_file="$(mktemp /tmp/visionops_inference_XXXXXX.service)"

  build_deploy_env "$task" "$local_class_names" "$remote_class_names" "$local_deploy_env"
  build_inference_service_file "$local_service_file"

  log_info "目标设备: ${user}@${host}:${port}"
  remote_run "$user" "$host" "$port" "echo connected >/dev/null" || { log_error "无法连接设备"; rm -f "$local_deploy_env" "$local_service_file"; exit 1; }
  log_ok "设备连通性验证通过"

  remote_sudo_cmd "$user" "$host" "$port" mkdir -p "${REMOTE_MODEL_DIR}" "${REMOTE_RUNTIME_DIR}"
  if remote_run "$user" "$host" "$port" "[ -f '${REMOTE_CURRENT_MODEL}' ]"; then remote_sudo_cmd "$user" "$host" "$port" cp "${REMOTE_CURRENT_MODEL}" "${remote_backup_model}"; log_ok "已备份当前模型 -> ${remote_backup_model}"; fi

  log_info "上传模型、类别、env、service..."
  scp ${SSH_OPTS} -P "$port" "$MODEL_PATH" "${user}@${host}:${remote_tmp_model}"
  scp ${SSH_OPTS} -P "$port" "$local_class_names" "${user}@${host}:${remote_tmp_class_names}"
  scp ${SSH_OPTS} -P "$port" "$local_deploy_env" "${user}@${host}:${remote_tmp_edge_env}"
  scp ${SSH_OPTS} -P "$port" "$local_service_file" "${user}@${host}:${remote_tmp_service}"

  local remote_md5
  remote_md5="$(remote_run "$user" "$host" "$port" "md5sum '${remote_tmp_model}' | awk '{print \$1}'")"
  if [[ "$remote_md5" != "$model_md5" ]]; then log_error "模型 MD5 校验失败"; exit 1; fi
  log_ok "模型 MD5 校验通过"

  if [[ "$SYNC_EDGE_CODE" == "true" ]]; then
    log_info "同步整个 edge/ 目录..."
    rsync -az --delete -e "ssh ${SSH_OPTS} -p ${port}" "${LOCAL_EDGE_DIR}/" "${user}@${host}:${REMOTE_EDGE_DIR}/"
    log_ok "edge/ 代码同步完成"
  fi

  remote_sudo_cmd "$user" "$host" "$port" mv "${remote_tmp_model}" "${REMOTE_CURRENT_MODEL}"
  remote_sudo_cmd "$user" "$host" "$port" chmod 644 "${REMOTE_CURRENT_MODEL}"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_class_names}" "${remote_class_names}"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_edge_env}" "${REMOTE_EDGE_ENV}"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_edge_env}" "${REMOTE_ROOT_ENV}"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_service}" "${REMOTE_INFERENCE_SERVICE}"
  remote_run "$user" "$host" "$port" "rm -f '${remote_tmp_class_names}' '${remote_tmp_edge_env}' '${remote_tmp_service}'" || true
  rm -f "$local_deploy_env" "$local_service_file" || true

  log_ok "已更新模型、类别、env 与 systemd service"
  remote_sudo_cmd "$user" "$host" "$port" systemctl daemon-reload

  log_info "重启服务: ${service_name}"
  if ! remote_sudo_cmd "$user" "$host" "$port" systemctl restart "${service_name}"; then
    log_error "服务重启失败，回滚模型..."
    remote_run "$user" "$host" "$port" "[ -f '${remote_backup_model}' ]" && remote_sudo_cmd "$user" "$host" "$port" cp "${remote_backup_model}" "${REMOTE_CURRENT_MODEL}" || true
    exit 1
  fi

  sleep 2
  if ! do_health_check "$user" "$host" "$port" "$health_url" >/dev/null; then
    log_error "服务健康检查失败，最近日志如下："
    remote_run "$user" "$host" "$port" "journalctl -u ${service_name} -n 80 --no-pager" || true
    log_error "回滚模型中..."
    remote_run "$user" "$host" "$port" "[ -f '${remote_backup_model}' ]" && remote_sudo_cmd "$user" "$host" "$port" cp "${remote_backup_model}" "${REMOTE_CURRENT_MODEL}" || true
    remote_sudo_cmd "$user" "$host" "$port" systemctl restart "${service_name}" || true
    exit 1
  fi

  log_info "当前健康检查结果："
  local health_json
  health_json="$(remote_run "$user" "$host" "$port" "curl -sf '${health_url}'" || true)"
  echo "$health_json"
  if [[ "$health_json" != *"\"task\":\"${task}\""* && "$health_json" != *"\"task\": \"${task}\""* ]]; then
    log_error "健康检查 task 与部署任务不一致，请检查 /opt/visionops/.env"
    exit 1
  fi

  log_ok "部署完成：task=${task}, model=${REMOTE_CURRENT_MODEL}, class_names=${remote_class_names}"
}

main "$@"
