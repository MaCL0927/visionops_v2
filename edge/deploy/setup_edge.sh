#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VisionOps Edge Setup Script (RK3588 / Detection + Classification)
# 统一原则：
# 1. edge/runtime/edge.env 与 /opt/visionops/.env 中 INPUT_SIZE 都使用逗号格式，如 224,224
# 2. systemd 不传 --input-size，由 engine.py 读取环境变量 INPUT_SIZE
# 3. classification 使用 class_names_classification.yaml，避免覆盖 detection 的 class_names.yaml
# ============================================================

SERVER_URL="${1:-http://127.0.0.1:8000}"
DEVICE_ID="${2:-rk3588-001}"

INSTALL_DIR="/opt/visionops"
VENV_DIR="${INSTALL_DIR}/venv"
EDGE_DST_DIR="${INSTALL_DIR}/edge"
MODEL_DIR="${INSTALL_DIR}/models"
LOG_DIR="${INSTALL_DIR}/logs"
ENV_FILE="${INSTALL_DIR}/.env"

INFERENCE_SERVICE_PATH="/etc/systemd/system/visionops-inference.service"
MONITOR_SERVICE_PATH="/etc/systemd/system/visionops-monitor.service"

DEFAULT_MODEL_PATH="${MODEL_DIR}/current.rknn"
DEFAULT_REPORT_INTERVAL="60"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GENERATED_ENV_FILE="${REPO_ROOT}/edge/runtime/edge.env"

log_info()  { echo "[INFO] $*"; }
log_ok()    { echo "[OK] $*"; }
log_warn()  { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || { log_error "缺少命令: $1"; exit 1; }; }
ensure_root() { [[ "${EUID}" -eq 0 ]] || { log_error "请使用 sudo 运行本脚本"; exit 1; }; }

normalize_task() {
  local task="${1:-detection}"
  case "${task}" in
    detection|classification) echo "${task}" ;;
    *) log_warn "未知 TASK=${task}，回退为 detection"; echo "detection" ;;
  esac
}

normalize_input_size_comma() {
  local raw="${1:-}"
  raw="${raw//,/ }"
  raw="$(echo "$raw" | xargs || true)"
  local h="" w="" extra=""
  read -r h w extra <<< "$raw"
  if [[ -z "${h}" || -z "${w}" || -n "${extra}" ]]; then
    if [[ "${DEFAULT_TASK:-detection}" == "classification" ]]; then
      echo "224,224"
    else
      echo "640,640"
    fi
    return
  fi
  echo "${h},${w}"
}

load_generated_runtime_env() {
  if [[ -f "${GENERATED_ENV_FILE}" ]]; then
    log_info "检测到 runtime 配置: ${GENERATED_ENV_FILE}"
    # shellcheck disable=SC1090
    source "${GENERATED_ENV_FILE}"
    log_ok "已加载 runtime 配置"
  else
    log_warn "未找到 ${GENERATED_ENV_FILE}，继续使用脚本内默认值"
  fi
}

resolve_runtime_defaults() {
  DEFAULT_TASK="$(normalize_task "${TASK:-detection}")"
  DEFAULT_NPU_CORE="${NPU_CORE:-auto}"
  DEFAULT_NUM_CLASSES="${NUM_CLASSES:-2}"
  DEFAULT_PORT="${PORT:-8080}"
  DEFAULT_METRICS_PORT="${METRICS_PORT:-9091}"
  DEFAULT_INPUT_SIZE="$(normalize_input_size_comma "${INPUT_SIZE:-}")"
  DEFAULT_CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
  DEFAULT_NMS_THRESHOLD="${NMS_THRESHOLD:-0.45}"
  DEFAULT_TOPK="${TOPK:-5}"
  DEFAULT_WARMUP_RUNS="${WARMUP_RUNS:-3}"
  DEFAULT_REPORT_INTERVAL="${REPORT_INTERVAL:-${DEFAULT_REPORT_INTERVAL}}"

  if [[ -n "${CLASS_NAMES_FILE:-}" ]]; then
    DEFAULT_CLASS_NAMES_FILE="${CLASS_NAMES_FILE}"
  elif [[ "${DEFAULT_TASK}" == "classification" ]]; then
    DEFAULT_CLASS_NAMES_FILE="/opt/visionops/edge/runtime/class_names_classification.yaml"
  else
    DEFAULT_CLASS_NAMES_FILE="/opt/visionops/edge/runtime/class_names.yaml"
  fi
}

install_system_deps() {
  log_info "安装系统依赖..."
  apt-get update
  apt-get install -y python3 python3-pip python3-venv curl git wget rsync libopencv-dev libgomp1 systemd
}

create_dirs() {
  log_info "创建安装目录..."
  mkdir -p "${INSTALL_DIR}" "${MODEL_DIR}" "${LOG_DIR}" "${EDGE_DST_DIR}/runtime"
}

setup_python_env() {
  log_info "创建 Python 虚拟环境..."
  [[ -d "${VENV_DIR}" ]] || python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip wheel setuptools
  pip install fastapi "uvicorn[standard]" numpy opencv-python-headless requests psutil pyyaml python-multipart
  log_info "尝试安装 rknn-toolkit-lite2..."
  if ! pip install rknn-toolkit-lite2; then
    log_warn "rknn-toolkit-lite2 安装失败；板端真实推理必须安装成功"
  fi
}

deploy_edge_code() {
  log_info "部署边缘推理代码..."
  if [[ -d "${REPO_ROOT}/edge" ]]; then
    mkdir -p "${EDGE_DST_DIR}"
    rsync -a --delete "${REPO_ROOT}/edge/" "${EDGE_DST_DIR}/"
    log_ok "已同步 edge/ -> ${EDGE_DST_DIR}"
    return
  fi
  log_error "未找到 edge/ 代码"
  exit 1
}

prepare_runtime_class_names() {
  log_info "准备 runtime 类别文件..."
  local detection_file="${EDGE_DST_DIR}/runtime/class_names.yaml"
  local classification_file="${EDGE_DST_DIR}/runtime/class_names_classification.yaml"

  if [[ ! -f "${detection_file}" ]]; then
    log_warn "未找到 detection 类别文件，创建兜底文件: ${detection_file}"
    { echo "task: detection"; echo "num_classes: ${DEFAULT_NUM_CLASSES}"; echo "class_names:"; for ((i=0; i<DEFAULT_NUM_CLASSES; i++)); do echo "  - \"${i}\""; done; } > "${detection_file}"
  fi

  if [[ "${DEFAULT_TASK}" == "classification" ]]; then
    if [[ -f "${REPO_ROOT}/data/processed/class_names.yaml" ]]; then
      cp "${REPO_ROOT}/data/processed/class_names.yaml" "${classification_file}"
      log_ok "已复制分类类别文件 -> ${classification_file}"
    elif [[ ! -f "${classification_file}" ]]; then
      log_warn "未找到分类类别文件，创建兜底文件: ${classification_file}"
      { echo "task: classification"; echo "num_classes: ${DEFAULT_NUM_CLASSES}"; echo "class_names:"; for ((i=0; i<DEFAULT_NUM_CLASSES; i++)); do echo "  - \"${i}\""; done; echo "input_size: [224, 224]"; } > "${classification_file}"
    fi
  fi
}

check_runtime_lib() {
  log_info "检查 RKNN 运行时动态库..."
  if [[ -f "/usr/lib/librknnrt.so" ]]; then log_ok "已存在 /usr/lib/librknnrt.so"; return; fi
  if [[ -f "/tmp/librknnrt.so" ]]; then cp /tmp/librknnrt.so /usr/lib/librknnrt.so; ldconfig; log_ok "已安装 librknnrt.so"; return; fi
  log_warn "未找到 /usr/lib/librknnrt.so，请手动安装 RKNN runtime"
}

write_env_file() {
  log_info "写入环境配置: ${ENV_FILE}"
  cat > "${ENV_FILE}" <<EOF_ENV
DEVICE_ID=${DEVICE_ID}
SERVER_URL=${SERVER_URL}
MODEL_PATH=${DEFAULT_MODEL_PATH}
INFERENCE_URL=http://localhost:${DEFAULT_PORT}
REPORT_INTERVAL=${DEFAULT_REPORT_INTERVAL}
TASK=${DEFAULT_TASK}
NPU_CORE=${DEFAULT_NPU_CORE}
NUM_CLASSES=${DEFAULT_NUM_CLASSES}
INPUT_SIZE=${DEFAULT_INPUT_SIZE}
CLASS_NAMES_FILE=${DEFAULT_CLASS_NAMES_FILE}
PORT=${DEFAULT_PORT}
METRICS_PORT=${DEFAULT_METRICS_PORT}
CONF_THRESHOLD=${DEFAULT_CONF_THRESHOLD}
NMS_THRESHOLD=${DEFAULT_NMS_THRESHOLD}
TOPK=${DEFAULT_TOPK}
WARMUP_RUNS=${DEFAULT_WARMUP_RUNS}
EOF_ENV
  chmod 644 "${ENV_FILE}"
}

write_inference_service() {
  log_info "生成 systemd 推理服务..."
  cat > "${INFERENCE_SERVICE_PATH}" <<EOF_SERVICE
[Unit]
Description=VisionOps RK3588 Edge Inference Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${ENV_FILE}

ExecStart=${VENV_DIR}/bin/python ${EDGE_DST_DIR}/inference/engine.py \\
    --model \${MODEL_PATH} \\
    --task \${TASK} \\
    --host 0.0.0.0 \\
    --port \${PORT} \\
    --npu-core \${NPU_CORE} \\
    --num-classes \${NUM_CLASSES} \\
    --class-names-file \${CLASS_NAMES_FILE} \\
    --metrics-port \${METRICS_PORT} \\
    --conf-threshold \${CONF_THRESHOLD} \\
    --nms-threshold \${NMS_THRESHOLD} \\
    --topk \${TOPK} \\
    --warmup-runs \${WARMUP_RUNS}

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

write_monitor_service() {
  local monitor_py="${EDGE_DST_DIR}/monitor/monitor.py"
  if [[ -f "${monitor_py}" ]]; then
    cat > "${MONITOR_SERVICE_PATH}" <<EOF_MONITOR
[Unit]
Description=VisionOps RK3588 Edge Monitor Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${VENV_DIR}/bin/python ${monitor_py}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-monitor

[Install]
WantedBy=multi-user.target
EOF_MONITOR
  else
    rm -f "${MONITOR_SERVICE_PATH}" || true
  fi
}

prepare_placeholder_model() {
  if [[ ! -f "${DEFAULT_MODEL_PATH}" ]]; then
    echo "PLACEHOLDER_RKNN_MODEL" > "${DEFAULT_MODEL_PATH}"
    chmod 644 "${DEFAULT_MODEL_PATH}"
  fi
}

register_services() {
  systemctl daemon-reload
  systemctl enable visionops-inference.service
  [[ -f "${MONITOR_SERVICE_PATH}" ]] && systemctl enable visionops-monitor.service || true
}

fix_permissions() {
  chown -R root:root "${INSTALL_DIR}" || true
  if id ubuntu >/dev/null 2>&1; then chown -R ubuntu:ubuntu "${MODEL_DIR}" "${EDGE_DST_DIR}" "${LOG_DIR}" || true; fi
}

print_summary() {
  log_ok "VisionOps 边缘初始化完成：TASK=${DEFAULT_TASK}, INPUT_SIZE=${DEFAULT_INPUT_SIZE}, CLASS_NAMES_FILE=${DEFAULT_CLASS_NAMES_FILE}"
  log_info "健康检查: curl http://localhost:${DEFAULT_PORT}/health"
}

main() {
  ensure_root
  require_cmd python3; require_cmd pip3; require_cmd rsync
  load_generated_runtime_env
  resolve_runtime_defaults
  install_system_deps
  create_dirs
  setup_python_env
  deploy_edge_code
  prepare_runtime_class_names
  check_runtime_lib
  write_env_file
  write_inference_service
  write_monitor_service
  prepare_placeholder_model
  fix_permissions
  register_services
  print_summary
}

main "$@"
