#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VisionOps Edge Setup Script for Fresh RK3588 Board
#
# 用法：
#   sudo bash setup_edge.sh [SERVER_URL] [DEVICE_ID]
#
# 示例：
#   sudo bash setup_edge.sh http://192.168.1.100:8000 rk3588-001
#
# 设计原则：
# 1. 面向“全新板子”初始化，优先保证可部署、可维护、可快速接入
# 2. 只部署边缘运行所需组件，不在板端安装训练/MLOps整套环境
# 3. systemd 默认创建并 enable，但不强制 start 推理服务
#    （因为首次初始化时通常还没有真实的 current.rknn）
# 4. 支持 detection / classification
# 5. INPUT_SIZE 使用逗号格式，例如 224,224 / 640,640
# ============================================================

SERVER_URL="${1:-http://127.0.0.1:8000}"
DEVICE_ID="${2:-rk3588-001}"

INSTALL_DIR="/opt/visionops"
VENV_DIR="${INSTALL_DIR}/venv"
EDGE_DST_DIR="${INSTALL_DIR}/edge"
MODEL_DIR="${INSTALL_DIR}/models"
LOG_DIR="${INSTALL_DIR}/logs"
SCRIPT_DST_DIR="${INSTALL_DIR}/scripts"
RUNTIME_DIR="${EDGE_DST_DIR}/runtime"
ENV_FILE="${INSTALL_DIR}/.env"

INFERENCE_SERVICE_PATH="/etc/systemd/system/visionops-inference.service"
MONITOR_SERVICE_PATH="/etc/systemd/system/visionops-monitor.service"

DEFAULT_MODEL_PATH="${MODEL_DIR}/current.rknn"
DEFAULT_REPORT_INTERVAL="60"
DEFAULT_INSTALL_USER="${SUDO_USER:-ubuntu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GENERATED_ENV_FILE="${REPO_ROOT}/edge/runtime/edge.env"

log_info()  { echo "[INFO] $*"; }
log_ok()    { echo "[OK] $*"; }
log_warn()  { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    log_error "缺少命令: $1"
    exit 1
  }
}

ensure_root() {
  [[ "${EUID}" -eq 0 ]] || {
    log_error "请使用 sudo 运行本脚本"
    exit 1
  }
}

user_exists() {
  id "$1" >/dev/null 2>&1
}

get_run_user() {
  if user_exists "${DEFAULT_INSTALL_USER}"; then
    echo "${DEFAULT_INSTALL_USER}"
  elif user_exists "ubuntu"; then
    echo "ubuntu"
  else
    echo "root"
  fi
}

RUN_USER="$(get_run_user)"

normalize_task() {
  local task="${1:-detection}"
  case "${task}" in
    detection|classification)
      echo "${task}"
      ;;
    *)
      log_warn "未知 TASK=${task}，回退为 detection"
      echo "detection"
      ;;
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
    log_warn "未找到 ${GENERATED_ENV_FILE}，继续使用脚本默认值"
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
    DEFAULT_CLASS_NAMES_FILE="${RUNTIME_DIR}/class_names_classification.yaml"
  else
    DEFAULT_CLASS_NAMES_FILE="${RUNTIME_DIR}/class_names.yaml"
  fi
}

install_system_deps() {
  log_info "安装系统依赖..."
  export DEBIAN_FRONTEND=noninteractive

  apt-get update
  apt-get install -y \
    python3 python3-pip python3-venv \
    curl wget git rsync vim tmux \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libjpeg-dev zlib1g-dev libgomp1 \
    net-tools iproute2 \
    openssh-client ca-certificates

  log_ok "系统依赖安装完成"
}

create_dirs() {
  log_info "创建安装目录..."
  mkdir -p "${INSTALL_DIR}"
  mkdir -p "${MODEL_DIR}" "${LOG_DIR}" "${SCRIPT_DST_DIR}" "${RUNTIME_DIR}"
  log_ok "目录创建完成"
}

setup_python_env() {
  log_info "创建 Python 虚拟环境..."
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  python -m pip install --upgrade pip setuptools wheel

  log_info "安装边缘运行依赖..."
  pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    requests \
    pyyaml \
    numpy \
    psutil \
    opencv-python-headless

  log_info "尝试安装 rknn-toolkit-lite2..."
  if pip install --no-cache-dir rknn-toolkit-lite2; then
    log_ok "rknn-toolkit-lite2 安装成功"
  else
    log_warn "rknn-toolkit-lite2 安装失败"
    log_warn "若板端推理需要真实运行，请手动安装匹配 RK3588 / Python 版本的 whl"
  fi
}

deploy_edge_code() {
  log_info "同步边缘代码..."
  if [[ ! -d "${REPO_ROOT}/edge" ]]; then
    log_error "未找到仓库中的 edge/ 目录: ${REPO_ROOT}/edge"
    exit 1
  fi

  rsync -a --delete "${REPO_ROOT}/edge/" "${EDGE_DST_DIR}/"
  log_ok "已同步 edge/ -> ${EDGE_DST_DIR}"
}

copy_helper_scripts() {
  log_info "复制部署辅助脚本..."
  mkdir -p "${SCRIPT_DST_DIR}"

  if [[ -f "${SCRIPT_DIR}/push.sh" ]]; then
    cp -f "${SCRIPT_DIR}/push.sh" "${SCRIPT_DST_DIR}/push.sh"
    chmod +x "${SCRIPT_DST_DIR}/push.sh"
  fi

  cp -f "${BASH_SOURCE[0]}" "${SCRIPT_DST_DIR}/setup_edge.sh"
  chmod +x "${SCRIPT_DST_DIR}/setup_edge.sh"

  log_ok "辅助脚本已复制到 ${SCRIPT_DST_DIR}"
}

prepare_runtime_class_names() {
  log_info "准备 runtime 类别文件..."

  local detection_file="${RUNTIME_DIR}/class_names.yaml"
  local classification_file="${RUNTIME_DIR}/class_names_classification.yaml"

  if [[ ! -f "${detection_file}" ]]; then
    log_warn "未找到 detection 类别文件，创建兜底文件: ${detection_file}"
    {
      echo "task: detection"
      echo "num_classes: ${DEFAULT_NUM_CLASSES}"
      echo "class_names:"
      for ((i=0; i<DEFAULT_NUM_CLASSES; i++)); do
        echo "  - \"${i}\""
      done
      echo "input_size: [640, 640]"
    } > "${detection_file}"
  fi

  if [[ "${DEFAULT_TASK}" == "classification" ]]; then
    if [[ -f "${REPO_ROOT}/data/processed/class_names.yaml" ]]; then
      cp -f "${REPO_ROOT}/data/processed/class_names.yaml" "${classification_file}"
      log_ok "已复制分类类别文件 -> ${classification_file}"
    elif [[ ! -f "${classification_file}" ]]; then
      log_warn "未找到分类类别文件，创建兜底文件: ${classification_file}"
      {
        echo "task: classification"
        echo "num_classes: ${DEFAULT_NUM_CLASSES}"
        echo "class_names:"
        for ((i=0; i<DEFAULT_NUM_CLASSES; i++)); do
          echo "  - \"${i}\""
        done
        echo "input_size: [224, 224]"
      } > "${classification_file}"
    fi
  fi
}

check_runtime_lib() {
  log_info "检查 RKNN 运行时动态库..."

  if [[ -f "/usr/lib/librknnrt.so" || -f "/usr/lib/aarch64-linux-gnu/librknnrt.so" ]]; then
    log_ok "已检测到 librknnrt.so"
    return
  fi

  if [[ -f "/tmp/librknnrt.so" ]]; then
    cp -f /tmp/librknnrt.so /usr/lib/librknnrt.so
    ldconfig || true
    log_ok "已从 /tmp 安装 librknnrt.so"
    return
  fi

  log_warn "未检测到 librknnrt.so"
  log_warn "如果后续推理时报 RKNN runtime 相关错误，请手动安装 Rockchip runtime 动态库"
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
  log_ok ".env 写入完成"
}

write_inference_service() {
  log_info "生成 systemd 推理服务..."

  cat > "${INFERENCE_SERVICE_PATH}" <<EOF_SERVICE
[Unit]
Description=VisionOps RK3588 Edge Inference Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${INSTALL_DIR}
Environment=PYTHONUNBUFFERED=1
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
TimeoutStartSec=60
TimeoutStopSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-inference

[Install]
WantedBy=multi-user.target
EOF_SERVICE

  log_ok "已生成 ${INFERENCE_SERVICE_PATH}"
}

write_monitor_service() {
  local monitor_py="${EDGE_DST_DIR}/monitor/monitor.py"

  if [[ -f "${monitor_py}" ]]; then
    log_info "生成 systemd 监控服务..."

    cat > "${MONITOR_SERVICE_PATH}" <<EOF_MONITOR
[Unit]
Description=VisionOps RK3588 Edge Monitor Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${INSTALL_DIR}
Environment=PYTHONUNBUFFERED=1
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

    log_ok "已生成 ${MONITOR_SERVICE_PATH}"
  else
    log_warn "未找到 ${monitor_py}，跳过 monitor 服务生成"
    rm -f "${MONITOR_SERVICE_PATH}" || true
  fi
}

prepare_placeholder_model() {
  if [[ ! -f "${DEFAULT_MODEL_PATH}" ]]; then
    log_warn "未检测到真实模型，创建占位文件: ${DEFAULT_MODEL_PATH}"
    echo "PLACEHOLDER_RKNN_MODEL" > "${DEFAULT_MODEL_PATH}"
    chmod 644 "${DEFAULT_MODEL_PATH}"
  fi
}

fix_permissions() {
  log_info "修正目录权限..."
  chown -R "${RUN_USER}:${RUN_USER}" "${INSTALL_DIR}" || true
  chmod -R u+rwX "${INSTALL_DIR}" || true
  log_ok "权限修正完成"
}

register_services() {
  log_info "注册 systemd 服务..."
  systemctl daemon-reload
  systemctl enable visionops-inference.service

  if [[ -f "${MONITOR_SERVICE_PATH}" ]]; then
    systemctl enable visionops-monitor.service
  fi

  log_ok "systemd 服务注册完成"
}

print_summary() {
  cat <<EOF_SUMMARY

============================================================
VisionOps Edge 初始化完成
============================================================
安装目录:        ${INSTALL_DIR}
运行用户:        ${RUN_USER}
设备ID:          ${DEVICE_ID}
服务端地址:      ${SERVER_URL}
任务类型:        ${DEFAULT_TASK}
输入尺寸:        ${DEFAULT_INPUT_SIZE}
模型路径:        ${DEFAULT_MODEL_PATH}
类别文件:        ${DEFAULT_CLASS_NAMES_FILE}

systemd 服务:
  - visionops-inference.service
  - visionops-monitor.service (如果 monitor.py 存在)

注意：
1. 当前脚本不会自动启动推理服务，避免因为占位模型导致启动失败
2. 请先把真实模型推到:
   ${DEFAULT_MODEL_PATH}
3. 然后手动启动:
   sudo systemctl start visionops-inference
4. 查看状态:
   sudo systemctl status visionops-inference
5. 健康检查:
   curl http://localhost:${DEFAULT_PORT}/health

若要测试 RTSP:
  ffprobe -rtsp_transport tcp "rtsp://admin:Abcd123_@192.168.2.64:554/Streaming/Channels/101"

EOF_SUMMARY
}

main() {
  ensure_root
  require_cmd python3
  require_cmd pip3
  require_cmd rsync
  require_cmd git

  load_generated_runtime_env
  resolve_runtime_defaults

  install_system_deps
  create_dirs
  setup_python_env
  deploy_edge_code
  copy_helper_scripts
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