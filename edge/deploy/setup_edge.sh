#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VisionOps Edge Setup Script (RK3588 / Detection)
# 用途：
# 1. 初始化板端运行环境
# 2. 部署 /opt/visionops/edge 代码
# 3. 生成 .env 与 systemd 服务
# 4. 为后续 push.sh 模型热部署做准备
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
DEFAULT_NPU_CORE="auto"
DEFAULT_NUM_CLASSES="6"
DEFAULT_PORT="8080"
DEFAULT_METRICS_PORT="9091"
DEFAULT_REPORT_INTERVAL="60"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log_info()  { echo "[INFO] $*"; }
log_ok()    { echo "[OK] $*"; }
log_warn()  { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log_error "缺少命令: $cmd"
    exit 1
  fi
}

ensure_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    log_error "请使用 sudo 运行本脚本"
    exit 1
  fi
}

install_system_deps() {
  log_info "安装系统依赖..."
  apt-get update
  apt-get install -y \
    python3 python3-pip python3-venv \
    curl git wget rsync \
    libopencv-dev libgomp1 \
    systemd
}

create_dirs() {
  log_info "创建安装目录..."
  mkdir -p "${INSTALL_DIR}"
  mkdir -p "${MODEL_DIR}"
  mkdir -p "${LOG_DIR}"
}

setup_python_env() {
  log_info "创建 Python 虚拟环境..."
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  log_info "升级 pip..."
  python -m pip install --upgrade pip wheel setuptools

  log_info "安装 Python 依赖..."
  pip install \
    fastapi \
    "uvicorn[standard]" \
    numpy \
    opencv-python-headless \
    requests \
    psutil \
    pyyaml \
    python-multipart

  log_info "尝试安装 rknn-toolkit-lite2..."
  if ! pip install rknn-toolkit-lite2; then
    log_warn "rknn-toolkit-lite2 安装失败，请后续手动检查"
  fi
}

deploy_edge_code() {
  log_info "部署边缘推理代码..."

  if [[ -d "${REPO_ROOT}/edge" ]]; then
    mkdir -p "${EDGE_DST_DIR}"
    rsync -a --delete "${REPO_ROOT}/edge/" "${EDGE_DST_DIR}/"
    log_ok "已从当前仓库同步 edge/ -> ${EDGE_DST_DIR}"
    return
  fi

  if [[ -f "/tmp/visionops_edge.tar.gz" ]]; then
    mkdir -p "${EDGE_DST_DIR}"
    tar -xzf /tmp/visionops_edge.tar.gz -C "${INSTALL_DIR}"
    log_ok "已从 /tmp/visionops_edge.tar.gz 解压边缘代码"
    return
  fi

  log_error "未找到 edge/ 代码。请确保当前仓库存在 edge/ 目录，或预先提供 /tmp/visionops_edge.tar.gz"
  exit 1
}

check_runtime_lib() {
  log_info "检查 RKNN 运行时动态库..."

  if [[ -f "/usr/lib/librknnrt.so" ]]; then
    log_ok "已存在 /usr/lib/librknnrt.so"
    return
  fi

  if [[ -f "/tmp/librknnrt.so" ]]; then
    cp /tmp/librknnrt.so /usr/lib/librknnrt.so
    ldconfig
    log_ok "已从 /tmp/librknnrt.so 安装到 /usr/lib/librknnrt.so"
    return
  fi

  log_warn "未找到 /usr/lib/librknnrt.so"
  log_warn "请手动将 librknnrt.so 放到 /usr/lib/ 后执行 sudo ldconfig"
}

write_env_file() {
  log_info "写入环境配置: ${ENV_FILE}"
  cat > "${ENV_FILE}" <<EOF
DEVICE_ID=${DEVICE_ID}
SERVER_URL=${SERVER_URL}
MODEL_PATH=${DEFAULT_MODEL_PATH}
INFERENCE_URL=http://localhost:${DEFAULT_PORT}
REPORT_INTERVAL=${DEFAULT_REPORT_INTERVAL}
NPU_CORE=${DEFAULT_NPU_CORE}
NUM_CLASSES=${DEFAULT_NUM_CLASSES}
EOF
  chmod 644 "${ENV_FILE}"
  log_ok "环境配置写入完成"
}

write_inference_service() {
  log_info "生成 systemd 推理服务..."

  cat > "${INFERENCE_SERVICE_PATH}" <<EOF
[Unit]
Description=VisionOps RK3588 Edge Inference Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}

Environment="DEVICE_ID=${DEVICE_ID}"
Environment="SERVER_URL=${SERVER_URL}"
Environment="MODEL_PATH=${DEFAULT_MODEL_PATH}"
Environment="NPU_CORE=${DEFAULT_NPU_CORE}"
Environment="NUM_CLASSES=${DEFAULT_NUM_CLASSES}"

ExecStart=${VENV_DIR}/bin/python ${EDGE_DST_DIR}/inference/engine.py \\
    --model \${MODEL_PATH} \\
    --host 0.0.0.0 \\
    --port ${DEFAULT_PORT} \\
    --npu-core \${NPU_CORE} \\
    --num-classes \${NUM_CLASSES}

Restart=always
RestartSec=5

TimeoutStartSec=30
TimeoutStopSec=10

StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-inference

[Install]
WantedBy=multi-user.target
EOF

  log_ok "推理服务文件已生成: ${INFERENCE_SERVICE_PATH}"
}

write_monitor_service() {
  log_info "生成 systemd 监控服务..."

  # 如果 monitor.py 不存在，就生成一个最小占位 service，避免 systemctl 注册失败
  local monitor_py="${EDGE_DST_DIR}/monitor/monitor.py"

  if [[ -f "${monitor_py}" ]]; then
    cat > "${MONITOR_SERVICE_PATH}" <<EOF
[Unit]
Description=VisionOps RK3588 Edge Monitor Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}

Environment="DEVICE_ID=${DEVICE_ID}"
Environment="SERVER_URL=${SERVER_URL}"
Environment="MODEL_PATH=${DEFAULT_MODEL_PATH}"
Environment="INFERENCE_URL=http://localhost:${DEFAULT_PORT}"
Environment="REPORT_INTERVAL=${DEFAULT_REPORT_INTERVAL}"

ExecStart=${VENV_DIR}/bin/python ${monitor_py}

Restart=always
RestartSec=10

StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-monitor

[Install]
WantedBy=multi-user.target
EOF
    log_ok "监控服务文件已生成: ${MONITOR_SERVICE_PATH}"
  else
    log_warn "未找到 ${monitor_py}，跳过监控服务生成"
    rm -f "${MONITOR_SERVICE_PATH}" || true
  fi
}

prepare_placeholder_model() {
  if [[ ! -f "${DEFAULT_MODEL_PATH}" ]]; then
    echo "PLACEHOLDER_RKNN_MODEL" > "${DEFAULT_MODEL_PATH}"
    chmod 644 "${DEFAULT_MODEL_PATH}"
    log_warn "占位模型已创建，等待服务器首次部署真实模型"
  else
    log_ok "已存在模型文件: ${DEFAULT_MODEL_PATH}"
  fi
}

register_services() {
  log_info "安装 systemd 服务..."
  systemctl daemon-reload
  systemctl enable visionops-inference.service

  if [[ -f "${MONITOR_SERVICE_PATH}" ]]; then
    systemctl enable visionops-monitor.service
  fi

  log_ok "systemd 服务已注册（默认开机自启）"
}

fix_permissions() {
  log_info "修正目录权限..."
  # 让 ubuntu 部署用户可读写模型与 edge 目录；如果设备没有 ubuntu 用户也不报错
  chown -R root:root "${INSTALL_DIR}" || true
  if id ubuntu >/dev/null 2>&1; then
    chown -R ubuntu:ubuntu "${MODEL_DIR}" "${EDGE_DST_DIR}" "${LOG_DIR}" || true
  fi
}

print_summary() {
  log_ok "=========================================="
  log_ok "VisionOps 边缘初始化完成！"
  log_ok "  设备ID: ${DEVICE_ID}"
  log_ok "  安装路径: ${INSTALL_DIR}"
  log_ok ""
  log_ok "后续操作："
  log_info "  1. 从训练机部署模型:"
  log_info "     bash edge/deploy/push.sh models/export_detection/model.rknn ${DEVICE_ID}"
  log_info "  2. 手动启动服务:"
  log_info "     sudo systemctl daemon-reload"
  log_info "     sudo systemctl restart visionops-inference"
  if [[ -f "${MONITOR_SERVICE_PATH}" ]]; then
    log_info "     sudo systemctl restart visionops-monitor"
  fi
  log_info "  3. 查看日志:"
  log_info "     journalctl -u visionops-inference -f"
  log_info "  4. 健康检查:"
  log_info "     curl http://localhost:${DEFAULT_PORT}/health"
  log_ok "=========================================="
}

main() {
  ensure_root
  require_cmd python3
  require_cmd pip3
  require_cmd rsync

  log_info "=== VisionOps 边缘初始化 ==="
  log_info "设备ID: ${DEVICE_ID}"
  log_info "服务器: ${SERVER_URL}"
  log_info "安装目录: ${INSTALL_DIR}"

  install_system_deps
  create_dirs
  setup_python_env
  deploy_edge_code
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