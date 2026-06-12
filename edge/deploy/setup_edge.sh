#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VisionOps LB3576 + HP60C SDK + C++ RKNN + Modbus TCP 一键部署脚本
#
# 用法：
#   sudo bash setup_edge.sh [SERVER_URL] [DEVICE_ID]
#
# 示例：
#   sudo bash setup_edge.sh http://192.168.1.100:8000 lb3576-211
#
# 目标服务：
#   - visionops-collector.service              Web / Collector，默认 8090
#   - visionops-hp60c-sdk-bridge.service       HP60C C++ SDK 取图，默认 18181
#   - visionops-inference-cpp.service          C++ RKNN 推理，默认 18080
#   - visionops-tube-station.service           Modbus TCP 触发检测，默认 1502
#
# 不再安装/启用旧服务：
#   - visionops-inference.service              旧 Python 推理服务
#   - visionops-monitor.service                旧 Monitor 服务
#   - visionops-hp60c-ros1-bridge.service      旧 ROS1 bridge
#
# 常用可覆盖环境变量：
#   INSTALL_DIR=/opt/visionops
#   HP60C_SDK_ROOT=/home/neardi/AngstrongCameraSdk_v1.2.61.20250910/demo/linux_ros
#   VISIONOPS_START_SERVICES=1
#   VISIONOPS_TUBE_MODBUS_PORT=1502
#   VISIONOPS_CPP_MODEL_PATH=/opt/visionops/models/current.rknn
#   VISIONOPS_CPP_CLASS_NAMES_FILE=/opt/visionops/models/current.yaml
# ============================================================

SERVER_URL="${1:-http://127.0.0.1:8000}"
DEVICE_ID="${2:-lb3576-001}"

INSTALL_DIR="${INSTALL_DIR:-/opt/visionops}"
VENV_DIR="${INSTALL_DIR}/venv"
EDGE_DST_DIR="${INSTALL_DIR}/edge"
MODEL_DIR="${INSTALL_DIR}/models"
LOG_DIR="${INSTALL_DIR}/logs"
RUNTIME_DIR="${EDGE_DST_DIR}/runtime"
SCRIPT_DST_DIR="${INSTALL_DIR}/scripts"

COLLECTOR_SERVICE="/etc/systemd/system/visionops-collector.service"
CPP_SERVICE="/etc/systemd/system/visionops-inference-cpp.service"
HP60C_SERVICE_NAME="visionops-hp60c-sdk-bridge.service"
TUBE_SERVICE_NAME="visionops-tube-station.service"

CPP_ENV="${RUNTIME_DIR}/cpp.env"
COLLECTOR_ENV="${RUNTIME_DIR}/collector.env"
RUNTIME_OVERRIDES="${RUNTIME_DIR}/runtime_overrides.yaml"

HP60C_SDK_ROOT="${HP60C_SDK_ROOT:-/home/neardi/AngstrongCameraSdk_v1.2.61.20250910/demo/linux_ros}"
HP60C_SDK_LIB_DIR="${HP60C_SDK_LIB_DIR:-${HP60C_SDK_ROOT}/libs/lib/aarch64-linux-gnu-gcc-5}"
HP60C_CONFIG="${HP60C_CONFIG:-${HP60C_SDK_ROOT}/configurationfiles/hp60c_v2_01_20241104_configEncrypt.json}"

VISIONOPS_START_SERVICES="${VISIONOPS_START_SERVICES:-1}"
VISIONOPS_TUBE_MODBUS_PORT="${VISIONOPS_TUBE_MODBUS_PORT:-1502}"
VISIONOPS_COLLECTOR_PORT="${VISIONOPS_COLLECTOR_PORT:-8090}"
VISIONOPS_CPP_PORT="${VISIONOPS_CPP_PORT:-18080}"
VISIONOPS_HP60C_PORT="${VISIONOPS_HP60C_PORT:-18181}"

DEFAULT_INSTALL_USER="${SUDO_USER:-neardi}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log_info()  { echo "[INFO] $*"; }
log_ok()    { echo "[OK] $*"; }
log_warn()  { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

ensure_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    log_error "请使用 sudo 运行本脚本"
    exit 1
  fi
}

user_exists() { id "$1" >/dev/null 2>&1; }

get_run_user() {
  if user_exists "${DEFAULT_INSTALL_USER}"; then
    echo "${DEFAULT_INSTALL_USER}"
  elif user_exists "neardi"; then
    echo "neardi"
  elif user_exists "ubuntu"; then
    echo "ubuntu"
  else
    echo "root"
  fi
}

RUN_USER="$(get_run_user)"

run_as_user() {
  local cmd="$*"
  if [[ "${RUN_USER}" == "root" ]]; then
    bash -lc "${cmd}"
  else
    sudo -H -u "${RUN_USER}" bash -lc "${cmd}"
  fi
}

install_system_deps() {
  log_info "安装系统依赖..."
  export DEBIAN_FRONTEND=noninteractive

  apt-get update
  apt-get install -y \
    build-essential cmake pkg-config git rsync \
    wget curl unzip tar vim nano tmux \
    python3 python3-pip python3-venv python3-dev python3-yaml python3-numpy \
    ffmpeg v4l-utils tcpdump iputils-ping openssh-server \
    net-tools iproute2 htop ca-certificates \
    libopencv-dev libgomp1 libyaml-cpp-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libjpeg-dev zlib1g-dev

  log_ok "系统依赖安装完成"
}

create_dirs() {
  log_info "创建目录..."
  mkdir -p "${INSTALL_DIR}" "${MODEL_DIR}" "${LOG_DIR}" "${SCRIPT_DST_DIR}" "${RUNTIME_DIR}"
  log_ok "目录创建完成"
}

setup_python_env() {
  log_info "创建/更新 Python venv: ${VENV_DIR}"
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --no-cache-dir \
    fastapi "uvicorn[standard]" python-multipart \
    requests pyyaml numpy psutil \
    "pymodbus>=2.5,<4" \
    opencv-python-headless \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

  log_ok "Python 运行依赖安装完成"
}

deploy_edge_code() {
  log_info "同步 edge 代码..."
  if [[ ! -d "${REPO_ROOT}/edge" ]]; then
    log_error "未找到仓库 edge/ 目录: ${REPO_ROOT}/edge"
    exit 1
  fi

  rsync -a --delete "${REPO_ROOT}/edge/" "${EDGE_DST_DIR}/"
  cp -f "${BASH_SOURCE[0]}" "${SCRIPT_DST_DIR}/setup_edge.sh"
  chmod +x "${SCRIPT_DST_DIR}/setup_edge.sh"
  log_ok "edge/ 已同步到 ${EDGE_DST_DIR}"
}

cleanup_legacy_services() {
  log_info "彻底清理旧 Python/ROS 相机与旧推理服务..."

  local legacy_services=(
    visionops-inference.service
    visionops-monitor.service
    visionops-hp60c-ros1-bridge.service
  )

  for svc in "${legacy_services[@]}"; do
    log_info "清理旧服务: ${svc}"

    systemctl disable --now "${svc}" 2>/dev/null || true
    systemctl stop "${svc}" 2>/dev/null || true
    systemctl reset-failed "${svc}" 2>/dev/null || true

    # 删除常见 systemd unit 文件、软链接、target wants/requires 残留
    find /etc/systemd /run/systemd /lib/systemd /usr/lib/systemd \
      \( -name "${svc}" -o -lname "*${svc}" \) \
      -exec rm -f {} \; 2>/dev/null || true
  done

  # 兼容极少数 SysV init 残留，避免 systemd generator 重新生成 unit
  rm -f /etc/init.d/visionops-inference /etc/init.d/visionops-monitor 2>/dev/null || true

  systemctl daemon-reload
  systemctl reset-failed "${legacy_services[@]}" 2>/dev/null || true

  log_ok "旧服务清理完成"
}

write_collector_runtime_files() {
  log_info "写入 Collector runtime，禁用旧 Python RTSP/USB camera_service..."

  cat > "${COLLECTOR_ENV}" <<EOF
# Auto-generated by setup_edge.sh
# Legacy Python camera_service disabled. Use HP60C C++ SDK bridge instead.
VISIONOPS_CAMERA_SOURCE="browser"
VISIONOPS_COLLECTOR_HOST="0.0.0.0"
VISIONOPS_COLLECTOR_PORT="${VISIONOPS_COLLECTOR_PORT}"
EOF

  cat > "${RUNTIME_OVERRIDES}" <<EOF
# Auto-generated by setup_edge.sh
device:
  id: "${DEVICE_ID}"
server:
  url: "${SERVER_URL}"
camera:
  # Legacy Python RTSP/USB camera path disabled.
  # Use C++ settings: HP60C C++ SDK 相机 -> http://127.0.0.1:${VISIONOPS_HP60C_PORT}/stream.mjpeg
  type: disabled
  common:
    preview_width: 960
    jpeg_quality: 100
vision_box:
  device_id: "${DEVICE_ID}"
  server_url: "${SERVER_URL}"
EOF

  log_ok "Collector runtime 已写入"
}

auto_pick_model() {
  local model_path="${VISIONOPS_CPP_MODEL_PATH:-${MODEL_DIR}/current.rknn}"
  local class_file="${VISIONOPS_CPP_CLASS_NAMES_FILE:-${MODEL_DIR}/current.yaml}"

  if [[ ! -f "${model_path}" ]]; then
    local candidate
    candidate="$(find "${MODEL_DIR}" -maxdepth 1 -type f -name "*.rknn" | sort | head -n 1 || true)"
    if [[ -n "${candidate}" ]]; then
      model_path="${candidate}"
    fi
  fi

  if [[ ! -f "${class_file}" ]]; then
    local base="${model_path%.rknn}"
    if [[ -f "${base}.yaml" ]]; then
      class_file="${base}.yaml"
    else
      local y
      y="$(find "${MODEL_DIR}" -maxdepth 1 -type f \( -name "*.yaml" -o -name "*.yml" \) | sort | head -n 1 || true)"
      if [[ -n "${y}" ]]; then
        class_file="${y}"
      fi
    fi
  fi

  echo "${model_path}|${class_file}"
}

write_cpp_env() {
  log_info "写入 C++ 推理运行配置: ${CPP_ENV}"
  local picked model_path class_file
  picked="$(auto_pick_model)"
  model_path="${picked%%|*}"
  class_file="${picked##*|}"

  cat > "${CPP_ENV}" <<EOF
# Auto-generated by setup_edge.sh
DEVICE_ID=${DEVICE_ID}
SERVER_URL=${SERVER_URL}

VISIONOPS_CPP_BIN=/opt/visionops/bin/visionops_inference_cpp
VISIONOPS_CPP_PORT=${VISIONOPS_CPP_PORT}

# Model
VISIONOPS_CPP_MODEL_PATH=${model_path}
VISIONOPS_CPP_CLASS_NAMES_FILE=${class_file}
VISIONOPS_CPP_TASK=${VISIONOPS_CPP_TASK:-detection}
VISIONOPS_CPP_NUM_CLASSES=${VISIONOPS_CPP_NUM_CLASSES:-2}
VISIONOPS_CPP_INPUT_SIZE=${VISIONOPS_CPP_INPUT_SIZE:-640,640}

# Thresholds
VISIONOPS_CPP_CONF_THRESHOLD=${VISIONOPS_CPP_CONF_THRESHOLD:-0.25}
VISIONOPS_CPP_NMS_THRESHOLD=${VISIONOPS_CPP_NMS_THRESHOLD:-0.45}
VISIONOPS_CPP_MAX_DET=${VISIONOPS_CPP_MAX_DET:-100}
VISIONOPS_CPP_TOPK=${VISIONOPS_CPP_TOPK:-5}
VISIONOPS_CPP_MASK_THRESHOLD=${VISIONOPS_CPP_MASK_THRESHOLD:-0.5}

# HP60C SDK bridge as C++ camera source
VISIONOPS_CPP_UI_CAMERA_TYPE=hp60c_sdk
VISIONOPS_CPP_CAMERA_TYPE=auto
VISIONOPS_CPP_CAMERA_SOURCE=http://127.0.0.1:${VISIONOPS_HP60C_PORT}/stream.mjpeg
VISIONOPS_CAMERA_SOURCE=http://127.0.0.1:${VISIONOPS_HP60C_PORT}/stream.mjpeg
VISIONOPS_CPP_STREAM_BACKEND=opencv
VISIONOPS_HP60C_SDK_BRIDGE_URL=http://127.0.0.1:${VISIONOPS_HP60C_PORT}
VISIONOPS_CPP_HP60C_SDK_BRIDGE_URL=http://127.0.0.1:${VISIONOPS_HP60C_PORT}

# Camera / stream
VISIONOPS_CPP_CAMERA_READ_FPS=${VISIONOPS_CPP_CAMERA_READ_FPS:-10}
VISIONOPS_CPP_DETECT_FPS=${VISIONOPS_CPP_DETECT_FPS:-1}
VISIONOPS_CPP_SNAPSHOT_FPS=${VISIONOPS_CPP_SNAPSHOT_FPS:-10}
VISIONOPS_CPP_STREAM_AUTO_START=${VISIONOPS_CPP_STREAM_AUTO_START:-False}
VISIONOPS_CPP_ENABLE_SNAPSHOT=True
VISIONOPS_CPP_ENABLE_ANNOTATED=True
VISIONOPS_CPP_QUIET_FFMPEG_LOG=True
VISIONOPS_CPP_CAMERA_BUFFER_SIZE=1
VISIONOPS_CPP_RTSP_TRANSPORT=tcp
VISIONOPS_CPP_RTSP_TIMEOUT_MS=5000
VISIONOPS_CPP_GST_LATENCY_MS=100
VISIONOPS_CPP_STREAM_CODEC=h264

# RKNN / preprocess
VISIONOPS_CPP_NPU_CORE=${VISIONOPS_CPP_NPU_CORE:-auto}
VISIONOPS_CPP_PREPROCESS_BACKEND=${VISIONOPS_CPP_PREPROCESS_BACKEND:-auto}
VISIONOPS_CPP_RGA_MODE=${VISIONOPS_CPP_RGA_MODE:-resize_color}
VISIONOPS_CPP_OUTPUT_MODE=${VISIONOPS_CPP_OUTPUT_MODE:-float}

# Compatibility placeholders
VISIONOPS_CPP_CAMERA_WIDTH=0
VISIONOPS_CPP_CAMERA_HEIGHT=0
VISIONOPS_CPP_CAMERA_FPS=0
VISIONOPS_CPP_CAMERA_FOURCC=''
VISIONOPS_CPP_PIPELINE_CONFIG=''
EOF

  log_ok "cpp.env 已写入"
  if [[ ! -f "${model_path}" ]]; then
    log_warn "当前模型不存在: ${model_path}，C++ 推理服务可能无法启动"
  fi
  if [[ ! -f "${class_file}" ]]; then
    log_warn "当前类别文件不存在: ${class_file}，C++ 推理服务可能无法启动"
  fi
}

check_rknn_runtime() {
  log_info "检查 RKNN runtime..."
  local has_header=0 has_lib=0
  [[ -f /usr/include/rknn_api.h || -f /usr/local/include/rknn_api.h ]] && has_header=1
  [[ -f /usr/lib/librknnrt.so || -f /usr/lib/aarch64-linux-gnu/librknnrt.so || -f /usr/local/lib/librknnrt.so ]] && has_lib=1

  if [[ "${has_header}" -eq 1 ]]; then
    log_ok "检测到 rknn_api.h"
  else
    log_warn "未检测到 rknn_api.h。若编译失败，请从 RKNN runtime / rknpu2 SDK 安装到 /usr/include/rknn_api.h"
  fi

  if [[ "${has_lib}" -eq 1 ]]; then
    log_ok "检测到 librknnrt.so"
  else
    log_warn "未检测到 librknnrt.so。若运行失败，请安装 RKNN runtime 动态库并执行 ldconfig"
  fi
}

build_cpp_inference() {
  local src="${EDGE_DST_DIR}/inference_cpp"
  if [[ ! -d "${src}" || ! -f "${src}/CMakeLists.txt" ]]; then
    log_warn "未找到 C++ 推理源码 ${src}，跳过编译"
    return
  fi

  log_info "编译 C++ RKNN 推理服务..."
  cmake -S "${src}" -B "${src}/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${src}/build" -j"$(nproc)"
  cmake --install "${src}/build" --prefix "${INSTALL_DIR}" || true

  if [[ -f "${src}/build/visionops_inference_cpp" && ! -f "${INSTALL_DIR}/bin/visionops_inference_cpp" ]]; then
    mkdir -p "${INSTALL_DIR}/bin"
    cp -f "${src}/build/visionops_inference_cpp" "${INSTALL_DIR}/bin/visionops_inference_cpp"
  fi

  if [[ -f "${INSTALL_DIR}/bin/visionops_inference_cpp" ]]; then
    chmod 755 "${INSTALL_DIR}/bin/visionops_inference_cpp"
    log_ok "C++ 推理二进制已安装: ${INSTALL_DIR}/bin/visionops_inference_cpp"
  else
    log_warn "未找到 ${INSTALL_DIR}/bin/visionops_inference_cpp，请检查 CMake install 规则"
  fi
}

install_cpp_service() {
  log_info "安装 visionops-inference-cpp.service"

  if [[ -f "${EDGE_DST_DIR}/deploy/visionops-inference-cpp.service" ]]; then
    cp -f "${EDGE_DST_DIR}/deploy/visionops-inference-cpp.service" "${CPP_SERVICE}"
  else
    cat > "${CPP_SERVICE}" <<EOF
[Unit]
Description=VisionOps RK3588 C++ RKNN Inference Service
After=network-online.target ${HP60C_SERVICE_NAME}
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=-${CPP_ENV}
ExecStart=${EDGE_DST_DIR}/inference_cpp/scripts/start_visionops_inference_cpp.sh
Restart=always
RestartSec=3
TimeoutStartSec=30
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-inference-cpp

[Install]
WantedBy=multi-user.target
EOF
  fi

  chmod 644 "${CPP_SERVICE}"
  log_ok "已安装 ${CPP_SERVICE}"
}

install_hp60c_sdk_bridge() {
  local bridge_dir="${EDGE_DST_DIR}/robot_gateway/hp60c_sdk_bridge"
  if [[ ! -d "${bridge_dir}" ]]; then
    log_warn "未找到 HP60C SDK bridge 目录: ${bridge_dir}，跳过"
    return
  fi

  log_info "配置 HP60C SDK bridge..."

  cat > "${bridge_dir}/hp60c_sdk_bridge.env" <<EOF
# Auto-generated by setup_edge.sh
VISIONOPS_HP60C_SDK_ROOT=${HP60C_SDK_ROOT}
VISIONOPS_HP60C_SDK_LIB_DIR=${HP60C_SDK_LIB_DIR}
VISIONOPS_HP60C_CONFIG=${HP60C_CONFIG}
VISIONOPS_HP60C_HTTP_HOST=127.0.0.1
VISIONOPS_HP60C_HTTP_PORT=${VISIONOPS_HP60C_PORT}
VISIONOPS_HP60C_FLIP_VERTICAL=${VISIONOPS_HP60C_FLIP_VERTICAL:-true}
VISIONOPS_HP60C_RGB_ORDER=${VISIONOPS_HP60C_RGB_ORDER:-bgr}
VISIONOPS_HP60C_JPEG_QUALITY=${VISIONOPS_HP60C_JPEG_QUALITY:-90}
EOF

  if [[ -f "${HP60C_SDK_ROOT}/linux/scripts/create_udev_rules.sh" ]]; then
    log_info "安装 HP60C udev 权限规则..."
    bash "${HP60C_SDK_ROOT}/linux/scripts/create_udev_rules.sh" || true
    udevadm control --reload-rules || true
    udevadm trigger || true
  else
    log_warn "未找到 SDK udev 脚本: ${HP60C_SDK_ROOT}/linux/scripts/create_udev_rules.sh"
  fi

  if [[ -x "${bridge_dir}/install_hp60c_sdk_bridge_service.sh" || -f "${bridge_dir}/install_hp60c_sdk_bridge_service.sh" ]]; then
    bash "${bridge_dir}/install_hp60c_sdk_bridge_service.sh"
  else
    log_warn "未找到 install_hp60c_sdk_bridge_service.sh，无法自动安装 SDK bridge service"
  fi
}

install_tube_station_service() {
  local tube_dir="${EDGE_DST_DIR}/robot_gateway/tube_station"
  if [[ ! -d "${tube_dir}" ]]; then
    log_warn "未找到 tube_station 目录: ${tube_dir}，跳过"
    return
  fi

  log_info "配置 tube_station Modbus TCP 触发检测服务..."

  cat > "${tube_dir}/tube_station.env" <<EOF
# Auto-generated by setup_edge.sh
VISIONOPS_TUBE_MODBUS_HOST=0.0.0.0
VISIONOPS_TUBE_MODBUS_PORT=${VISIONOPS_TUBE_MODBUS_PORT}
VISIONOPS_TUBE_MODBUS_UNIT_ID=${VISIONOPS_TUBE_MODBUS_UNIT_ID:-1}
VISIONOPS_TUBE_ADDRESS_BASE=${VISIONOPS_TUBE_ADDRESS_BASE:-0}

VISIONOPS_TUBE_RESULT_MODE=infer_once
VISIONOPS_TUBE_SNAPSHOT_URL=http://127.0.0.1:${VISIONOPS_HP60C_PORT}/stream/snapshot.jpg
VISIONOPS_TUBE_INFER_URL=http://127.0.0.1:${VISIONOPS_COLLECTOR_PORT}/api/cpp/infer

VISIONOPS_TUBE_STAND_CLASS_IDS=${VISIONOPS_TUBE_STAND_CLASS_IDS:-0}
VISIONOPS_TUBE_LYING_CLASS_IDS=${VISIONOPS_TUBE_LYING_CLASS_IDS:-1}
VISIONOPS_TUBE_MIN_CONF=${VISIONOPS_TUBE_MIN_CONF:-0.25}
VISIONOPS_TUBE_TRIGGER_TIMEOUT_MS=${VISIONOPS_TUBE_TRIGGER_TIMEOUT_MS:-5000}
EOF

  if [[ -f "${tube_dir}/install_tube_station_service.sh" ]]; then
    bash "${tube_dir}/install_tube_station_service.sh"
  else
    cat > "/etc/systemd/system/${TUBE_SERVICE_NAME}" <<EOF
[Unit]
Description=VisionOps Tube Station Modbus TCP Trigger Service
After=network-online.target ${HP60C_SERVICE_NAME} visionops-collector.service visionops-inference-cpp.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${tube_dir}
Environment=VISIONOPS_TUBE_ENV=${tube_dir}/tube_station.env
ExecStart=${VENV_DIR}/bin/python ${tube_dir}/tube_station_modbus_tcp.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-tube-station

[Install]
WantedBy=multi-user.target
EOF
  fi

  log_ok "tube_station 服务已配置"
}

install_collector_service() {
  local collector_dir="${EDGE_DST_DIR}/collector"
  if [[ ! -d "${collector_dir}" || ! -f "${collector_dir}/start_collector.sh" ]]; then
    log_warn "未找到 Collector 或 start_collector.sh，跳过 Collector systemd service"
    return
  fi

  log_info "安装 visionops-collector.service"

  chmod +x "${collector_dir}/start_collector.sh"

  cat > "${COLLECTOR_SERVICE}" <<EOF
[Unit]
Description=VisionOps Collector Web Service
After=network-online.target ${HP60C_SERVICE_NAME}
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${collector_dir}
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-${COLLECTOR_ENV}
ExecStart=/bin/bash ${collector_dir}/start_collector.sh
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-collector

[Install]
WantedBy=multi-user.target
EOF

  chmod 644 "${COLLECTOR_SERVICE}"
  log_ok "已安装 ${COLLECTOR_SERVICE}"
}

configure_sudoers() {
  if [[ "${RUN_USER}" == "root" ]]; then
    return
  fi

  local sudoers_file="/etc/sudoers.d/visionops-${RUN_USER}"
  log_info "配置 ${RUN_USER} 的 VisionOps 免密 sudo..."

  cat > "${sudoers_file}" <<EOF
${RUN_USER} ALL=(ALL) NOPASSWD: /bin/systemctl, /usr/bin/systemctl, /usr/bin/pkill, /bin/pkill, /usr/bin/fuser, /bin/fuser, /usr/bin/install, /bin/mkdir, /bin/cp, /bin/rm, /bin/kill, /usr/bin/kill, /usr/bin/tee, /bin/chmod, /usr/bin/chmod
EOF
  chmod 0440 "${sudoers_file}"
  if command -v visudo >/dev/null 2>&1; then
    visudo -cf "${sudoers_file}" >/dev/null
  fi
  log_ok "已配置 sudoers: ${sudoers_file}"
}

fix_permissions() {
  log_info "修正目录权限..."
  chown -R "${RUN_USER}:${RUN_USER}" "${INSTALL_DIR}" || true
  chmod -R u+rwX "${INSTALL_DIR}" || true
  chmod 755 "${INSTALL_DIR}" "${EDGE_DST_DIR}" || true
  log_ok "权限修正完成"
}

enable_services() {
  log_info "注册 systemd 服务..."
  systemctl daemon-reload

  for svc in visionops-collector.service "${HP60C_SERVICE_NAME}" visionops-inference-cpp.service "${TUBE_SERVICE_NAME}"; do
    if systemctl list-unit-files | grep -q "^${svc}"; then
      systemctl enable "${svc}" || true
    fi
  done

  log_ok "服务 enable 完成"
}

start_services_if_needed() {
  if [[ "${VISIONOPS_START_SERVICES}" != "1" ]]; then
    log_warn "VISIONOPS_START_SERVICES=${VISIONOPS_START_SERVICES}，跳过自动启动"
    return
  fi

  log_info "按顺序启动服务..."

  systemctl restart "${HP60C_SERVICE_NAME}" || log_warn "HP60C SDK bridge 启动失败，请检查日志"
  systemctl restart visionops-collector.service || log_warn "Collector 启动失败，请检查日志"

  local model_path
  model_path="$(grep -E '^VISIONOPS_CPP_MODEL_PATH=' "${CPP_ENV}" | cut -d= -f2- || true)"
  if [[ -n "${model_path}" && -f "${model_path}" ]]; then
    systemctl restart visionops-inference-cpp.service || log_warn "C++ 推理服务启动失败，请检查日志"
    systemctl restart "${TUBE_SERVICE_NAME}" || log_warn "tube_station 启动失败，请检查日志"
  else
    log_warn "未找到真实 RKNN 模型，暂不启动 C++ 推理和 tube_station: ${model_path}"
  fi
}

print_summary() {
  cat <<EOF

============================================================
VisionOps LB3576 + HP60C SDK + Tube Modbus 部署完成
============================================================
安装目录:       ${INSTALL_DIR}
运行用户:       ${RUN_USER}
设备 ID:        ${DEVICE_ID}
服务端地址:     ${SERVER_URL}

核心服务:
  - visionops-collector.service              Web / Collector: http://<板子IP>:${VISIONOPS_COLLECTOR_PORT}
  - visionops-hp60c-sdk-bridge.service       HP60C SDK bridge: http://127.0.0.1:${VISIONOPS_HP60C_PORT}
  - visionops-inference-cpp.service          C++ RKNN inference: http://127.0.0.1:${VISIONOPS_CPP_PORT}
  - visionops-tube-station.service           Modbus TCP: 0.0.0.0:${VISIONOPS_TUBE_MODBUS_PORT}

已禁用旧服务:
  - visionops-inference.service
  - visionops-monitor.service
  - visionops-hp60c-ros1-bridge.service

验证命令:
  curl -s http://127.0.0.1:${VISIONOPS_HP60C_PORT}/health | python3 -m json.tool
  curl -s http://127.0.0.1:${VISIONOPS_COLLECTOR_PORT}/api/cpp/hp60c_sdk/health | python3 -m json.tool
  curl -s http://127.0.0.1:${VISIONOPS_CPP_PORT}/health | python3 -m json.tool
  ss -ltnp | grep -E "${VISIONOPS_COLLECTOR_PORT}|${VISIONOPS_HP60C_PORT}|${VISIONOPS_CPP_PORT}|${VISIONOPS_TUBE_MODBUS_PORT}"

查看日志:
  sudo journalctl -u visionops-hp60c-sdk-bridge.service -f -o cat
  sudo journalctl -u visionops-inference-cpp.service -f -o cat
  sudo journalctl -u visionops-collector.service -f -o cat
  sudo journalctl -u visionops-tube-station.service -f -o cat

如果 C++ 推理服务未启动，多数是模型/类别文件或 RKNN runtime 还没准备好。
配置文件:
  ${CPP_ENV}
  ${COLLECTOR_ENV}
  ${RUNTIME_OVERRIDES}

EOF
}

main() {
  ensure_root

  install_system_deps
  create_dirs
  setup_python_env
  deploy_edge_code
  cleanup_legacy_services
  write_collector_runtime_files
  check_rknn_runtime
  write_cpp_env
  build_cpp_inference
  install_cpp_service
  install_hp60c_sdk_bridge
  install_collector_service
  install_tube_station_service
  configure_sudoers
  fix_permissions
  enable_services
  start_services_if_needed
  print_summary
}

main "$@"
