#!/usr/bin/env bash
set -euo pipefail

# VisionOps C++ service one-key deploy
# Steps: sync C++ code -> compile on RK3588 -> install binary/service -> restart -> health check.
# Optional: sync Collector proxy files for v0.5.0 Web/Collector -> C++ service integration.
# Usage examples:
#   bash edge/deploy/deploy_cpp.sh --host 192.168.1.200
#   EDGE_HOST=192.168.1.200 NUM_CLASSES=80 CAMERA_SOURCE='rtsp://...' bash edge/deploy/deploy_cpp.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EDGE_HOST="${EDGE_HOST:-192.168.1.200}"
EDGE_USER="${EDGE_USER:-ubuntu}"
EDGE_PORT="${EDGE_PORT:-22}"
INSTALL_DIR="${INSTALL_DIR:-/opt/visionops}"
SERVICE_NAME="${SERVICE_NAME:-visionops-inference-cpp}"
CPP_PORT="${CPP_PORT:-18080}"

MODEL_PATH="${MODEL_PATH:-/opt/visionops/models/rk3588-001_CUST-000_det_20260427_105932.rknn}"
CLASS_NAMES_FILE="${CLASS_NAMES_FILE:-/opt/visionops/models/rk3588-001_CUST-000_det_20260427_105932.yaml}"
TASK="${TASK:-detection}"
INPUT_SIZE="${INPUT_SIZE:-640,640}"
NUM_CLASSES="${NUM_CLASSES:-80}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.45}"
NPU_CORE="${NPU_CORE:-auto}"
OUTPUT_MODE="${OUTPUT_MODE:-float}"
PREPROCESS_BACKEND="${PREPROCESS_BACKEND:-auto}"
RGA_MODE="${RGA_MODE:-resize_color}"
MAX_DET="${MAX_DET:-100}"
CAMERA_SOURCE="${CAMERA_SOURCE:-rtsp://admin:Abcd123_@192.168.2.64:554/Streaming/channels/101}"
CAMERA_READ_FPS="${CAMERA_READ_FPS:-10}"
DETECT_FPS="${DETECT_FPS:-10}"
SNAPSHOT_FPS="${SNAPSHOT_FPS:-1}"
ENABLE_SNAPSHOT="${ENABLE_SNAPSHOT:-true}"
ENABLE_ANNOTATED="${ENABLE_ANNOTATED:-true}"
STREAM_AUTO_START="${STREAM_AUTO_START:-false}"
STREAM_BACKEND="${STREAM_BACKEND:-opencv}"
STREAM_CODEC="${STREAM_CODEC:-h264}"
RTSP_TRANSPORT="${RTSP_TRANSPORT:-tcp}"
RTSP_TIMEOUT_MS="${RTSP_TIMEOUT_MS:-5000}"
GST_LATENCY_MS="${GST_LATENCY_MS:-100}"
QUIET_FFMPEG_LOG="${QUIET_FFMPEG_LOG:-true}"
INSTALL_GST="${INSTALL_GST:-0}"

# v0.5.0 optional Collector proxy deployment.
# Default keeps this script focused on C++ service only.
SYNC_COLLECTOR="${SYNC_COLLECTOR:-0}"
RESTART_COLLECTOR="${RESTART_COLLECTOR:-1}"
APPLY_CPP_PROXY="${APPLY_CPP_PROXY:-1}"
COLLECTOR_PORT="${COLLECTOR_PORT:-8090}"
COLLECTOR_SERVICE="${COLLECTOR_SERVICE:-visionops-collector}"
CPP_SERVICE_URL="${CPP_SERVICE_URL:-http://127.0.0.1:${CPP_PORT}}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "${EDGE_PORT}")
SCP_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -P "${EDGE_PORT}")
RSYNC_SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${EDGE_PORT}"

log_info() { echo "[INFO] $*"; }
log_ok() { echo "[OK] $*"; }
log_warn() { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

usage() {
  cat <<USAGE
Usage: bash edge/deploy/deploy_cpp.sh [options]

Options:
  --host HOST                 RK3588 host, default: ${EDGE_HOST}
  --user USER                 SSH user, default: ${EDGE_USER}
  --port PORT                 SSH port, default: ${EDGE_PORT}
  --install-dir DIR           Remote install dir, default: ${INSTALL_DIR}
  --cpp-port PORT             C++ service HTTP port, default: ${CPP_PORT}
  --model PATH                Remote RKNN path, default: ${MODEL_PATH}
  --class-names-file PATH     Remote class_names yaml, default: ${CLASS_NAMES_FILE}
  --task TASK                 detection/classification/etc., default: ${TASK}
  --input-size H,W            default: ${INPUT_SIZE}
  --num-classes N             default: 80 for current RKNN test model
  --preprocess-backend MODE    cpu|rga|auto, default: ${PREPROCESS_BACKEND}
  --rga-mode MODE              off|resize_color|resize_only, default: ${RGA_MODE}
  --camera-source URL_OR_IDX  Optional RTSP URL or USB camera index, e.g. 0
  --stream-auto-start true|false, default: ${STREAM_AUTO_START}
  --enable-snapshot true|false, default: ${ENABLE_SNAPSHOT}
  --enable-annotated true|false, default: ${ENABLE_ANNOTATED}
  --stream-backend opencv|gst-mpp, default: ${STREAM_BACKEND}
  --install-gst 0|1           Install common GStreamer packages, default: ${INSTALL_GST}

  # v0.5.0 optional Collector proxy deployment:
  --sync-collector true|false    Sync edge/collector and tools, default: ${SYNC_COLLECTOR}
  --restart-collector true|false Restart Collector service after sync, default: ${RESTART_COLLECTOR}
  --collector-service NAME       Collector systemd service name, default: ${COLLECTOR_SERVICE}
  --collector-port PORT          Collector HTTP port for proxy check, default: ${COLLECTOR_PORT}
  --cpp-service-url URL          C++ service URL for Collector proxy, default: ${CPP_SERVICE_URL}
  -h, --help

Environment variables with the same names are also supported, e.g. EDGE_HOST, NUM_CLASSES, CAMERA_SOURCE.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) EDGE_HOST="$2"; shift 2 ;;
    --user) EDGE_USER="$2"; shift 2 ;;
    --port) EDGE_PORT="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --cpp-port) CPP_PORT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --class-names-file) CLASS_NAMES_FILE="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --input-size) INPUT_SIZE="$2"; shift 2 ;;
    --num-classes) NUM_CLASSES="$2"; shift 2 ;;
    --preprocess-backend) PREPROCESS_BACKEND="$2"; shift 2 ;;
    --rga-mode) RGA_MODE="$2"; shift 2 ;;
    --camera-source) CAMERA_SOURCE="$2"; shift 2 ;;
    --stream-auto-start) STREAM_AUTO_START="$2"; shift 2 ;;
    --enable-snapshot) ENABLE_SNAPSHOT="$2"; shift 2 ;;
    --enable-annotated) ENABLE_ANNOTATED="$2"; shift 2 ;;
    --stream-backend) STREAM_BACKEND="$2"; shift 2 ;;
    --stream-codec) STREAM_CODEC="$2"; shift 2 ;;
    --install-gst) INSTALL_GST="$2"; shift 2 ;;
    --sync-collector) SYNC_COLLECTOR="$2"; shift 2 ;;
    --restart-collector) RESTART_COLLECTOR="$2"; shift 2 ;;
    --collector-service) COLLECTOR_SERVICE="$2"; shift 2 ;;
    --collector-port) COLLECTOR_PORT="$2"; shift 2 ;;
    --cpp-service-url) CPP_SERVICE_URL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) log_error "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

# Rebuild SSH opts if CLI changed port.
SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "${EDGE_PORT}")
SCP_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -P "${EDGE_PORT}")
RSYNC_SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${EDGE_PORT}"

remote() {
  ssh "${SSH_OPTS[@]}" "${EDGE_USER}@${EDGE_HOST}" "$@"
}

remote_sudo() {
  ssh "${SSH_OPTS[@]}" "${EDGE_USER}@${EDGE_HOST}" sudo -n "$@"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log_error "Missing local command: $1"
    exit 1
  fi
}

infer_num_classes() {
  # v0.4.3 当前固定使用指定 RKNN/YAML；不要再从 本地旧 class_names.yaml 推断。
  # 如需改类别数，用 --num-classes 或 NUM_CLASSES=... 显式覆盖。
  echo "${NUM_CLASSES:-80}"
}

write_remote_cpp_env() {
  local remote_tmp="/tmp/visionops_cpp.env.$$"
  local local_tmp
  local_tmp="$(mktemp /tmp/visionops_cpp.env.local.XXXXXX)"
  local num_classes_final="$1"
  {
    echo "# Auto-generated by edge/deploy/deploy_cpp.sh"
    printf 'VISIONOPS_CPP_BIN=%q\n' "${INSTALL_DIR}/bin/visionops_inference_cpp"
    printf 'VISIONOPS_CPP_MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'VISIONOPS_CPP_CLASS_NAMES_FILE=%q\n' "${CLASS_NAMES_FILE}"
    printf 'VISIONOPS_CPP_TASK=%q\n' "${TASK}"
    printf 'VISIONOPS_CPP_PORT=%q\n' "${CPP_PORT}"
    printf 'VISIONOPS_CPP_NPU_CORE=%q\n' "${NPU_CORE}"
    printf 'VISIONOPS_CPP_NUM_CLASSES=%q\n' "${num_classes_final}"
    printf 'VISIONOPS_CPP_INPUT_SIZE=%q\n' "${INPUT_SIZE}"
    printf 'VISIONOPS_CPP_CONF_THRESHOLD=%q\n' "${CONF_THRESHOLD}"
    printf 'VISIONOPS_CPP_NMS_THRESHOLD=%q\n' "${NMS_THRESHOLD}"
    printf 'VISIONOPS_CPP_MAX_DET=%q\n' "${MAX_DET}"
    printf 'VISIONOPS_CPP_OUTPUT_MODE=%q\n' "${OUTPUT_MODE}"
    printf 'VISIONOPS_CPP_PREPROCESS_BACKEND=%q\n' "${PREPROCESS_BACKEND}"
    printf 'VISIONOPS_CPP_RGA_MODE=%q\n' "${RGA_MODE}"
    printf 'VISIONOPS_CPP_CAMERA_READ_FPS=%q\n' "${CAMERA_READ_FPS}"
    printf 'VISIONOPS_CPP_DETECT_FPS=%q\n' "${DETECT_FPS}"
    printf 'VISIONOPS_CPP_SNAPSHOT_FPS=%q\n' "${SNAPSHOT_FPS}"
    printf 'VISIONOPS_CPP_ENABLE_SNAPSHOT=%q\n' "${ENABLE_SNAPSHOT}"
    printf 'VISIONOPS_CPP_ENABLE_ANNOTATED=%q\n' "${ENABLE_ANNOTATED}"
    printf 'VISIONOPS_CPP_STREAM_AUTO_START=%q\n' "${STREAM_AUTO_START}"
    printf 'VISIONOPS_CPP_STREAM_BACKEND=%q\n' "${STREAM_BACKEND}"
    printf 'VISIONOPS_CPP_STREAM_CODEC=%q\n' "${STREAM_CODEC}"
    printf 'VISIONOPS_CPP_GST_LATENCY_MS=%q\n' "${GST_LATENCY_MS}"
    printf 'VISIONOPS_CPP_RTSP_TRANSPORT=%q\n' "${RTSP_TRANSPORT}"
    printf 'VISIONOPS_CPP_RTSP_TIMEOUT_MS=%q\n' "${RTSP_TIMEOUT_MS}"
    printf 'VISIONOPS_CPP_QUIET_FFMPEG_LOG=%q\n' "${QUIET_FFMPEG_LOG}"
    printf 'VISIONOPS_CAMERA_SOURCE=%q\n' "${CAMERA_SOURCE}"
  } > "${local_tmp}"
  scp "${SCP_OPTS[@]}" "${local_tmp}" "${EDGE_USER}@${EDGE_HOST}:${remote_tmp}" >/dev/null
  remote_sudo mkdir -p "${INSTALL_DIR}/edge/runtime"
  remote_sudo mv "${remote_tmp}" "${INSTALL_DIR}/edge/runtime/cpp.env"
  remote_sudo chmod 644 "${INSTALL_DIR}/edge/runtime/cpp.env"
  rm -f "${local_tmp}"
}

normalize_bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) echo "1" ;;
    *) echo "0" ;;
  esac
}

remote_service_exists() {
  local svc="$1"
  remote "systemctl list-unit-files --type=service 2>/dev/null | awk '{print \$1}' | grep -qx '${svc}.service' || systemctl list-units --type=service --all 2>/dev/null | awk '{print \$1}' | grep -qx '${svc}.service'"
}

restart_collector_service() {
  if [[ "$(normalize_bool "${RESTART_COLLECTOR}")" != "1" ]]; then
    log_warn "Collector restart skipped because RESTART_COLLECTOR=${RESTART_COLLECTOR}"
    return 0
  fi

  local candidates=()
  candidates+=("${COLLECTOR_SERVICE}")
  candidates+=("visionops-collector")
  candidates+=("visionops-edge-collector")

  local svc=""
  for c in "${candidates[@]}"; do
    if [[ -n "${c}" ]] && remote "systemctl list-unit-files --type=service 2>/dev/null | awk '{print \$1}' | grep -qx '${c}.service' || systemctl list-units --type=service --all 2>/dev/null | awk '{print \$1}' | grep -qx '${c}.service'"; then
      svc="${c}"
      break
    fi
  done

  if [[ -z "${svc}" ]]; then
    log_warn "Collector systemd service not found. Tried: ${candidates[*]}"
    log_warn "Please restart the Collector manually, otherwise /api/cpp routes will still be Not Found."
    return 0
  fi

  log_info "Restart Collector service: ${svc}.service"
  remote_sudo systemctl restart "${svc}.service"
}

sync_collector_proxy() {
  if [[ "$(normalize_bool "${SYNC_COLLECTOR}")" != "1" ]]; then
    log_info "Skip Collector sync. Use --sync-collector true to deploy v0.5.0 Web/Collector proxy files."
    return 0
  fi

  if [[ ! -d "${REPO_ROOT}/edge/collector" ]]; then
    log_error "Missing ${REPO_ROOT}/edge/collector. Cannot sync Collector proxy."
    exit 1
  fi

  log_info "Sync edge/collector for v0.5.0 C++ proxy"
  remote_sudo mkdir -p "${INSTALL_DIR}/edge/collector"
  remote_sudo chown -R "${EDGE_USER}:${EDGE_USER}" "${INSTALL_DIR}/edge/collector" || true
  rsync -az -e "${RSYNC_SSH}" \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "${REPO_ROOT}/edge/collector/" \
    "${EDGE_USER}@${EDGE_HOST}:${INSTALL_DIR}/edge/collector/"

  if [[ -d "${REPO_ROOT}/tools" ]]; then
    log_info "Sync tools directory"
    remote "mkdir -p '${INSTALL_DIR}/tools'"
    rsync -az -e "${RSYNC_SSH}" \
      --exclude '__pycache__/' \
      --exclude '*.pyc' \
      "${REPO_ROOT}/tools/" \
      "${EDGE_USER}@${EDGE_HOST}:${INSTALL_DIR}/tools/"
  fi

  if [[ "$(normalize_bool "${APPLY_CPP_PROXY}")" == "1" ]]; then
    if remote "test -f '${INSTALL_DIR}/tools/apply_v0_5_0_cpp_proxy.py'"; then
      log_info "Apply v0.5.0 Collector proxy registration"
      remote "cd '${INSTALL_DIR}' && python3 tools/apply_v0_5_0_cpp_proxy.py"
    else
      log_warn "tools/apply_v0_5_0_cpp_proxy.py not found on remote. Ensure main.py registered cpp_inference_router manually."
    fi
  fi

  log_info "Check Collector proxy Python files"
  remote "cd '${INSTALL_DIR}' && python3 -m py_compile \
    edge/collector/backend/services/cpp_inference_client.py \
    edge/collector/backend/routers/cpp_inference.py \
    edge/collector/backend/main.py"

  # If Collector uses an env file/service, this is a helpful default. It is harmless if config.py already defaults to 127.0.0.1:18080.
  log_info "Write optional C++ proxy env hints"
  remote "grep -q '^VISIONOPS_CPP_SERVICE_URL=' '${INSTALL_DIR}/edge/runtime/cpp.env' 2>/dev/null || true"
  remote_sudo mkdir -p "${INSTALL_DIR}/edge/runtime"
  remote "cat > /tmp/visionops_cpp_proxy.env.$$ <<EOF
# Optional hints for Collector C++ proxy
VISIONOPS_CPP_SERVICE_URL=${CPP_SERVICE_URL}
CPP_INFERENCE_URL=${CPP_SERVICE_URL}
CPP_INFERENCE_ENABLED=1
CPP_INFERENCE_TIMEOUT_SEC=3
CPP_INFERENCE_IMAGE_TIMEOUT_SEC=10
EOF"
  remote_sudo mv "/tmp/visionops_cpp_proxy.env.$$" "${INSTALL_DIR}/edge/runtime/cpp_proxy.env"
  remote_sudo chmod 644 "${INSTALL_DIR}/edge/runtime/cpp_proxy.env"

  restart_collector_service

  log_info "Check Collector proxy route"
  if remote "curl -sf 'http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/proxy_info' >/dev/null"; then
    log_ok "Collector C++ proxy route is available: http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/proxy_info"
  else
    log_warn "Collector proxy route check failed."
    log_warn "Run on RK3588: curl -s http://127.0.0.1:${COLLECTOR_PORT}/openapi.json | grep -o '/api/cpp[^\" ]*' | sort | uniq"
    log_warn "If routes are missing, confirm main.py registration and restart the actual Collector process."
  fi
}


main() {
  require_cmd ssh
  require_cmd scp
  require_cmd rsync
  require_cmd python3

  if [[ ! -d "${REPO_ROOT}/edge/inference_cpp" ]]; then
    log_error "Missing ${REPO_ROOT}/edge/inference_cpp. Run this script from the repository after applying v0.4.3 patch."
    exit 1
  fi

  local num_classes_final
  num_classes_final="$(infer_num_classes | tail -n 1)"
  if [[ -z "${num_classes_final}" ]]; then
    num_classes_final="80"
  fi

  log_info "Target: ${EDGE_USER}@${EDGE_HOST}:${EDGE_PORT}"
  log_info "Install dir: ${INSTALL_DIR}"
  log_info "C++ port: ${CPP_PORT}, task=${TASK}, num_classes=${num_classes_final}, input_size=${INPUT_SIZE}, preprocess_backend=${PREPROCESS_BACKEND}, rga_mode=${RGA_MODE}, snapshot=${ENABLE_SNAPSHOT}, annotated=${ENABLE_ANNOTATED}"

  remote "echo connected >/dev/null"
  if ! remote "sudo -n true"; then
    log_error "Remote sudo requires password. Please configure NOPASSWD sudo for ${EDGE_USER}, otherwise non-interactive deploy cannot restart service."
    exit 1
  fi
  log_ok "SSH and sudo check passed"

  log_info "Create remote directories"
  remote_sudo mkdir -p "${INSTALL_DIR}/edge" "${INSTALL_DIR}/bin" "${INSTALL_DIR}/logs" "${INSTALL_DIR}/models"
  if id -u >/dev/null 2>&1; then :; fi
  remote_sudo chown -R "${EDGE_USER}:${EDGE_USER}" "${INSTALL_DIR}/edge" "${INSTALL_DIR}/logs" || true

  log_info "Sync edge/inference_cpp"
  rsync -az --delete -e "${RSYNC_SSH}" \
    "${REPO_ROOT}/edge/inference_cpp/" \
    "${EDGE_USER}@${EDGE_HOST}:${INSTALL_DIR}/edge/inference_cpp/"

  if [[ -d "${REPO_ROOT}/edge/runtime" ]]; then
    log_info "Sync edge/runtime without deleting remote generated files"
    rsync -az -e "${RSYNC_SSH}" \
      "${REPO_ROOT}/edge/runtime/" \
      "${EDGE_USER}@${EDGE_HOST}:${INSTALL_DIR}/edge/runtime/" || true
  fi

  log_info "Write cpp.env"
  write_remote_cpp_env "${num_classes_final}"

  log_info "Install build dependencies on RK3588"
  remote_sudo apt-get update
  remote_sudo apt-get install -y build-essential cmake pkg-config curl libopencv-dev libgomp1
  if [[ "${INSTALL_GST}" == "1" ]]; then
    remote_sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-libav || true
  fi

  log_info "Check RKNN runtime header/library"
  if ! remote "test -f /usr/include/rknn_api.h -o -f /usr/local/include/rknn_api.h -o -f /usr/include/rknn/rknn_api.h -o -f /usr/local/include/rknn/rknn_api.h"; then
    log_warn "rknn_api.h was not found in common include paths. CMake will fail unless CMAKE_INCLUDE_PATH is configured."
  fi
  if ! remote "ldconfig -p 2>/dev/null | grep -q librknnrt || test -f /usr/lib/librknnrt.so -o -f /usr/lib/aarch64-linux-gnu/librknnrt.so -o -f /usr/local/lib/librknnrt.so"; then
    log_warn "librknnrt.so was not found in common library paths. Put it in /usr/lib and run sudo ldconfig if build/link fails."
  fi

  log_info "Check optional RGA header/library"
  if remote "test -f /usr/include/rga/im2d.hpp -o -f /usr/local/include/rga/im2d.hpp -o -f /usr/include/im2d.hpp -o -f /usr/local/include/im2d.hpp -o -f /usr/include/rga/RgaUtils.h -o -f /usr/local/include/rga/RgaUtils.h"; then
    log_ok "RGA header found"
  else
    log_warn "RGA header was not found in common include paths. v0.4.3 will still build with CPU preprocessing fallback."
  fi
  if remote "ldconfig -p 2>/dev/null | grep -q librga || test -f /usr/lib/librga.so -o -f /usr/lib/aarch64-linux-gnu/librga.so -o -f /usr/local/lib/librga.so"; then
    log_ok "RGA library found"
  else
    log_warn "librga.so was not found in common library paths. v0.4.3 will still build with CPU preprocessing fallback."
  fi

  log_info "Compile C++ service on RK3588"
  remote "cd '${INSTALL_DIR}/edge/inference_cpp' && rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"

  log_info "Install binary to ${INSTALL_DIR}/bin"
  remote_sudo cmake --install "${INSTALL_DIR}/edge/inference_cpp/build" --prefix "${INSTALL_DIR}"
  remote_sudo chmod 755 "${INSTALL_DIR}/bin/visionops_inference_cpp"
  remote_sudo chmod 755 "${INSTALL_DIR}/edge/inference_cpp/scripts/start_visionops_inference_cpp.sh"

  log_info "Install systemd service"
  scp "${SCP_OPTS[@]}" "${REPO_ROOT}/edge/deploy/visionops-inference-cpp.service" "${EDGE_USER}@${EDGE_HOST}:/tmp/visionops-inference-cpp.service" >/dev/null
  remote_sudo mv /tmp/visionops-inference-cpp.service /etc/systemd/system/visionops-inference-cpp.service
  remote_sudo chmod 644 /etc/systemd/system/visionops-inference-cpp.service
  remote_sudo systemctl daemon-reload
  remote_sudo systemctl enable "${SERVICE_NAME}.service"

  log_info "Restart ${SERVICE_NAME}"
  remote_sudo systemctl restart "${SERVICE_NAME}.service"

  log_info "Health check"
  local ok=0
  for i in {1..15}; do
    if remote "curl -sf 'http://127.0.0.1:${CPP_PORT}/health'"; then
      ok=1
      break
    fi
    sleep 1
  done

  if [[ "${ok}" != "1" ]]; then
    log_error "Health check failed. Recent logs:"
    remote_sudo journalctl -u "${SERVICE_NAME}.service" -n 80 --no-pager || true
    exit 1
  fi

  log_ok "C++ service deployed successfully"
  log_ok "Local check from RK3588: curl http://127.0.0.1:${CPP_PORT}/health"

  sync_collector_proxy

  log_ok "Remote C++ logs: ssh -p ${EDGE_PORT} ${EDGE_USER}@${EDGE_HOST} 'sudo journalctl -u ${SERVICE_NAME} -f'"
  if [[ "$(normalize_bool "${SYNC_COLLECTOR}")" == "1" ]]; then
    log_ok "Collector proxy check: curl http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/health"
  fi
}

main "$@"