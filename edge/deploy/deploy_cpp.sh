#!/usr/bin/env bash
set -euo pipefail

# VisionOps C++ 服务部署脚本
# 功能：
#   1. all：同步 C++ 代码到 RK3588，编译安装服务，写入 cpp.env，重启并健康检查。
#   2. code-only：只同步 edge/ 代码到 RK3588，不编译、不重启。
#   3. model-only：只把本地 RKNN 模型和同名 YAML meta 写入 models/share_models/<device_id>，
#      由 Syncthing 同步到边缘端；不再通过 SSH/SCP 部署。
#   4. model-only + --roi-classification：生成 ROI 分类双模型 bundle。
# 示例：
#   bash edge/deploy/deploy_cpp.sh --host 192.168.1.200
#   bash edge/deploy/deploy_cpp.sh --model-only --device-id rk3588-001
#   bash edge/deploy/deploy_cpp.sh --model-only --roi-classification --device-id rk3588-001

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EDGE_HOST="${EDGE_HOST:-192.168.1.200}"
EDGE_USER="${EDGE_USER:-ubuntu}"
EDGE_PORT="${EDGE_PORT:-22}"
INSTALL_DIR="${INSTALL_DIR:-/opt/visionops}"
SERVICE_NAME="${SERVICE_NAME:-visionops-inference-cpp}"
CPP_PORT="${CPP_PORT:-18080}"

MODEL_PATH="${MODEL_PATH:-/opt/visionops/models/rk3588-001_COCO128_det_20260511_155057.rknn}"
CLASS_NAMES_FILE="${CLASS_NAMES_FILE:-/opt/visionops/models/rk3588-001_COCO128_det_20260511_155057.yaml}"
TASK="${TASK:-detection}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-}"
INPUT_SIZE="${INPUT_SIZE:-640,640}"
NUM_CLASSES="${NUM_CLASSES:-80}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.45}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
TOPK="${TOPK:-5}"
NPU_CORE="${NPU_CORE:-auto}"
OUTPUT_MODE="${OUTPUT_MODE:-float}"
PREPROCESS_BACKEND="${PREPROCESS_BACKEND:-auto}"
RGA_MODE="${RGA_MODE:-resize_color}"
MAX_DET="${MAX_DET:-100}"
CAMERA_SOURCE="${CAMERA_SOURCE:-rtsp://admin:Abcd123_@192.168.2.64:554/Streaming/channels/101}"
# v0.7.1 C++ USB/OpenCV camera options. For Orbbec UVC RGB, use:
#   CAMERA_TYPE=usb CAMERA_SOURCE=/dev/video7 CAMERA_WIDTH=1280 CAMERA_HEIGHT=800 CAMERA_FPS=10 CAMERA_FOURCC=YUYV
CAMERA_TYPE="${CAMERA_TYPE:-usb}"
CAMERA_WIDTH="${CAMERA_WIDTH:-0}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-0}"
CAMERA_FPS="${CAMERA_FPS:-10}"
CAMERA_BUFFER_SIZE="${CAMERA_BUFFER_SIZE:-1}"
CAMERA_FOURCC="${CAMERA_FOURCC:-YUYV}"
CAMERA_READ_FPS="${CAMERA_READ_FPS:-10}"
DETECT_FPS="${DETECT_FPS:-1}"
SNAPSHOT_FPS="${SNAPSHOT_FPS:-10}"
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

# 部署模式：
#   all        : 同步代码、写 cpp.env、在 RK3588 编译/安装 C++ 服务、重启并健康检查（默认）
#   code-only  : 只同步本地 edge/ 目录到板端；不编译、不改环境、不重启
#   model-only : 只把本地 .rknn + .yaml 写入 models/share_models/<device_id>；不使用 SSH
DEPLOY_MODE="${DEPLOY_MODE:-all}"
SHARE_MODEL_DIR="${SHARE_MODEL_DIR:-models/share_models}"
MODEL_VERSION="${MODEL_VERSION:-}"
DEVICE_ID="${DEVICE_ID:-}"
CUSTOMER_ID="${CUSTOMER_ID:-}"
ROI_CLASSIFICATION="${ROI_CLASSIFICATION:-0}"

# v0.5.0 optional Collector proxy deployment.
# v0.7.2 default syncs Collector so the C++ preview/detect buttons are deployed together.
# Set SYNC_COLLECTOR=0 if you only want to deploy the C++ service.
SYNC_COLLECTOR="${SYNC_COLLECTOR:-1}"
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
  --task TASK                 detection/classification/obb_detection/segmentation/roi_classification, default: ${TASK}
  --pipeline-config PATH       ROI classification pipeline.yaml path, default: ${PIPELINE_CONFIG}
  --input-size H,W            default: ${INPUT_SIZE}
  --num-classes N             default: 80 for current RKNN test model
  --mask-threshold X           Segmentation mask threshold, default: ${MASK_THRESHOLD}
  --topk N                    Classification top-k, default: ${TOPK}
  --preprocess-backend MODE    cpu|rga|auto, default: ${PREPROCESS_BACKEND}
  --rga-mode MODE              off|resize_color|resize_only, default: ${RGA_MODE}
  --camera-source URL_OR_IDX  Optional RTSP URL, USB camera index, or /dev/videoX
  --camera-type auto|rtsp|usb  Camera type hint, default: ${CAMERA_TYPE}
  --camera-width W            USB/OpenCV requested width, default: ${CAMERA_WIDTH}
  --camera-height H           USB/OpenCV requested height, default: ${CAMERA_HEIGHT}
  --camera-fps FPS            USB/OpenCV requested capture FPS, default: ${CAMERA_FPS}
  --camera-fourcc FOURCC      USB/OpenCV FOURCC, e.g. YUYV or MJPG, default: ${CAMERA_FOURCC}
  --camera-buffer-size N      USB/OpenCV buffer size, default: ${CAMERA_BUFFER_SIZE}
  --stream-auto-start true|false, default: ${STREAM_AUTO_START}
  --enable-snapshot true|false, default: ${ENABLE_SNAPSHOT}
  --enable-annotated true|false, default: ${ENABLE_ANNOTATED}
  --stream-backend opencv|gst-mpp, default: ${STREAM_BACKEND}
  --install-gst 0|1           Install common GStreamer packages, default: ${INSTALL_GST}

  # 部署模式：
  --all                       完整部署：同步代码、写 cpp.env、编译安装、重启并健康检查（默认）
  --code-only                 只同步本地 edge/ 到远端 ${INSTALL_DIR}/edge；不编译、不重启
  --model-only                只把本地 .rknn + 同名 YAML 写入 ${SHARE_MODEL_DIR}/<device_id>；不 SSH
  --roi-classification        model-only 模式下生成 ROI 分类双模型 bundle
  --device-id ID              覆盖 manifest.json 中的 device_id
  --customer-id ID            覆盖 manifest.json 中的 customer_id
  --mode MODE                 all|code-only|model-only，默认: ${DEPLOY_MODE}
  --share-model-dir DIR       本地 Syncthing 模型共享目录，默认: ${SHARE_MODEL_DIR}
  --model-version NAME        覆盖自动生成的版本名，不带后缀

  # v0.5.0 可选 Collector 代理部署：
  --sync-collector true|false    Sync edge/collector and tools, default: ${SYNC_COLLECTOR} (v0.7.2 defaults to true)
  --restart-collector true|false Restart Collector service after sync, default: ${RESTART_COLLECTOR}
  --collector-service NAME       Collector systemd service name, default: ${COLLECTOR_SERVICE}
  --collector-port PORT          Collector HTTP port for proxy check, default: ${COLLECTOR_PORT}
  --cpp-service-url URL          C++ service URL for Collector proxy, default: ${CPP_SERVICE_URL}
  -h, --help

同名环境变量也支持，例如 EDGE_HOST、NUM_CLASSES、CAMERA_SOURCE、DEVICE_ID、CUSTOMER_ID。
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) DEPLOY_MODE="all"; shift ;;
    --code-only) DEPLOY_MODE="code-only"; shift ;;
    --model-only) DEPLOY_MODE="model-only"; shift ;;
    --roi-classification) ROI_CLASSIFICATION="1"; shift ;;
    --device-id) DEVICE_ID="$2"; shift 2 ;;
    --customer-id) CUSTOMER_ID="$2"; shift 2 ;;
    --mode) DEPLOY_MODE="$2"; shift 2 ;;
    --share-model-dir) SHARE_MODEL_DIR="$2"; shift 2 ;;
    --model-version) MODEL_VERSION="$2"; shift 2 ;;
    --host) EDGE_HOST="$2"; shift 2 ;;
    --user) EDGE_USER="$2"; shift 2 ;;
    --port) EDGE_PORT="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --cpp-port) CPP_PORT="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --class-names-file) CLASS_NAMES_FILE="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --pipeline-config) PIPELINE_CONFIG="$2"; shift 2 ;;
    --input-size) INPUT_SIZE="$2"; shift 2 ;;
    --num-classes) NUM_CLASSES="$2"; shift 2 ;;
    --mask-threshold) MASK_THRESHOLD="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --preprocess-backend) PREPROCESS_BACKEND="$2"; shift 2 ;;
    --rga-mode) RGA_MODE="$2"; shift 2 ;;
    --camera-source) CAMERA_SOURCE="$2"; shift 2 ;;
    --camera-type) CAMERA_TYPE="$2"; shift 2 ;;
    --camera-width) CAMERA_WIDTH="$2"; shift 2 ;;
    --camera-height) CAMERA_HEIGHT="$2"; shift 2 ;;
    --camera-fps) CAMERA_FPS="$2"; shift 2 ;;
    --camera-fourcc) CAMERA_FOURCC="$2"; shift 2 ;;
    --camera-buffer-size) CAMERA_BUFFER_SIZE="$2"; shift 2 ;;
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
    printf 'VISIONOPS_CPP_PIPELINE_CONFIG=%q\n' "${PIPELINE_CONFIG}"
    printf 'VISIONOPS_CPP_PORT=%q\n' "${CPP_PORT}"
    printf 'VISIONOPS_CPP_NPU_CORE=%q\n' "${NPU_CORE}"
    printf 'VISIONOPS_CPP_NUM_CLASSES=%q\n' "${num_classes_final}"
    printf 'VISIONOPS_CPP_INPUT_SIZE=%q\n' "${INPUT_SIZE}"
    printf 'VISIONOPS_CPP_CONF_THRESHOLD=%q\n' "${CONF_THRESHOLD}"
    printf 'VISIONOPS_CPP_NMS_THRESHOLD=%q\n' "${NMS_THRESHOLD}"
    printf 'VISIONOPS_CPP_MASK_THRESHOLD=%q\n' "${MASK_THRESHOLD}"
    printf 'VISIONOPS_CPP_TOPK=%q\n' "${TOPK}"
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
    printf 'VISIONOPS_CPP_CAMERA_TYPE=%q\n' "${CAMERA_TYPE}"
    printf 'VISIONOPS_CPP_CAMERA_WIDTH=%q\n' "${CAMERA_WIDTH}"
    printf 'VISIONOPS_CPP_CAMERA_HEIGHT=%q\n' "${CAMERA_HEIGHT}"
    printf 'VISIONOPS_CPP_CAMERA_FPS=%q\n' "${CAMERA_FPS}"
    printf 'VISIONOPS_CPP_CAMERA_BUFFER_SIZE=%q\n' "${CAMERA_BUFFER_SIZE}"
    printf 'VISIONOPS_CPP_CAMERA_FOURCC=%q\n' "${CAMERA_FOURCC}"
    printf 'VISIONOPS_CPP_CAMERA_SOURCE=%q\n' "${CAMERA_SOURCE}"
    printf 'VISIONOPS_CAMERA_SOURCE=%q\n' "${CAMERA_SOURCE}"
  } > "${local_tmp}"
  scp "${SCP_OPTS[@]}" "${local_tmp}" "${EDGE_USER}@${EDGE_HOST}:${remote_tmp}" >/dev/null
  remote_sudo mkdir -p "${INSTALL_DIR}/edge/runtime"
  remote_sudo mv "${remote_tmp}" "${INSTALL_DIR}/edge/runtime/cpp.env"
  remote_sudo chmod 664 "${INSTALL_DIR}/edge/runtime/cpp.env"
  remote_sudo chown "${EDGE_USER}:${EDGE_USER}" "${INSTALL_DIR}/edge/runtime/cpp.env" || true
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
    edge/collector/backend/services/cpp_runtime_settings.py \
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




normalize_task_key() {
  case "${1:-}" in
    cls|classify|classification|image_classification) echo "classification" ;;
    obb|obb_detection|oriented_detection|rotated_detection) echo "obb" ;;
    seg|segment|segmentation|instance_segmentation|yolo_seg|yolov8_seg) echo "segmentation" ;;
    roi|roi_classification) echo "roi_classification" ;;
    ""|detect|detection|yolo_detection) echo "detection" ;;
    *) echo "$1" ;;
  esac
}

meta_task_name() {
  case "$(normalize_task_key "$1")" in
    classification) echo "classification" ;;
    obb) echo "obb_detection" ;;
    segmentation) echo "segmentation" ;;
    roi_classification) echo "roi_classification" ;;
    *) echo "detection" ;;
  esac
}

task_short_name() {
  case "$(normalize_task_key "$1")" in
    classification) echo "cls" ;;
    obb) echo "obb" ;;
    segmentation) echo "seg" ;;
    roi_classification) echo "roi_cls" ;;
    *) echo "det" ;;
  esac
}

normalize_name_part() {
  local value="$1"
  value="${value##*/}"
  value="$(echo "${value}" | sed -E 's/[^A-Za-z0-9_-]+/-/g; s/^-+//; s/-+$//')"
  echo "${value}"
}

get_file_md5() {
  md5sum "$1" | awk '{print $1}'
}

get_file_size_bytes() {
  stat -c%s "$1"
}

manifest_path() {
  echo "${REPO_ROOT}/data/model_context/manifest.json"
}

read_manifest_field() {
  local key="$1"
  export VISIONOPS_REPO_ROOT_FOR_DEPLOY="${REPO_ROOT}"
  export VISIONOPS_MANIFEST_KEY="${key}"
  python3 - <<'PY'
from pathlib import Path
import json
import os

root = Path(os.environ.get("VISIONOPS_REPO_ROOT_FOR_DEPLOY", ".")).resolve()
key = os.environ.get("VISIONOPS_MANIFEST_KEY", "")
p = root / "data" / "model_context" / "manifest.json"
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    data = {}
print(str(data.get(key) or "").strip())
PY
}

resolve_device_and_customer() {
  local did="${DEVICE_ID:-}"
  local cid="${CUSTOMER_ID:-}"

  if [[ -z "${did}" ]]; then
    did="$(read_manifest_field device_id)"
  fi
  if [[ -z "${cid}" ]]; then
    cid="$(read_manifest_field customer_id)"
  fi

  did="$(normalize_name_part "${did}")"
  cid="$(normalize_name_part "${cid:-CUST-000}")"

  if [[ -z "${did}" ]]; then
    log_error "缺少 device_id。请传 --device-id，或确认 data/model_context/manifest.json 中包含 device_id。"
    exit 1
  fi
  if [[ -z "${cid}" ]]; then
    cid="CUST-000"
  fi

  DEVICE_ID="${did}"
  CUSTOMER_ID="${cid}"
}

current_timestamp() {
  date +%Y%m%d_%H%M%S
}

build_version_name() {
  local task_key="$1"
  local override="${MODEL_VERSION:-}"
  if [[ -n "${override}" ]]; then
    normalize_name_part "${override}"
  else
    printf "%s_%s_%s_%s\n" \
      "$(normalize_name_part "${DEVICE_ID}")" \
      "$(normalize_name_part "${CUSTOMER_ID}")" \
      "$(task_short_name "${task_key}")" \
      "$(current_timestamp)"
  fi
}

device_share_dir() {
  local root="${SHARE_MODEL_DIR}"
  if [[ "${root}" != /* ]]; then
    root="${REPO_ROOT}/${root}"
  fi
  echo "${root}/${DEVICE_ID}"
}

infer_task_from_config() {
  export VISIONOPS_REPO_ROOT_FOR_DEPLOY="${REPO_ROOT}"
  python3 - <<'PY'
from pathlib import Path
import os
import re

root = Path(os.environ.get("VISIONOPS_REPO_ROOT_FOR_DEPLOY", ".")).resolve()
paths = [
    root / "pipeline" / "configs" / "generated" / "task.generated.yaml",
    root / "pipeline" / "configs" / "task.yaml",
]

def norm(s):
    s = (s or "detection").strip().lower()
    if s in {"cls", "classify", "classification", "image_classification"}:
        return "classification"
    if s in {"obb", "obb_detection", "oriented_detection", "rotated_detection"}:
        return "obb"
    if s in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg", "yolov8_seg"}:
        return "segmentation"
    if s in {"roi", "roi_classification"}:
        return "roi_classification"
    return "detection"

for p in paths:
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(txt) or {}
        task = cfg.get("task") if isinstance(cfg.get("task"), dict) else {}
        value = task.get("type") or task.get("name") or cfg.get("task_type")
        if value:
            print(norm(str(value)))
            raise SystemExit
    except SystemExit:
        raise
    except Exception:
        pass
    m = re.search(r"(?m)^\s*type:\s*['\"]?([A-Za-z0-9_ -]+)['\"]?\s*$", txt)
    if not m:
        m = re.search(r"(?m)^\s*name:\s*['\"]?([A-Za-z0-9_ -]+)['\"]?\s*$", txt)
    if m:
        print(norm(m.group(1)))
        raise SystemExit
print("detection")
PY
}

model_suffix_for_task() {
  case "$(normalize_task_key "$1")" in
    classification) echo "classification" ;;
    obb) echo "obb" ;;
    segmentation) echo "segmentation" ;;
    *) echo "detection" ;;
  esac
}

metrics_file_for_task() {
  case "$(normalize_task_key "$1")" in
    classification) echo "${REPO_ROOT}/models/metrics_classification/eval_metrics.json" ;;
    obb) echo "${REPO_ROOT}/models/metrics_obb/eval_metrics.json" ;;
    segmentation) echo "${REPO_ROOT}/models/metrics_segmentation/eval_metrics.json" ;;
    *) echo "${REPO_ROOT}/models/metrics_detection/eval_metrics.json" ;;
  esac
}

default_local_model_for_task() {
  local task_key suffix
  task_key="$(normalize_task_key "$1")"
  suffix="$(model_suffix_for_task "${task_key}")"
  local candidates=(
    "${REPO_ROOT}/models/export_${suffix}/model.rknn"
    "${REPO_ROOT}/models/export_${suffix}/best.rknn"
    "${REPO_ROOT}/models/export/model.rknn"
  )
  for p in "${candidates[@]}"; do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  echo "${REPO_ROOT}/models/export_${suffix}/model.rknn"
}

default_local_meta_for_task() {
  local task_key suffix
  task_key="$(normalize_task_key "$1")"
  suffix="$(model_suffix_for_task "${task_key}")"
  local candidates=(
    "${REPO_ROOT}/models/export_${suffix}/model.yaml"
    "${REPO_ROOT}/models/export_${suffix}/class_names.yaml"
    "${REPO_ROOT}/models/export_${suffix}/${suffix}.yaml"
    "${REPO_ROOT}/models/export/model.yaml"
  )
  if [[ "${2:-}" == "allow_runtime_fallback" ]]; then
    candidates+=("${REPO_ROOT}/edge/runtime/class_names.yaml")
  fi
  for p in "${candidates[@]}"; do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  echo ""
}

build_single_model_meta_context() {
  local task_key="$1"
  local src_model="$2"
  local src_meta="$3"
  local version_name="$4"
  local version_file="$5"

  export VISIONOPS_REPO_ROOT_FOR_DEPLOY="${REPO_ROOT}"
  export VISIONOPS_DEPLOY_TASK="${task_key}"
  export VISIONOPS_DEPLOY_MODEL="${src_model}"
  export VISIONOPS_DEPLOY_META="${src_meta}"
  export VISIONOPS_DEPLOY_VERSION="${version_name}"
  export VISIONOPS_DEPLOY_MODEL_FILE="${version_file}"
  export VISIONOPS_DEPLOY_DEVICE_ID="${DEVICE_ID}"
  export VISIONOPS_DEPLOY_CUSTOMER_ID="${CUSTOMER_ID}"
  export VISIONOPS_DEPLOY_MODEL_MD5="$(get_file_md5 "${src_model}")"
  export VISIONOPS_DEPLOY_MODEL_SIZE="$(get_file_size_bytes "${src_model}")"
  export VISIONOPS_DEPLOY_CONF_THRESHOLD="${CONF_THRESHOLD}"
  export VISIONOPS_DEPLOY_NMS_THRESHOLD="${NMS_THRESHOLD}"
  export VISIONOPS_DEPLOY_MASK_THRESHOLD="${MASK_THRESHOLD}"
  export VISIONOPS_DEPLOY_TOPK="${TOPK}"
  export VISIONOPS_DEPLOY_INPUT_SIZE="${INPUT_SIZE}"

  python3 - <<'PY'
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import re
import sys

try:
    import yaml  # type: ignore
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"缺少 PyYAML: {exc}"}, ensure_ascii=False))
    raise SystemExit(0)

root = Path(os.environ["VISIONOPS_REPO_ROOT_FOR_DEPLOY"]).resolve()
task_key = os.environ.get("VISIONOPS_DEPLOY_TASK", "detection")
model_path = Path(os.environ["VISIONOPS_DEPLOY_MODEL"])
meta_path = Path(os.environ.get("VISIONOPS_DEPLOY_META", ""))
version_name = os.environ["VISIONOPS_DEPLOY_VERSION"]
version_file = os.environ["VISIONOPS_DEPLOY_MODEL_FILE"]
device_id = os.environ["VISIONOPS_DEPLOY_DEVICE_ID"]
customer_id = os.environ.get("VISIONOPS_DEPLOY_CUSTOMER_ID") or "CUST-000"
model_md5 = os.environ["VISIONOPS_DEPLOY_MODEL_MD5"]
model_size = int(os.environ["VISIONOPS_DEPLOY_MODEL_SIZE"])

def fail(msg):
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    raise SystemExit(0)

def norm_task(t):
    s = str(t or "").strip().lower()
    if s in {"classification", "classify", "cls", "image_classification"}:
        return "classification"
    if s in {"obb", "obb_detection", "oriented_detection", "rotated_detection"}:
        return "obb_detection"
    if s in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg", "yolov8_seg"}:
        return "segmentation"
    return "detection"

def suffix_for_task(t):
    return {"classification": "classification", "obb_detection": "obb", "segmentation": "segmentation"}.get(t, "detection")

def load_yaml_optional(p):
    try:
        if p and p.exists() and p.is_file():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {}

def load_json_optional(p):
    try:
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}

def normalize_names(v):
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, dict):
        def key_sort(k):
            try:
                return int(k)
            except Exception:
                return str(k)
        return [str(v[k]) for k in sorted(v.keys(), key=key_sort)]
    return []

def normalize_input(v, default):
    if isinstance(v, str):
        items = v.replace(",", " ").split()
    elif isinstance(v, (list, tuple)):
        items = list(v)
    else:
        items = default
    if len(items) != 2:
        return default
    try:
        h, w = int(items[0]), int(items[1])
        if h <= 0 or w <= 0:
            raise ValueError
        return [h, w]
    except Exception:
        return default

def first_nonempty(*vals):
    for v in vals:
        if v is not None and str(v).strip() != "":
            return v
    return ""

def read_data_yaml_names(task):
    if task == "classification":
        raw = root / "data" / "raw_classification"
        if raw.exists():
            return [p.name for p in sorted(raw.iterdir()) if p.is_dir()]
        return []
    suffix = suffix_for_task(task)
    data_yaml = root / "data" / f"raw_{suffix}" / "data.yaml"
    data = load_yaml_optional(data_yaml)
    return normalize_names(data.get("names"))

def metrics_path_for_task(task):
    suffix = suffix_for_task(task)
    return root / "models" / f"metrics_{suffix}" / "eval_metrics.json"

task = norm_task(task_key)
cfg = load_yaml_optional(meta_path)
metrics = load_json_optional(metrics_path_for_task(task))
manifest_path = root / "data" / "model_context" / "manifest.json"
manifest = load_json_optional(manifest_path)
if not isinstance(manifest, dict):
    manifest = {}

class_names = (
    normalize_names(cfg.get("class_names"))
    or normalize_names(cfg.get("names"))
    or normalize_names(metrics.get("class_names"))
    or normalize_names(metrics.get("names"))
    or read_data_yaml_names(task)
)
if not class_names:
    class_names = [f"class_{i}" for i in range(int(os.environ.get("NUM_CLASSES", "80") or 80))]

num_classes = first_nonempty(cfg.get("num_classes"), metrics.get("num_classes"), len(class_names))
try:
    num_classes = int(num_classes)
except Exception:
    num_classes = len(class_names)
if num_classes != len(class_names):
    num_classes = len(class_names)

default_input = [224, 224] if task == "classification" else [640, 640]
input_size = normalize_input(os.environ.get("VISIONOPS_DEPLOY_INPUT_SIZE"), normalize_input(cfg.get("input_size"), default_input))

topk_default = min(5, num_classes) if task == "classification" else 5
try:
    topk = int(first_nonempty(cfg.get("topk"), os.environ.get("VISIONOPS_DEPLOY_TOPK"), topk_default))
except Exception:
    topk = topk_default
topk = max(1, min(topk, max(1, num_classes)))

def as_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)

conf_threshold = as_float(first_nonempty(cfg.get("conf_threshold"), os.environ.get("VISIONOPS_DEPLOY_CONF_THRESHOLD"), 0.25), 0.25)
nms_threshold = as_float(first_nonempty(cfg.get("nms_threshold"), os.environ.get("VISIONOPS_DEPLOY_NMS_THRESHOLD"), 0.45), 0.45)
mask_threshold = as_float(first_nonempty(cfg.get("mask_threshold"), os.environ.get("VISIONOPS_DEPLOY_MASK_THRESHOLD"), 0.5), 0.5)

source_device_id = first_nonempty(manifest.get("device_id"), manifest.get("equipment_id"), manifest.get("edge_device_id"))
dataset_customer = first_nonempty(manifest.get("customer_id"), manifest.get("customer"), manifest.get("cust_id"), customer_id, "CUST-000")
counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}

meta = {
    "schema_version": 1,
    "task": task,
    "input_size": input_size,
    "num_classes": num_classes,
    "class_names": class_names,
    "topk": topk,
    "conf_threshold": conf_threshold,
    "nms_threshold": nms_threshold,
    "model": {
        "name": version_name,
        "file": version_file,
        "display_name": version_name,
        "md5": model_md5,
        "size_bytes": model_size,
        "source_path": str(model_path.relative_to(root) if model_path.is_absolute() and str(model_path).startswith(str(root)) else model_path),
    },
    "dataset": {
        "manifest_path": str(manifest_path.relative_to(root) if manifest_path.exists() else manifest_path),
        "dataset_id": first_nonempty(manifest.get("dataset_id"), manifest.get("package_id"), manifest.get("batch_id")),
        "device_id": str(device_id),
        "source_device_id": str(source_device_id or ""),
        "customer_id": str(dataset_customer),
        "counts": counts,
        "raw_manifest": manifest,
    },
    "deploy": {
        "deployed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deployed_by": "edge/deploy/deploy_cpp.sh --model-only",
        "target_device": str(device_id),
    },
}

if task == "segmentation":
    meta["mask_threshold"] = mask_threshold
if metrics:
    meta["metrics"] = metrics

print(json.dumps({"ok": True, "meta": meta}, ensure_ascii=False))
PY
}

write_yaml_from_context() {
  local context_file="$1"
  local out_file="$2"
  python3 - <<'PY' "$context_file" "$out_file"
from pathlib import Path
import json
import sys
import yaml

ctx = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not ctx.get("ok"):
    raise SystemExit(ctx.get("error", "生成 YAML 失败"))
Path(sys.argv[2]).write_text(
    yaml.safe_dump(ctx["meta"], allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
PY
}

copy_pair_to_share() {
  local src_model="$1"
  local target_dir="$2"
  local version_name="$3"
  local task_key="$4"
  local src_meta="${5:-}"

  [[ -f "${src_model}" ]] || { log_error "本地 RKNN 文件不存在: ${src_model}"; exit 1; }

  mkdir -p "${target_dir}"
  local target_model="${target_dir}/${version_name}.rknn"
  local target_meta="${target_dir}/${version_name}.yaml"
  local tmp_model="${target_dir}/.${version_name}.rknn.$$.tmp"
  local tmp_meta="${target_dir}/.${version_name}.yaml.$$.tmp"
  local context_file
  context_file="$(mktemp /tmp/visionops_cpp_model_meta_XXXXXX.json)"

  build_single_model_meta_context "${task_key}" "${src_model}" "${src_meta}" "${version_name}" "${version_name}.rknn" > "${context_file}"
  local ok err
  ok="$(python3 - <<'PY' "$context_file"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("ok"))
PY
)"
  if [[ "${ok}" != "True" && "${ok}" != "true" ]]; then
    err="$(python3 - <<'PY' "$context_file"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("error", "生成模型 YAML 失败"))
PY
)"
    log_error "${err}"
    rm -f "${context_file}"
    exit 1
  fi

  cp "${src_model}" "${tmp_model}"
  write_yaml_from_context "${context_file}" "${tmp_meta}"
  mv "${tmp_model}" "${target_model}"
  mv "${tmp_meta}" "${target_meta}"
  rm -f "${context_file}"

  log_ok "已写入模型: ${target_model}"
  log_ok "已写入 YAML: ${target_meta}"
}

sync_model_only_to_device_share() {
  resolve_device_and_customer
  local target_dir
  target_dir="$(device_share_dir)"
  mkdir -p "${target_dir}"

  local task_key
  if [[ -n "${TASK:-}" && "${TASK}" != "detection" ]]; then
    task_key="$(normalize_task_key "${TASK}")"
  else
    task_key="$(infer_task_from_config)"
  fi

  local src_model="${MODEL_PATH:-}"
  local src_meta="${CLASS_NAMES_FILE:-}"

  # MODEL_PATH / CLASS_NAMES_FILE 的默认值是远端 /opt/visionops 路径，
  # model-only 本地同步时不能使用远端路径，因此自动切换到本地 export 目录。
  if [[ -z "${src_model}" || "${src_model}" == /opt/visionops/* || ! -f "${src_model}" ]]; then
    src_model="$(default_local_model_for_task "${task_key}")"
  fi
  if [[ -z "${src_meta}" || "${src_meta}" == /opt/visionops/* || ! -f "${src_meta}" ]]; then
    src_meta="$(default_local_meta_for_task "${task_key}" allow_runtime_fallback)"
  fi

  local version_name
  version_name="$(build_version_name "${task_key}")"

  log_info "单模型同步到 Syncthing 共享目录"
  log_info "device_id=${DEVICE_ID}"
  log_info "customer_id=${CUSTOMER_ID}"
  log_info "target_dir=${target_dir}"
  log_info "task=${task_key}"
  log_info "version=${version_name}"
  log_info "model=${src_model}"
  log_info "meta_source=${src_meta:-<自动从 data.yaml / metrics 生成>}"

  copy_pair_to_share "${src_model}" "${target_dir}" "${version_name}" "${task_key}" "${src_meta}"
  log_warn "只写入 .rknn 和 .yaml，不生成 .sha256，不生成 .READY，不通过 SSH 重启边缘端。"
}

build_roi_bundle_context() {
  local det_model="$1"
  local cls_model="$2"
  local version_name="$3"

  export VISIONOPS_REPO_ROOT_FOR_DEPLOY="${REPO_ROOT}"
  export VISIONOPS_ROI_DET_MODEL="${det_model}"
  export VISIONOPS_ROI_CLS_MODEL="${cls_model}"
  export VISIONOPS_ROI_VERSION="${version_name}"
  export VISIONOPS_ROI_DEVICE_ID="${DEVICE_ID}"
  export VISIONOPS_ROI_CUSTOMER_ID="${CUSTOMER_ID}"
  export VISIONOPS_ROI_DET_MD5="$(get_file_md5 "${det_model}")"
  export VISIONOPS_ROI_CLS_MD5="$(get_file_md5 "${cls_model}")"
  export VISIONOPS_ROI_DET_SIZE="$(get_file_size_bytes "${det_model}")"
  export VISIONOPS_ROI_CLS_SIZE="$(get_file_size_bytes "${cls_model}")"
  export VISIONOPS_ROI_CONF_THRESHOLD="${CONF_THRESHOLD}"
  export VISIONOPS_ROI_NMS_THRESHOLD="${NMS_THRESHOLD}"
  export VISIONOPS_ROI_TOPK="${TOPK}"
  export VISIONOPS_ROI_SESSION_MANIFEST="${REPO_ROOT}/data/roi_classification_sessions/current/manifest.json"

  python3 - <<'PY'
from pathlib import Path
from datetime import datetime
import json
import os
import sys

try:
    import yaml  # type: ignore
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"缺少 PyYAML: {exc}"}, ensure_ascii=False))
    raise SystemExit(0)

root = Path(os.environ["VISIONOPS_REPO_ROOT_FOR_DEPLOY"]).resolve()
det_model = Path(os.environ["VISIONOPS_ROI_DET_MODEL"])
cls_model = Path(os.environ["VISIONOPS_ROI_CLS_MODEL"])
version_name = os.environ["VISIONOPS_ROI_VERSION"]
device_id = os.environ["VISIONOPS_ROI_DEVICE_ID"]
customer_id = os.environ.get("VISIONOPS_ROI_CUSTOMER_ID") or "CUST-000"
roi_session_manifest = Path(os.environ.get("VISIONOPS_ROI_SESSION_MANIFEST", ""))

def fail(msg):
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    raise SystemExit(0)

def load_json_optional(path):
    try:
        if path.exists() and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}

def load_yaml_optional(path):
    try:
        if path.exists() and path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}

def normalize_names(v):
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, dict):
        def key_sort(k):
            try:
                return int(k)
            except Exception:
                return str(k)
        return [str(v[k]) for k in sorted(v.keys(), key=key_sort)]
    return []

def first_nonempty(*vals):
    for v in vals:
        if v is not None and str(v).strip() != "":
            return v
    return ""

def raw_data_names(suffix):
    data = load_yaml_optional(root / "data" / f"raw_{suffix}" / "data.yaml")
    return normalize_names(data.get("names"))

def classification_raw_names():
    raw = root / "data" / "raw_classification"
    return [p.name for p in sorted(raw.iterdir()) if p.is_dir()] if raw.exists() else []

det_metrics_path = root / "models" / "metrics_detection" / "eval_metrics.json"
cls_metrics_path = root / "models" / "metrics_classification" / "eval_metrics.json"
det_metrics = load_json_optional(det_metrics_path)
cls_metrics = load_json_optional(cls_metrics_path)
det_meta = load_yaml_optional(root / "models" / "export_detection" / "model.yaml")
cls_meta = load_yaml_optional(root / "models" / "export_classification" / "model.yaml")
manifest_path = root / "data" / "model_context" / "manifest.json"
manifest = load_json_optional(manifest_path)
roi_session = load_json_optional(roi_session_manifest)

det_names = (
    normalize_names(det_metrics.get("class_names"))
    or normalize_names(det_metrics.get("names"))
    or normalize_names(det_meta.get("class_names"))
    or normalize_names(det_meta.get("names"))
    or raw_data_names("detection")
)
cls_names = (
    normalize_names(cls_metrics.get("class_names"))
    or normalize_names(cls_metrics.get("names"))
    or normalize_names(cls_meta.get("class_names"))
    or normalize_names(cls_meta.get("names"))
    or classification_raw_names()
)
if not det_names:
    fail("无法获取检测模型类别名：请检查 models/metrics_detection/eval_metrics.json、models/export_detection/model.yaml 或 data/raw_detection/data.yaml")
if not cls_names:
    fail("无法获取分类模型类别名：请检查 models/metrics_classification/eval_metrics.json、models/export_classification/model.yaml 或 data/raw_classification/<类别名>")

det_num = int(first_nonempty(det_metrics.get("num_classes"), det_meta.get("num_classes"), len(det_names)))
cls_num = int(first_nonempty(cls_metrics.get("num_classes"), cls_meta.get("num_classes"), len(cls_names)))
det_num = len(det_names) if det_num != len(det_names) else det_num
cls_num = len(cls_names) if cls_num != len(cls_names) else cls_num

def as_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)

def as_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)

def normalize_rel_box(v):
    if not isinstance(v, dict):
        v = {}
    def f(key, default):
        try:
            x = float(v.get(key, default))
        except Exception:
            x = float(default)
        return max(0.0, min(1.0, x))
    x1, y1, x2, y2 = f("x1", 0.0), f("y1", 0.0), f("x2", 1.0), f("y2", 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

def build_roi_runtime_policy(session):
    session_padding = first_nonempty(session.get("padding_ratio"), os.environ.get("VISIONOPS_ROI_CLS_PADDING"), 0.05)
    session_padding = as_float(session_padding, 0.05)
    src_policy = session.get("roi_policy") if isinstance(session.get("roi_policy"), dict) else {}
    by_class_src = src_policy.get("by_detector_class") if isinstance(src_policy.get("by_detector_class"), dict) else {}

    default_src = src_policy.get("default") if isinstance(src_policy.get("default"), dict) else {}
    default_padding = as_float(default_src.get("padding_ratio", session_padding), session_padding)

    by_class = {}
    for key, entry in by_class_src.items():
        if not isinstance(entry, dict):
            continue
        enabled = bool(entry.get("enabled", False))
        padding = as_float(entry.get("padding_ratio", default_padding), default_padding)
        by_class[str(key)] = {
            "enabled": enabled,
            "mode": str(entry.get("mode") or ("relative_box" if enabled else "full_box")),
            "base": str(entry.get("base") or "det_bbox_with_padding"),
            "padding_ratio": padding,
            "relative_box": normalize_rel_box(entry.get("relative_box")),
            "det_class_id": entry.get("det_class_id"),
            "det_class_name": entry.get("det_class_name"),
            "class_key": str(entry.get("class_key") or key),
            "coordinate": str(entry.get("coordinate") or "relative_to_padded_detection_box"),
            "updated_at": entry.get("updated_at", ""),
        }

    if by_class:
        return {
            "schema_version": int(src_policy.get("schema_version") or 1),
            "mode": str(src_policy.get("mode") or "class_relative_box"),
            "coordinate": str(src_policy.get("coordinate") or "relative_to_padded_detection_box"),
            "default": {
                "enabled": bool(default_src.get("enabled", False)),
                "mode": str(default_src.get("mode") or "full_box"),
                "base": str(default_src.get("base") or "det_bbox_with_padding"),
                "padding_ratio": default_padding,
                "relative_box": normalize_rel_box(default_src.get("relative_box")),
            },
            "by_detector_class": by_class,
            "source_manifest": str(roi_session_manifest.relative_to(root) if roi_session_manifest.exists() else roi_session_manifest),
            "updated_at": src_policy.get("updated_at", session.get("updated_at", "")),
        }

    return {
        "mode": "full_box",
        "padding_ratio": session_padding,
        "source_manifest": str(roi_session_manifest.relative_to(root) if roi_session_manifest.exists() else ""),
    }

source_device_id = first_nonempty(manifest.get("device_id"), manifest.get("equipment_id"), manifest.get("edge_device_id"))
dataset_customer = first_nonempty(manifest.get("customer_id"), manifest.get("customer"), manifest.get("cust_id"), customer_id, "CUST-000")
det_conf = as_float(first_nonempty(os.environ.get("VISIONOPS_ROI_CONF_THRESHOLD"), roi_session.get("conf_threshold"), det_metrics.get("conf_threshold"), det_meta.get("conf_threshold"), 0.25), 0.25)
det_nms = as_float(first_nonempty(os.environ.get("VISIONOPS_ROI_NMS_THRESHOLD"), det_metrics.get("nms_threshold"), det_meta.get("nms_threshold"), 0.45), 0.45)
cls_topk = as_int(first_nonempty(os.environ.get("VISIONOPS_ROI_TOPK"), cls_meta.get("topk"), min(5, cls_num)), min(5, cls_num))
cls_topk = max(1, min(cls_topk, cls_num))

ctx = {
    "ok": True,
    "task": "roi_classification",
    "version_name": version_name,
    "device_id": str(device_id),
    "customer_id": str(dataset_customer),
    "detector": {
        "source_model": str(det_model),
        "source_metrics": str(det_metrics_path) if det_metrics_path.exists() else "",
        "file": "detection.rknn",
        "meta_file": "detection.yaml",
        "md5": os.environ["VISIONOPS_ROI_DET_MD5"],
        "size_bytes": int(os.environ["VISIONOPS_ROI_DET_SIZE"]),
        "num_classes": det_num,
        "class_names": det_names,
        "input_size": [640, 640],
        "conf_threshold": det_conf,
        "nms_threshold": det_nms,
        "select_policy": str(roi_session.get("select_policy") or "conf_area"),
        "target_class_id": roi_session.get("target_class_id"),
        "target_class_name": roi_session.get("target_class_name"),
        "metrics": det_metrics,
    },
    "classifier": {
        "source_model": str(cls_model),
        "source_metrics": str(cls_metrics_path) if cls_metrics_path.exists() else "",
        "file": "classification.rknn",
        "meta_file": "classification.yaml",
        "md5": os.environ["VISIONOPS_ROI_CLS_MD5"],
        "size_bytes": int(os.environ["VISIONOPS_ROI_CLS_SIZE"]),
        "num_classes": cls_num,
        "class_names": cls_names,
        "input_size": [224, 224],
        "topk": cls_topk,
        "metrics": cls_metrics,
    },
    "roi": build_roi_runtime_policy(roi_session),
    "dataset": {
        "manifest_path": str(manifest_path.relative_to(root) if manifest_path.exists() else manifest_path),
        "device_id": str(device_id),
        "source_device_id": str(source_device_id or ""),
        "customer_id": str(dataset_customer),
        "dataset_id": first_nonempty(manifest.get("dataset_id"), manifest.get("package_id"), manifest.get("batch_id")),
        "roi_session_manifest": str(roi_session_manifest.relative_to(root) if roi_session_manifest.exists() else ""),
        "roi_session_summary": {
            "session_id": roi_session.get("session_id", ""),
            "batch_id": roi_session.get("batch_id", ""),
            "padding_ratio": roi_session.get("padding_ratio", None),
            "select_policy": roi_session.get("select_policy", ""),
            "roi_policy_updated_at": (roi_session.get("roi_policy") or {}).get("updated_at", "") if isinstance(roi_session.get("roi_policy"), dict) else "",
        },
        "raw_manifest": manifest,
    },
    "deploy": {
        "deployed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deployed_by": "edge/deploy/deploy_cpp.sh --model-only --roi-classification",
        "target_device": str(device_id),
    },
}
print(json.dumps(ctx, ensure_ascii=False))
PY
}

write_roi_bundle_files() {
  local context_file="$1"
  local bundle_dir="$2"

  python3 - <<'PY' "$context_file" "$bundle_dir"
from pathlib import Path
import json
import shutil
import sys
import yaml

ctx = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
bundle_dir = Path(sys.argv[2])
bundle_dir.mkdir(parents=True, exist_ok=True)

det = ctx["detector"]
cls = ctx["classifier"]

shutil.copy2(det["source_model"], bundle_dir / det["file"])
shutil.copy2(cls["source_model"], bundle_dir / cls["file"])

detector_meta = {
    "schema_version": 1,
    "task": "detection",
    "input_size": det["input_size"],
    "num_classes": det["num_classes"],
    "class_names": det["class_names"],
    "conf_threshold": det["conf_threshold"],
    "nms_threshold": det["nms_threshold"],
    "model": {
        "file": det["file"],
        "md5": det["md5"],
        "size_bytes": det["size_bytes"],
        "source_path": det["source_model"],
    },
    "metrics": det["metrics"],
}

classifier_meta = {
    "schema_version": 1,
    "task": "classification",
    "input_size": cls["input_size"],
    "num_classes": cls["num_classes"],
    "class_names": cls["class_names"],
    "topk": cls["topk"],
    "model": {
        "file": cls["file"],
        "md5": cls["md5"],
        "size_bytes": cls["size_bytes"],
        "source_path": cls["source_model"],
        "architecture": cls["metrics"].get("architecture", ""),
    },
    "metrics": cls["metrics"],
}

pipeline = {
    "schema_version": 1,
    "pipeline_type": "roi_classification",
    "task": "roi_classification",
    "pipeline_name": ctx["version_name"],
    "stage1": {
        "name": "detector",
        "task": "detection",
        "model_path": det["file"],
        "meta_path": det["meta_file"],
        "input_size": det["input_size"],
        "num_classes": det["num_classes"],
        "class_names": det["class_names"],
        "conf_threshold": det["conf_threshold"],
        "nms_threshold": det["nms_threshold"],
        "select_policy": det.get("select_policy", "conf_area"),
        "target_class_id": det.get("target_class_id", 0 if det["num_classes"] == 1 else None),
        "target_class_name": det.get("target_class_name", det["class_names"][0] if det["num_classes"] == 1 else ""),
    },
    "roi": ctx["roi"],
    "stage2": {
        "name": "classifier",
        "task": "classification",
        "model_path": cls["file"],
        "meta_path": cls["meta_file"],
        "input_size": cls["input_size"],
        "num_classes": cls["num_classes"],
        "class_names": cls["class_names"],
        "topk": cls["topk"],
    },
    "decision": {
        "low_det_conf_policy": "REVIEW",
        "bad_roi_policy": "REVIEW",
        "low_cls_conf_policy": "REVIEW",
    },
    "dataset": ctx["dataset"],
    "deploy": ctx["deploy"],
}

bundle_manifest = {
    "schema_version": 1,
    "bundle_type": "roi_classification",
    "name": ctx["version_name"],
    "device_id": ctx["device_id"],
    "customer_id": ctx["customer_id"],
    "created_at": ctx["deploy"]["deployed_at"],
    "files": {
        "detection_model": det["file"],
        "detection_meta": det["meta_file"],
        "classification_model": cls["file"],
        "classification_meta": cls["meta_file"],
        "pipeline_config": "pipeline.yaml",
    },
    "dataset": ctx["dataset"],
    "deploy": ctx["deploy"],
}

(bundle_dir / det["meta_file"]).write_text(yaml.safe_dump(detector_meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
(bundle_dir / cls["meta_file"]).write_text(yaml.safe_dump(classifier_meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
(bundle_dir / "pipeline.yaml").write_text(yaml.safe_dump(pipeline, allow_unicode=True, sort_keys=False), encoding="utf-8")
(bundle_dir / "bundle_manifest.json").write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

sync_roi_classification_bundle_to_device_share() {
  resolve_device_and_customer
  local target_root
  target_root="$(device_share_dir)"

  local version_name
  version_name="$(build_version_name roi_classification)"
  local bundle_dir="${target_root}/${version_name}"
  local tmp_bundle="${target_root}/.${version_name}.$$.tmp"

  log_info "ROI 分类双模型 bundle 同步到 Syncthing 共享目录"
  log_info "device_id=${DEVICE_ID}"
  log_info "customer_id=${CUSTOMER_ID}"
  log_info "bundle=${bundle_dir}"

  local det_model cls_model
  det_model="$(default_local_model_for_task detection)"
  cls_model="$(default_local_model_for_task classification)"
  [[ -f "${det_model}" ]] || { log_error "检测 RKNN 不存在: ${det_model}"; exit 1; }
  [[ -f "${cls_model}" ]] || { log_error "分类 RKNN 不存在: ${cls_model}"; exit 1; }

  mkdir -p "${target_root}"
  rm -rf "${tmp_bundle}"
  mkdir -p "${tmp_bundle}"

  local context_file ok err
  context_file="$(mktemp /tmp/visionops_cpp_roi_context_XXXXXX.json)"
  build_roi_bundle_context "${det_model}" "${cls_model}" "${version_name}" > "${context_file}"

  ok="$(python3 - <<'PY' "$context_file"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("ok"))
PY
)"
  if [[ "${ok}" != "True" && "${ok}" != "true" ]]; then
    err="$(python3 - <<'PY' "$context_file"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("error", "生成 ROI bundle 失败"))
PY
)"
    log_error "${err}"
    rm -f "${context_file}"
    rm -rf "${tmp_bundle}"
    exit 1
  fi

  write_roi_bundle_files "${context_file}" "${tmp_bundle}"

  rm -rf "${bundle_dir}"
  mv "${tmp_bundle}" "${bundle_dir}"
  rm -f "${context_file}"

  log_ok "已写入 ROI 分类双模型 bundle: ${bundle_dir}"
  find "${bundle_dir}" -maxdepth 1 -type f -printf "[OK] %p\n" | sort
  log_warn "只写入本地 Syncthing 共享目录；不通过 SSH 安装、不重启边缘端、不生成 .sha256 / .READY。"
}

validate_deploy_mode() {
  case "${DEPLOY_MODE}" in
    all|code-only|model-only) ;;
    *) log_error "DEPLOY_MODE must be one of: all, code-only, model-only. Current: ${DEPLOY_MODE}"; exit 1 ;;
  esac
}

safe_model_stem() {
  local value="$1"
  value="${value##*/}"
  value="${value%.rknn}"
  value="${value%.yaml}"
  value="${value%.yml}"
  value="$(echo "${value}" | sed -E 's/[^A-Za-z0-9_-]+/-/g; s/^-+//; s/-+$//')"
  [[ -n "${value}" ]] || value="visionops_model_$(date +%Y%m%d_%H%M%S)"
  echo "${value}"
}

sync_model_pair_to_share_dir() {
  local src_model="$1"
  local src_meta="$2"
  local stem="${3:-}"
  local share_dir="$4"

  if [[ ! -f "${src_model}" ]]; then
    local auto_model=""
    for candidate in       "${REPO_ROOT}/models/export_detection/model.rknn"       "${REPO_ROOT}/models/export_classification/model.rknn"       "${REPO_ROOT}/models/export_obb/model.rknn"       "${REPO_ROOT}/models/export_segmentation/model.rknn"       "${REPO_ROOT}/models/export/model.rknn"; do
      if [[ -f "${candidate}" ]]; then auto_model="${candidate}"; break; fi
    done
    if [[ -n "${auto_model}" ]]; then
      log_warn "Local --model not found (${src_model}); auto use ${auto_model}"
      src_model="${auto_model}"
    fi
  fi

  if [[ ! -f "${src_meta}" ]]; then
    local auto_meta=""
    for candidate in       "${REPO_ROOT}/edge/runtime/class_names.yaml"       "${REPO_ROOT}/models/export_detection/model.yaml"       "${REPO_ROOT}/models/export_classification/model.yaml"       "${REPO_ROOT}/models/export_obb/model.yaml"       "${REPO_ROOT}/models/export_segmentation/model.yaml"       "${REPO_ROOT}/models/export/model.yaml"; do
      if [[ -f "${candidate}" ]]; then auto_meta="${candidate}"; break; fi
    done
    if [[ -n "${auto_meta}" ]]; then
      log_warn "Local --class-names-file not found (${src_meta}); auto use ${auto_meta}"
      src_meta="${auto_meta}"
    fi
  fi

  [[ -f "${src_model}" ]] || { log_error "Local RKNN file not found for --model-only: ${src_model}"; exit 1; }
  [[ -f "${src_meta}" ]] || { log_error "Local YAML file not found for --model-only: ${src_meta}"; exit 1; }

  if [[ -z "${stem}" ]]; then
    stem="$(safe_model_stem "${src_model}")"
  else
    stem="$(safe_model_stem "${stem}")"
  fi

  mkdir -p "${share_dir}"
  local target_model="${share_dir}/${stem}.rknn"
  local target_meta="${share_dir}/${stem}.yaml"
  local tmp_model="${share_dir}/.${stem}.rknn.$$.tmp"
  local tmp_meta="${share_dir}/.${stem}.yaml.$$.tmp"

  cp "${src_model}" "${tmp_model}"
  cp "${src_meta}" "${tmp_meta}"
  mv "${tmp_model}" "${target_model}"
  mv "${tmp_meta}" "${target_meta}"

  log_ok "Model-only deploy wrote Syncthing share files"
  log_ok "model=${target_model}"
  log_ok "meta=${target_meta}"
  log_warn "Only .rknn and .yaml are written. No .sha256 and no .READY file are generated."
}

sync_edge_code_only() {
  if [[ ! -d "${REPO_ROOT}/edge" ]]; then
    log_error "Missing ${REPO_ROOT}/edge. Run this script from the VisionOps repository."
    exit 1
  fi
  require_cmd ssh
  require_cmd rsync

  log_info "Target: ${EDGE_USER}@${EDGE_HOST}:${EDGE_PORT}"
  remote "echo connected >/dev/null"
  if ! remote "sudo -n true"; then
    log_error "Remote sudo requires password. Please configure NOPASSWD sudo for ${EDGE_USER}."
    exit 1
  fi

  log_info "Sync local edge/ to ${INSTALL_DIR}/edge only"
  remote_sudo mkdir -p "${INSTALL_DIR}/edge"
  remote_sudo chown -R "${EDGE_USER}:${EDGE_USER}" "${INSTALL_DIR}/edge" || true
  remote "sudo -n find '${INSTALL_DIR}/edge' -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true"
  remote "sudo -n find '${INSTALL_DIR}/edge' -type f -name '*.pyc' -delete 2>/dev/null || true"

  rsync -az --delete -e "${RSYNC_SSH}" \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'runtime/cpp.env' \
    --exclude 'runtime/edge.env' \
    --exclude 'runtime/cpp_proxy.env' \
    --exclude 'runtime/runtime_overrides.yaml' \
    "${REPO_ROOT}/edge/" \
    "${EDGE_USER}@${EDGE_HOST}:${INSTALL_DIR}/edge/"

  log_ok "edge/ code sync completed. No compile, env rewrite, service restart, or health check was performed."
}


main() {
  validate_deploy_mode

  if [[ "${DEPLOY_MODE}" == "model-only" ]]; then
    if [[ "$(normalize_bool "${ROI_CLASSIFICATION}")" == "1" ]]; then
      sync_roi_classification_bundle_to_device_share
    else
      sync_model_only_to_device_share
    fi
    exit 0
  fi

  if [[ "${DEPLOY_MODE}" == "code-only" ]]; then
    sync_edge_code_only
    exit 0
  fi

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
  log_info "C++ port: ${CPP_PORT}, task=${TASK}, num_classes=${num_classes_final}, input_size=${INPUT_SIZE}, topk=${TOPK}, preprocess_backend=${PREPROCESS_BACKEND}, rga_mode=${RGA_MODE}, snapshot=${ENABLE_SNAPSHOT}, annotated=${ENABLE_ANNOTATED}"
  log_info "Camera: type=${CAMERA_TYPE}, source=${CAMERA_SOURCE}, width=${CAMERA_WIDTH}, height=${CAMERA_HEIGHT}, fps=${CAMERA_FPS}, fourcc=${CAMERA_FOURCC}, buffer_size=${CAMERA_BUFFER_SIZE}, stream_backend=${STREAM_BACKEND}"

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