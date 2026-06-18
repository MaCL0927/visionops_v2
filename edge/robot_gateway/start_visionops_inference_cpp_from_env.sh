#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-/opt/visionops/edge/runtime/cpp-partition.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "[ERROR] env file not found: $ENV_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

BIN="${VISIONOPS_CPP_BIN:-/opt/visionops/bin/visionops_inference_cpp}"
MODEL="${VISIONOPS_CPP_MODEL_PATH:-}"
CLASS_NAMES="${VISIONOPS_CPP_CLASS_NAMES_FILE:-${VISIONOPS_CPP_MODEL_YAML:-}}"
TASK="${VISIONOPS_CPP_TASK:-detection}"
HOST="${VISIONOPS_CPP_HOST:-0.0.0.0}"
PORT="${VISIONOPS_CPP_PORT:-8091}"

if [ ! -x "$BIN" ]; then
  echo "[ERROR] inference binary not executable: $BIN" >&2
  exit 3
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "[ERROR] model not found: $MODEL" >&2
  exit 4
fi
if [ -z "$CLASS_NAMES" ] || [ ! -f "$CLASS_NAMES" ]; then
  echo "[ERROR] class names yaml not found: $CLASS_NAMES" >&2
  exit 5
fi

args=(
  "$BIN"
  --model "$MODEL"
  --class-names-file "$CLASS_NAMES"
  --task "$TASK"
  --host "$HOST"
  --port "$PORT"
  --npu-core "${VISIONOPS_CPP_NPU_CORE:-auto}"
  --num-classes "${VISIONOPS_CPP_NUM_CLASSES:-1}"
  --input-size "${VISIONOPS_CPP_INPUT_SIZE:-640,640}"
  --conf-threshold "${VISIONOPS_CPP_CONF_THRESHOLD:-0.25}"
  --nms-threshold "${VISIONOPS_CPP_NMS_THRESHOLD:-0.45}"
  --mask-threshold "${VISIONOPS_CPP_MASK_THRESHOLD:-0.5}"
  --topk "${VISIONOPS_CPP_TOPK:-5}"
  --max-det "${VISIONOPS_CPP_MAX_DET:-100}"
  --output-mode "${VISIONOPS_CPP_OUTPUT_MODE:-float}"
  --preprocess-backend "${VISIONOPS_CPP_PREPROCESS_BACKEND:-auto}"
  --rga-mode "${VISIONOPS_CPP_RGA_MODE:-resize_color}"
  --camera-read-fps "${VISIONOPS_CPP_CAMERA_READ_FPS:-10}"
  --detect-fps "${VISIONOPS_CPP_DETECT_FPS:-1}"
  --snapshot-fps "${VISIONOPS_CPP_SNAPSHOT_FPS:-10}"
  --snapshot-jpeg-quality "${VISIONOPS_CPP_SNAPSHOT_JPEG_QUALITY:-80}"
  --enable-snapshot "${VISIONOPS_CPP_ENABLE_SNAPSHOT:-True}"
  --enable-annotated "${VISIONOPS_CPP_ENABLE_ANNOTATED:-True}"
  --stream-backend "${VISIONOPS_CPP_STREAM_BACKEND:-opencv}"
  --stream-codec "${VISIONOPS_CPP_STREAM_CODEC:-h264}"
  --gst-latency-ms "${VISIONOPS_CPP_GST_LATENCY_MS:-100}"
  --rtsp-transport "${VISIONOPS_CPP_RTSP_TRANSPORT:-tcp}"
  --rtsp-timeout-ms "${VISIONOPS_CPP_RTSP_TIMEOUT_MS:-5000}"
  --quiet-ffmpeg-log "${VISIONOPS_CPP_QUIET_FFMPEG_LOG:-True}"
  --stream-auto-start "${VISIONOPS_CPP_STREAM_AUTO_START:-False}"
  --camera-type "${VISIONOPS_CPP_CAMERA_TYPE:-auto}"
  --camera-width "${VISIONOPS_CPP_CAMERA_WIDTH:-0}"
  --camera-height "${VISIONOPS_CPP_CAMERA_HEIGHT:-0}"
  --camera-fps "${VISIONOPS_CPP_CAMERA_FPS:-0}"
  --camera-buffer-size "${VISIONOPS_CPP_CAMERA_BUFFER_SIZE:-1}"
  --camera-source "${VISIONOPS_CPP_CAMERA_SOURCE:-http://127.0.0.1:18181/stream.mjpeg}"
)

if [ -n "${VISIONOPS_CPP_CAMERA_FOURCC:-}" ]; then
  args+=(--camera-fourcc "$VISIONOPS_CPP_CAMERA_FOURCC")
fi
if [ -n "${VISIONOPS_CPP_PIPELINE_CONFIG:-}" ]; then
  args+=(--pipeline-config "$VISIONOPS_CPP_PIPELINE_CONFIG")
fi

echo "[START] ${args[*]}"
exec "${args[@]}"
