#!/usr/bin/env bash
set -euo pipefail

# Keep this wrapper separate from systemd so optional args such as camera source
# can be omitted cleanly when they are empty.

if [[ -f /opt/visionops/.env ]]; then
  # shellcheck disable=SC1091
  source /opt/visionops/.env || true
fi

if [[ -f /opt/visionops/edge/runtime/cpp.env ]]; then
  # shellcheck disable=SC1091
  source /opt/visionops/edge/runtime/cpp.env || true
fi

: "${VISIONOPS_CPP_BIN:=/opt/visionops/bin/visionops_inference_cpp}"
: "${VISIONOPS_CPP_MODEL_PATH:=/opt/visionops/models/rk3588-001_CUST-000_det_20260427_105932.rknn}"
: "${VISIONOPS_CPP_CLASS_NAMES_FILE:=/opt/visionops/models/rk3588-001_CUST-000_det_20260427_105932.yaml}"
: "${VISIONOPS_CPP_TASK:=detection}"
: "${VISIONOPS_CPP_PORT:=18080}"
: "${VISIONOPS_CPP_NPU_CORE:=auto}"
: "${VISIONOPS_CPP_NUM_CLASSES:=80}"
: "${VISIONOPS_CPP_INPUT_SIZE:=640,640}"
: "${VISIONOPS_CPP_CONF_THRESHOLD:=0.25}"
: "${VISIONOPS_CPP_NMS_THRESHOLD:=0.45}"
: "${VISIONOPS_CPP_TOPK:=5}"
: "${VISIONOPS_CPP_MAX_DET:=100}"
: "${VISIONOPS_CPP_OUTPUT_MODE:=float}"
: "${VISIONOPS_CPP_PREPROCESS_BACKEND:=auto}"
: "${VISIONOPS_CPP_RGA_MODE:=resize_color}"
: "${VISIONOPS_CPP_CAMERA_READ_FPS:=10}"
: "${VISIONOPS_CPP_DETECT_FPS:=10}"
: "${VISIONOPS_CPP_SNAPSHOT_FPS:=1}"
: "${VISIONOPS_CPP_SNAPSHOT_JPEG_QUALITY:=80}"
: "${VISIONOPS_CPP_ENABLE_SNAPSHOT:=true}"
: "${VISIONOPS_CPP_ENABLE_ANNOTATED:=true}"
: "${VISIONOPS_CPP_STREAM_BACKEND:=opencv}"
: "${VISIONOPS_CPP_STREAM_CODEC:=h264}"
: "${VISIONOPS_CPP_GST_LATENCY_MS:=100}"
: "${VISIONOPS_CPP_RTSP_TRANSPORT:=tcp}"
: "${VISIONOPS_CPP_RTSP_TIMEOUT_MS:=5000}"
: "${VISIONOPS_CPP_QUIET_FFMPEG_LOG:=true}"
: "${VISIONOPS_CPP_STREAM_AUTO_START:=false}"

cmd=(
  "$VISIONOPS_CPP_BIN"
  --model "$VISIONOPS_CPP_MODEL_PATH"
  --class-names-file "$VISIONOPS_CPP_CLASS_NAMES_FILE"
  --task "$VISIONOPS_CPP_TASK"
  --host "0.0.0.0"
  --port "$VISIONOPS_CPP_PORT"
  --npu-core "$VISIONOPS_CPP_NPU_CORE"
  --num-classes "$VISIONOPS_CPP_NUM_CLASSES"
  --input-size "$VISIONOPS_CPP_INPUT_SIZE"
  --conf-threshold "$VISIONOPS_CPP_CONF_THRESHOLD"
  --nms-threshold "$VISIONOPS_CPP_NMS_THRESHOLD"
  --topk "$VISIONOPS_CPP_TOPK"
  --max-det "$VISIONOPS_CPP_MAX_DET"
  --output-mode "$VISIONOPS_CPP_OUTPUT_MODE"
  --preprocess-backend "$VISIONOPS_CPP_PREPROCESS_BACKEND"
  --rga-mode "$VISIONOPS_CPP_RGA_MODE"
  --camera-read-fps "$VISIONOPS_CPP_CAMERA_READ_FPS"
  --detect-fps "$VISIONOPS_CPP_DETECT_FPS"
  --snapshot-fps "$VISIONOPS_CPP_SNAPSHOT_FPS"
  --snapshot-jpeg-quality "$VISIONOPS_CPP_SNAPSHOT_JPEG_QUALITY"
  --enable-snapshot "$VISIONOPS_CPP_ENABLE_SNAPSHOT"
  --enable-annotated "$VISIONOPS_CPP_ENABLE_ANNOTATED"
  --stream-backend "$VISIONOPS_CPP_STREAM_BACKEND"
  --stream-codec "$VISIONOPS_CPP_STREAM_CODEC"
  --gst-latency-ms "$VISIONOPS_CPP_GST_LATENCY_MS"
  --rtsp-transport "$VISIONOPS_CPP_RTSP_TRANSPORT"
  --rtsp-timeout-ms "$VISIONOPS_CPP_RTSP_TIMEOUT_MS"
  --quiet-ffmpeg-log "$VISIONOPS_CPP_QUIET_FFMPEG_LOG"
  --stream-auto-start "$VISIONOPS_CPP_STREAM_AUTO_START"
)

if [[ -n "${VISIONOPS_CAMERA_SOURCE:-}" ]]; then
  cmd+=(--camera-source "$VISIONOPS_CAMERA_SOURCE")
elif [[ -n "${CAMERA_SOURCE:-}" ]]; then
  cmd+=(--camera-source "$CAMERA_SOURCE")
fi

echo "[START] ${cmd[*]}"
exec "${cmd[@]}"
