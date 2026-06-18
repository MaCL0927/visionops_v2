#!/usr/bin/env bash
set -euo pipefail

ROBOT_GATEWAY_DIR="/opt/visionops/edge/robot_gateway"
RUNTIME_DIR="/opt/visionops/edge/runtime"
PYTHON_BIN="/opt/visionops/venv/bin/python"
PARTITION_ENV="$ROBOT_GATEWAY_DIR/carton_partition_check/partition_check.env"
TUBE_ENV="$ROBOT_GATEWAY_DIR/carton_tube_check/carton_tube_check.env"
CPP_PARTITION_ENV="$RUNTIME_DIR/cpp-partition.env"
PARTITION_PORT="${VISIONOPS_PARTITION_CPP_PORT:-8091}"
ROBOT_PORT="${VISIONOPS_ROBOT_PROTOCOL_PORT:-5045}"

PARTITION_MODEL="/opt/visionops/models/rk3576-001_paper-cell_det_20260610_155540.rknn"
PARTITION_YAML="/opt/visionops/models/rk3576-001_paper-cell_det_20260610_155540.yaml"
TUBE_MODEL="/opt/visionops/models/rk3576-001_tube_new_obb_20260609_181148.rknn"
TUBE_YAML="/opt/visionops/models/rk3576-001_tube_new_obb_20260609_181148.yaml"

log(){ echo "[INFO] $*"; }
warn(){ echo "[WARN] $*"; }
fail(){ echo "[ERROR] $*" >&2; exit 1; }

replace_or_append() {
  local file="$1" key="$2" value="$3"
  mkdir -p "$(dirname "$file")"
  touch "$file"
  if grep -q "^${key}=" "$file"; then
    sed -i "s#^${key}=.*#${key}=${value}#" "$file"
  else
    printf '\n%s=%s\n' "$key" "$value" >> "$file"
  fi
}

log "VisionOps robot protocol runtime-split installer"
log "ROBOT_GATEWAY_DIR=$ROBOT_GATEWAY_DIR"
log "RUNTIME_DIR=$RUNTIME_DIR"
log "Original web-controlled C++ service: visionops-inference-cpp.service + $RUNTIME_DIR/cpp.env"
log "Dedicated partition C++ service: visionops-inference-cpp-partition.service + $CPP_PARTITION_ENV"

[ -d "$ROBOT_GATEWAY_DIR" ] || fail "robot_gateway dir not found: $ROBOT_GATEWAY_DIR"
[ -d "$RUNTIME_DIR" ] || fail "runtime dir not found: $RUNTIME_DIR"
[ -f "$PARTITION_MODEL" ] || fail "partition model missing: $PARTITION_MODEL"
[ -f "$PARTITION_YAML" ] || fail "partition yaml missing: $PARTITION_YAML"
[ -f "$TUBE_MODEL" ] || warn "tube model missing: $TUBE_MODEL"
[ -f "$TUBE_YAML" ] || warn "tube yaml missing: $TUBE_YAML"

log "remove bad direct systemd drop-ins from previous attempts"
rm -f /etc/systemd/system/visionops-inference-cpp.service.d/20-tube-direct.conf || true
rm -f /etc/systemd/system/visionops-inference-cpp-partition.service.d/10-carton-partition-model.conf || true
rm -f /etc/systemd/system/visionops-inference-cpp-partition.service.d/20-partition-direct.conf || true

log "install fixed partition runtime env: $CPP_PARTITION_ENV"
SRC_CPP_PARTITION_ENV=""
if [ -f "$ROBOT_GATEWAY_DIR/../runtime/cpp-partition.env" ]; then
  SRC_CPP_PARTITION_ENV="$ROBOT_GATEWAY_DIR/../runtime/cpp-partition.env"
elif [ -f "$ROBOT_GATEWAY_DIR/runtime/cpp-partition.env" ]; then
  SRC_CPP_PARTITION_ENV="$ROBOT_GATEWAY_DIR/runtime/cpp-partition.env"
fi
if [ -n "$SRC_CPP_PARTITION_ENV" ]; then
  if [ "$(readlink -f "$SRC_CPP_PARTITION_ENV")" != "$(readlink -f "$CPP_PARTITION_ENV" 2>/dev/null || echo "$CPP_PARTITION_ENV")" ]; then
    cp -f "$SRC_CPP_PARTITION_ENV" "$CPP_PARTITION_ENV"
  fi
else
  cat > "$CPP_PARTITION_ENV" <<EOC
VISIONOPS_CPP_BIN=/opt/visionops/bin/visionops_inference_cpp
VISIONOPS_CPP_MODEL_PATH=$PARTITION_MODEL
VISIONOPS_CPP_CLASS_NAMES_FILE=$PARTITION_YAML
VISIONOPS_CPP_TASK=detection
VISIONOPS_CPP_HOST=0.0.0.0
VISIONOPS_CPP_PORT=$PARTITION_PORT
VISIONOPS_CPP_NPU_CORE=auto
VISIONOPS_CPP_NUM_CLASSES=1
VISIONOPS_CPP_INPUT_SIZE=640,640
VISIONOPS_CPP_CONF_THRESHOLD=0.25
VISIONOPS_CPP_NMS_THRESHOLD=0.45
VISIONOPS_CPP_MASK_THRESHOLD=0.5
VISIONOPS_CPP_TOPK=5
VISIONOPS_CPP_MAX_DET=100
VISIONOPS_CPP_OUTPUT_MODE=float
VISIONOPS_CPP_PREPROCESS_BACKEND=auto
VISIONOPS_CPP_RGA_MODE=resize_color
VISIONOPS_CPP_CAMERA_READ_FPS=10
VISIONOPS_CPP_DETECT_FPS=1
VISIONOPS_CPP_SNAPSHOT_FPS=10
VISIONOPS_CPP_SNAPSHOT_JPEG_QUALITY=80
VISIONOPS_CPP_ENABLE_SNAPSHOT=True
VISIONOPS_CPP_ENABLE_ANNOTATED=True
VISIONOPS_CPP_STREAM_BACKEND=opencv
VISIONOPS_CPP_STREAM_CODEC=h264
VISIONOPS_CPP_GST_LATENCY_MS=100
VISIONOPS_CPP_RTSP_TRANSPORT=tcp
VISIONOPS_CPP_RTSP_TIMEOUT_MS=5000
VISIONOPS_CPP_QUIET_FFMPEG_LOG=True
VISIONOPS_CPP_STREAM_AUTO_START=False
VISIONOPS_CPP_CAMERA_TYPE=auto
VISIONOPS_CPP_CAMERA_WIDTH=0
VISIONOPS_CPP_CAMERA_HEIGHT=0
VISIONOPS_CPP_CAMERA_FPS=0
VISIONOPS_CPP_CAMERA_FOURCC=
VISIONOPS_CPP_CAMERA_BUFFER_SIZE=1
VISIONOPS_CPP_CAMERA_SOURCE=http://127.0.0.1:18181/stream.mjpeg
EOC
fi
replace_or_append "$CPP_PARTITION_ENV" "VISIONOPS_CPP_MODEL_PATH" "$PARTITION_MODEL"
replace_or_append "$CPP_PARTITION_ENV" "VISIONOPS_CPP_CLASS_NAMES_FILE" "$PARTITION_YAML"
replace_or_append "$CPP_PARTITION_ENV" "VISIONOPS_CPP_TASK" "detection"
replace_or_append "$CPP_PARTITION_ENV" "VISIONOPS_CPP_PORT" "$PARTITION_PORT"
replace_or_append "$CPP_PARTITION_ENV" "VISIONOPS_CPP_NUM_CLASSES" "1"

log "patch robot task env files"
replace_or_append "$PARTITION_ENV" "VISIONOPS_PARTITION_INFER_URL" "http://127.0.0.1:${PARTITION_PORT}/api/cpp/infer"
replace_or_append "$PARTITION_ENV" "VISIONOPS_PARTITION_EXPECTED_MODEL" "$PARTITION_MODEL"
replace_or_append "$PARTITION_ENV" "VISIONOPS_PARTITION_EXPECTED_MODEL_YAML" "$PARTITION_YAML"
replace_or_append "$TUBE_ENV" "VISIONOPS_CARTON_TUBE_INFER_URL" "http://127.0.0.1:8090/api/cpp/infer"
replace_or_append "$TUBE_ENV" "VISIONOPS_CARTON_TUBE_EXPECTED_MODEL" "$TUBE_MODEL"
replace_or_append "$TUBE_ENV" "VISIONOPS_CARTON_TUBE_EXPECTED_MODEL_YAML" "$TUBE_YAML"

log "install partition inference launcher"
install -m 755 "$ROBOT_GATEWAY_DIR/start_visionops_inference_cpp_from_env.sh" /opt/visionops/edge/robot_gateway/start_visionops_inference_cpp_from_env.sh

log "install visionops-inference-cpp-partition.service"
cat > /etc/systemd/system/visionops-inference-cpp-partition.service <<EOF2
[Unit]
Description=VisionOps RK3576 C++ RKNN Inference Service - Carton Partition Fixed Runtime
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/visionops
ExecStart=/bin/bash /opt/visionops/edge/robot_gateway/start_visionops_inference_cpp_from_env.sh $CPP_PARTITION_ENV
Restart=always
RestartSec=3
StartLimitIntervalSec=60
StartLimitBurst=10
TimeoutStartSec=30
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-inference-cpp-partition

[Install]
WantedBy=multi-user.target
EOF2

log "install/refresh robot protocol service"
cat > /etc/systemd/system/visionops-robot-protocol.service <<EOF2
[Unit]
Description=VisionOps Robot Protocol Modbus TCP Service
After=network-online.target visionops-inference-cpp.service visionops-inference-cpp-partition.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$ROBOT_GATEWAY_DIR
Environment=VISIONOPS_ROBOT_PROTOCOL_ENV=$ROBOT_GATEWAY_DIR/vision_robot_protocol.env
Environment=VISIONOPS_PARTITION_ENV=$PARTITION_ENV
Environment=VISIONOPS_CARTON_TUBE_ENV=$TUBE_ENV
ExecStart=$PYTHON_BIN $ROBOT_GATEWAY_DIR/vision_robot_protocol_modbus_tcp.py
Restart=always
RestartSec=3
StartLimitIntervalSec=60
StartLimitBurst=10
TimeoutStartSec=30
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-robot-protocol

[Install]
WantedBy=multi-user.target
EOF2

log "disable old single-task Modbus services"
systemctl disable --now visionops-carton-partition-check.service 2>/dev/null || true
systemctl disable --now visionops-carton-tube-check.service 2>/dev/null || true

log "reload and restart services"
systemctl daemon-reload

# The original service must continue to use /opt/visionops/edge/runtime/cpp.env so Web UI model switching remains valid.
systemctl enable visionops-inference-cpp.service >/dev/null 2>&1 || true
systemctl restart visionops-inference-cpp.service

systemctl enable visionops-inference-cpp-partition.service >/dev/null 2>&1 || true
systemctl restart visionops-inference-cpp-partition.service

systemctl enable visionops-robot-protocol.service >/dev/null 2>&1 || true
systemctl restart visionops-robot-protocol.service

log "done"
echo "[CHECK] systemctl status visionops-inference-cpp.service --no-pager -l"
echo "[CHECK] systemctl status visionops-inference-cpp-partition.service --no-pager -l"
echo "[CHECK] systemctl status visionops-robot-protocol.service --no-pager -l"
echo "[CHECK] ss -lntp | grep -E '8090|8091|5045|18080'"
echo "[NOTE] visionops-inference-cpp.service is intentionally left Web-controlled via /opt/visionops/edge/runtime/cpp.env."
