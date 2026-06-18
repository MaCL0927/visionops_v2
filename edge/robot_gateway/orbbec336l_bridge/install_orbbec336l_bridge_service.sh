#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="visionops-orbbec336l-bridge.service"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST_DIR="/opt/visionops/edge/robot_gateway/orbbec336l_bridge"
BIN_DIR="/opt/visionops/bin"
ENV_FILE="$DST_DIR/orbbec336l_bridge.env"
BIN_PATH="$BIN_DIR/visionops_orbbec336l_bridge"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err() { echo "[ERROR] $*" >&2; }

log "install VisionOps Orbbec Gemini 336L SDK bridge"
log "SRC_DIR=$SRC_DIR"
log "DST_DIR=$DST_DIR"

sudo mkdir -p "$DST_DIR" "$BIN_DIR"

# Only copy files into this directory. Do not remove robot_gateway or other bridge folders.
# If installer is already running from DST_DIR, skip self-copy to avoid:
# cp: source and destination are the same file
if [[ "$SRC_DIR" != "$DST_DIR" ]]; then
  sudo cp -f "$SRC_DIR/visionops_orbbec336l_bridge.cpp" "$DST_DIR/"
  sudo cp -f "$SRC_DIR/CMakeLists.txt" "$DST_DIR/"
  sudo cp -f "$SRC_DIR/orbbec336l_bridge.env" "$DST_DIR/"
  sudo cp -f "$SRC_DIR/README.md" "$DST_DIR/" 2>/dev/null || true
  sudo cp -f "$SRC_DIR/install_orbbec336l_bridge_service.sh" "$DST_DIR/"
else
  log "SRC_DIR and DST_DIR are the same; skip file self-copy."
fi
sudo chmod +x "$DST_DIR/install_orbbec336l_bridge_service.sh"

if [[ ! -f "$ENV_FILE" ]]; then
  err "missing env file: $ENV_FILE"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

find_first_dir() {
  for p in "$@"; do
    # allow simple globs
    for q in $p; do
      if [[ -d "$q" ]]; then
        echo "$q"
        return 0
      fi
    done
  done
  return 1
}

SDK_ROOT="${VISIONOPS_ORBBEC336L_SDK_ROOT:-}"
if [[ -z "$SDK_ROOT" || ! -d "$SDK_ROOT" ]]; then
  SDK_ROOT="$(find_first_dir \
    /opt/OrbbecSDK \
    /opt/orbbec/OrbbecSDK \
    /opt/OrbbecSDK_v* \
    /usr/local/OrbbecSDK \
    /home/neardi/OrbbecSDK \
    /home/neardi/OrbbecSDK_v* \
    /home/neardi/OrbbecSDK*/sdk \
    2>/dev/null || true)"
fi

if [[ -z "$SDK_ROOT" || ! -d "$SDK_ROOT" ]]; then
  err "Orbbec SDK root not found. Edit $ENV_FILE and set VISIONOPS_ORBBEC336L_SDK_ROOT."
  exit 2
fi

INCLUDE_DIR="${VISIONOPS_ORBBEC336L_SDK_INCLUDE_DIR:-$SDK_ROOT/include}"
if [[ ! -d "$INCLUDE_DIR" ]]; then
  # Sometimes SDK is nested one level down.
  INCLUDE_DIR="$(find "$SDK_ROOT" -maxdepth 3 -type d -name include | head -1 || true)"
fi
if [[ -z "$INCLUDE_DIR" || ! -d "$INCLUDE_DIR" ]]; then
  err "Orbbec SDK include dir not found under $SDK_ROOT"
  exit 3
fi

LIB_DIR="${VISIONOPS_ORBBEC336L_SDK_LIB_DIR:-$SDK_ROOT/lib}"
if [[ ! -d "$LIB_DIR" ]]; then
  LIB_DIR="$(find "$SDK_ROOT" -maxdepth 4 -type f \( -name 'libOrbbecSDK.so' -o -name 'libOrbbecSDK.so.*' \) -printf '%h\n' | head -1 || true)"
fi
if [[ -z "$LIB_DIR" || ! -d "$LIB_DIR" ]]; then
  err "Orbbec SDK lib dir not found under $SDK_ROOT"
  exit 4
fi

if [[ ! -f "$INCLUDE_DIR/libobsensor/ObSensor.hpp" && ! -f "$INCLUDE_DIR/ObSensor.hpp" ]]; then
  warn "ObSensor.hpp was not found in the usual include path: $INCLUDE_DIR"
  warn "Compilation may fail; set VISIONOPS_ORBBEC336L_SDK_INCLUDE_DIR in $ENV_FILE if needed."
fi

log "SDK_ROOT=$SDK_ROOT"
log "INCLUDE_DIR=$INCLUDE_DIR"
log "LIB_DIR=$LIB_DIR"

sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libopencv-dev

BUILD_DIR="$DST_DIR/build"
sudo rm -rf "$BUILD_DIR"
sudo mkdir -p "$BUILD_DIR"
sudo chown -R "$(id -u):$(id -g)" "$BUILD_DIR"

cd "$BUILD_DIR"
cmake .. \
  -DORBBEC_SDK_ROOT="$SDK_ROOT" \
  -DORBBEC_INCLUDE_DIR="$INCLUDE_DIR" \
  -DORBBEC_LIB_DIR="$LIB_DIR"
make -j"$(nproc)"

sudo cp -f visionops_orbbec336l_bridge "$BIN_PATH"
sudo chmod +x "$BIN_PATH"

# Keep the resolved SDK paths in env for later reinstall/debug.
sudo sed -i "s#^VISIONOPS_ORBBEC336L_SDK_ROOT=.*#VISIONOPS_ORBBEC336L_SDK_ROOT=$SDK_ROOT#" "$ENV_FILE"
sudo sed -i "s#^VISIONOPS_ORBBEC336L_SDK_INCLUDE_DIR=.*#VISIONOPS_ORBBEC336L_SDK_INCLUDE_DIR=$INCLUDE_DIR#" "$ENV_FILE"
sudo sed -i "s#^VISIONOPS_ORBBEC336L_SDK_LIB_DIR=.*#VISIONOPS_ORBBEC336L_SDK_LIB_DIR=$LIB_DIR#" "$ENV_FILE"

sudo tee "/etc/systemd/system/$SERVICE_NAME" >/dev/null <<EOF_SERVICE
[Unit]
Description=VisionOps Orbbec Gemini 336L SDK HTTP Bridge
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$DST_DIR
EnvironmentFile=$ENV_FILE
Environment=LD_LIBRARY_PATH=$LIB_DIR
ExecStart=$BIN_PATH
Restart=always
RestartSec=3
StartLimitIntervalSec=60
StartLimitBurst=10
TimeoutStartSec=30
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-orbbec336l-bridge

[Install]
WantedBy=multi-user.target
EOF_SERVICE

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

log "installed $SERVICE_NAME"
log "NEXT: sudo systemctl restart $SERVICE_NAME"
log "CHECK: curl -s http://127.0.0.1:${VISIONOPS_ORBBEC336L_HTTP_PORT:-18182}/health | python3 -m json.tool"
