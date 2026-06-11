#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="visionops-hp60c-sdk-bridge.service"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST_DIR="/opt/visionops/edge/robot_gateway/hp60c_sdk_bridge"
BIN_DIR="/opt/visionops/bin"
ENV_FILE="$DST_DIR/hp60c_sdk_bridge.env"
SDK_ROOT_DEFAULT="/home/neardi/AngstrongCameraSdk_v1.2.61.20250910/demo/linux_ros"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err() { echo "[ERROR] $*" >&2; }

log "install VisionOps HP60C Angstrong SDK bridge"
log "SRC_DIR=$SRC_DIR"
log "DST_DIR=$DST_DIR"

sudo mkdir -p "$DST_DIR" "$BIN_DIR"

if [[ "$SRC_DIR" != "$DST_DIR" ]]; then
  sudo cp -f "$SRC_DIR/visionops_hp60c_sdk_bridge.cpp" "$DST_DIR/"
  sudo cp -f "$SRC_DIR/CMakeLists.txt" "$DST_DIR/"
  sudo cp -f "$SRC_DIR/hp60c_sdk_bridge.env" "$DST_DIR/"
  sudo cp -f "$SRC_DIR/README.md" "$DST_DIR/" 2>/dev/null || true
  sudo cp -f "$SRC_DIR/install_hp60c_sdk_bridge_service.sh" "$DST_DIR/"
else
  log "source and destination are the same; skip self-copy"
fi

if [[ ! -f "$ENV_FILE" ]]; then
  err "missing env file: $ENV_FILE"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
SDK_ROOT="${VISIONOPS_HP60C_SDK_ROOT:-$SDK_ROOT_DEFAULT}"
SDK_LIB_DIR="${VISIONOPS_HP60C_SDK_LIB_DIR:-$SDK_ROOT/libs/lib/aarch64-linux-gnu-gcc-5}"
CONFIG_FILE="${VISIONOPS_HP60C_CONFIG:-$SDK_ROOT/configurationfiles/hp60c_v2_01_20241104_configEncrypt.json}"

if [[ ! -d "$SDK_ROOT" ]]; then
  err "SDK root not found: $SDK_ROOT"
  err "Edit $ENV_FILE and set VISIONOPS_HP60C_SDK_ROOT to the demo/linux_ros directory."
  exit 2
fi
if [[ ! -d "$SDK_LIB_DIR" ]]; then
  err "SDK lib dir not found: $SDK_LIB_DIR"
  err "Edit $ENV_FILE and set VISIONOPS_HP60C_SDK_LIB_DIR."
  exit 3
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  err "HP60C config file not found: $CONFIG_FILE"
  err "Edit $ENV_FILE and set VISIONOPS_HP60C_CONFIG."
  exit 4
fi

sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libopencv-dev

# Install vendor udev rules if present.
if [[ -f "$SDK_ROOT/scripts/create_udev_rules.sh" ]]; then
  log "install Angstrong udev rules"
  (cd "$SDK_ROOT/scripts" && sudo bash ./create_udev_rules.sh || true)
  sudo udevadm control --reload-rules || true
  sudo udevadm trigger || true
else
  warn "udev script not found: $SDK_ROOT/scripts/create_udev_rules.sh"
fi

BUILD_DIR="$DST_DIR/build"
sudo rm -rf "$BUILD_DIR"
sudo mkdir -p "$BUILD_DIR"
sudo chown -R "$(id -u):$(id -g)" "$BUILD_DIR"

cd "$BUILD_DIR"
cmake .. \
  -DANGSTRONG_SDK_ROOT="$SDK_ROOT" \
  -DANGSTRONG_LIB_DIR="$SDK_LIB_DIR"
make -j"$(nproc)"
sudo cp -f visionops_hp60c_sdk_bridge "$BIN_DIR/visionops_hp60c_sdk_bridge"
sudo chmod +x "$BIN_DIR/visionops_hp60c_sdk_bridge"

sudo tee "/etc/systemd/system/$SERVICE_NAME" > /dev/null <<EOF
[Unit]
Description=VisionOps HP60C Angstrong SDK HTTP Bridge
After=network-online.target
Wants=network-online.target
Conflicts=visionops-hp60c-ros1-bridge.service

[Service]
Type=simple
WorkingDirectory=$DST_DIR
EnvironmentFile=$ENV_FILE
Environment=LD_LIBRARY_PATH=$SDK_LIB_DIR
ExecStart=$BIN_DIR/visionops_hp60c_sdk_bridge
Restart=always
RestartSec=3
User=root

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

log "installed $SERVICE_NAME"
log "Before starting, stop ROS bridge if it is running:"
log "  sudo systemctl stop visionops-hp60c-ros1-bridge.service 2>/dev/null || true"
log "Start SDK bridge:"
log "  sudo systemctl restart $SERVICE_NAME"
log "Check:"
log "  curl -s http://127.0.0.1:18181/health | python3 -m json.tool"
