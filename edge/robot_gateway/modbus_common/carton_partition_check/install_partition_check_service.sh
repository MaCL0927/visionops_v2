#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="visionops-carton-partition-check.service"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIONOPS_ROOT="${VISIONOPS_ROOT:-/opt/visionops}"
DST_DIR="$VISIONOPS_ROOT/edge/robot_gateway/carton_partition_check"
PYTHON_BIN="${VISIONOPS_PYTHON:-$VISIONOPS_ROOT/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

echo "[INFO] install VisionOps Carton Partition Cell Check Modbus TCP service"
echo "[INFO] SRC_DIR=$SRC_DIR"
echo "[INFO] DST_DIR=$DST_DIR"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"

sudo mkdir -p "$DST_DIR"

copy_if_different() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    return 0
  fi
  if [[ "$(readlink -f "$src")" == "$(readlink -f "$dst" 2>/dev/null || true)" ]]; then
    return 0
  fi
  sudo cp "$src" "$dst"
}

copy_if_different "$SRC_DIR/partition_check_modbus_tcp.py" "$DST_DIR/partition_check_modbus_tcp.py"
copy_if_different "$SRC_DIR/debug_partition_check_once.py" "$DST_DIR/debug_partition_check_once.py"
copy_if_different "$SRC_DIR/partition_check.env" "$DST_DIR/partition_check.env"
copy_if_different "$SRC_DIR/test_partition_check_client.py" "$DST_DIR/test_partition_check_client.py"
copy_if_different "$SRC_DIR/register_map.md" "$DST_DIR/register_map.md"
copy_if_different "$SRC_DIR/README.md" "$DST_DIR/README.md"

sudo chmod +x "$DST_DIR/partition_check_modbus_tcp.py" "$DST_DIR/debug_partition_check_once.py" "$DST_DIR/test_partition_check_client.py"

sudo tee "/etc/systemd/system/$SERVICE_NAME" >/dev/null <<EOF
[Unit]
Description=VisionOps Carton Partition Cell Check Modbus TCP Service
After=network-online.target visionops-hp60c-ros1-bridge.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$DST_DIR
Environment=VISIONOPS_PARTITION_ENV=$DST_DIR/partition_check.env
ExecStart=$PYTHON_BIN $DST_DIR/partition_check_modbus_tcp.py
Restart=always
RestartSec=3
User=root

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
echo "[OK] installed $SERVICE_NAME"
echo "[NEXT] sudo systemctl restart $SERVICE_NAME"
echo "[NEXT] sudo journalctl -u $SERVICE_NAME -f -o cat"
echo "[NOTE] default Modbus-TCP port is 1504. Change VISIONOPS_PARTITION_MODBUS_PORT in $DST_DIR/partition_check.env if needed."
echo "[NOTE] run calibration first: cd $DST_DIR && VISIONOPS_PARTITION_ENV=./partition_check.env $PYTHON_BIN debug_partition_check_once.py --calibrate --save-dir /tmp/carton_partition_calib"