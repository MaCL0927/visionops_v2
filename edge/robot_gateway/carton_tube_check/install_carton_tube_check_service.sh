#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="visionops-carton-tube-check.service"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIONOPS_ROOT="${VISIONOPS_ROOT:-/opt/visionops}"
DST_DIR="$VISIONOPS_ROOT/edge/robot_gateway/carton_tube_check"
PYTHON_BIN="${VISIONOPS_PYTHON:-$VISIONOPS_ROOT/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

echo "[INFO] install VisionOps Carton Tube Check Modbus TCP service"
echo "[INFO] SRC_DIR=$SRC_DIR"
echo "[INFO] DST_DIR=$DST_DIR"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"

# Ensure Python dependencies are installed into the same interpreter used by systemd.
# Do NOT rely on plain `pip install ...`, because systemd runs $PYTHON_BIN.
echo "[INFO] install/check Python deps in $PYTHON_BIN"
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import pymodbus
PY
then
  "$PYTHON_BIN" -m pip install --no-cache-dir "pymodbus==3.6.9" requests numpy opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

sudo mkdir -p "$DST_DIR"

copy_if_different() {
  local src="$1"
  local dst="$2"
  if [[ "$(readlink -f "$src")" == "$(readlink -f "$dst" 2>/dev/null || true)" ]]; then
    return 0
  fi
  sudo cp "$src" "$dst"
}

copy_if_different "$SRC_DIR/carton_tube_check_modbus_tcp.py" "$DST_DIR/carton_tube_check_modbus_tcp.py"
copy_if_different "$SRC_DIR/debug_depth_check_once.py" "$DST_DIR/debug_depth_check_once.py"
copy_if_different "$SRC_DIR/carton_tube_check.env" "$DST_DIR/carton_tube_check.env"
copy_if_different "$SRC_DIR/test_carton_tube_check_client.py" "$DST_DIR/test_carton_tube_check_client.py"
copy_if_different "$SRC_DIR/register_map.md" "$DST_DIR/register_map.md"
copy_if_different "$SRC_DIR/README_STAGE2_DEBUG.md" "$DST_DIR/README_STAGE2_DEBUG.md" 2>/dev/null || true
copy_if_different "$SRC_DIR/README_MODBUS_SERVICE.md" "$DST_DIR/README_MODBUS_SERVICE.md" 2>/dev/null || true

sudo chmod +x "$DST_DIR/carton_tube_check_modbus_tcp.py" "$DST_DIR/debug_depth_check_once.py" "$DST_DIR/test_carton_tube_check_client.py"

sudo tee "/etc/systemd/system/$SERVICE_NAME" >/dev/null <<EOF
[Unit]
Description=VisionOps Carton Tube Check Modbus TCP Service
After=network-online.target visionops-hp60c-ros1-bridge.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$DST_DIR
Environment=VISIONOPS_CARTON_TUBE_ENV=$DST_DIR/carton_tube_check.env
ExecStart=$PYTHON_BIN $DST_DIR/carton_tube_check_modbus_tcp.py
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
echo "[NOTE] default Modbus-TCP port is 1503. Change VISIONOPS_CARTON_TUBE_MODBUS_PORT in $DST_DIR/carton_tube_check.env if needed."
