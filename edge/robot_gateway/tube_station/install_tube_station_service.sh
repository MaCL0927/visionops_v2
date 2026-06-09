#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="visionops-tube-station.service"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIONOPS_ROOT="${VISIONOPS_ROOT:-/opt/visionops}"
DST_DIR="$VISIONOPS_ROOT/edge/robot_gateway/tube_station"
PYTHON_BIN="${VISIONOPS_PYTHON:-$VISIONOPS_ROOT/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

echo "[INFO] install VisionOps Tube Station Modbus TCP service"
echo "[INFO] SRC_DIR=$SRC_DIR"
echo "[INFO] DST_DIR=$DST_DIR"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"

sudo mkdir -p "$DST_DIR"

copy_if_different() {
  local src="$1"
  local dst="$2"
  # 如果已经在目标目录执行安装，src 和 dst 可能是同一个文件；此时不复制，避免 cp 报 “are the same file”。
  if [[ "$(readlink -f "$src")" == "$(readlink -f "$dst")" ]]; then
    echo "[INFO] skip copy same file: $dst"
    return 0
  fi
  sudo cp -f "$src" "$dst"
}

copy_if_different "$SRC_DIR/tube_station_modbus_tcp.py" "$DST_DIR/tube_station_modbus_tcp.py"
copy_if_different "$SRC_DIR/test_tube_station_client.py" "$DST_DIR/test_tube_station_client.py"
copy_if_different "$SRC_DIR/register_map.md" "$DST_DIR/register_map.md"
if [[ ! -f "$DST_DIR/tube_station.env" ]]; then
  copy_if_different "$SRC_DIR/tube_station.env" "$DST_DIR/tube_station.env"
else
  echo "[INFO] keep existing $DST_DIR/tube_station.env"
fi
sudo chmod +x "$DST_DIR/tube_station_modbus_tcp.py" "$DST_DIR/test_tube_station_client.py"

# Install dependency into VisionOps venv/system python.
"$PYTHON_BIN" - <<'PY' || true
try:
    import pymodbus
    print('[OK] pymodbus already installed')
except Exception:
    raise SystemExit(1)
PY
if ! "$PYTHON_BIN" - <<'PY'
import pymodbus
PY
then
  echo "[INFO] installing pymodbus"
  "$PYTHON_BIN" -m pip install 'pymodbus>=2.5,<4'
fi

sudo tee "/etc/systemd/system/$SERVICE_NAME" > /dev/null <<EOF
[Unit]
Description=VisionOps Tube Station Modbus TCP Trigger Service
After=network-online.target visionops-hp60c-ros1-bridge.service visionops-inference-cpp.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$DST_DIR
Environment=VISIONOPS_TUBE_ENV=$DST_DIR/tube_station.env
ExecStart=$PYTHON_BIN $DST_DIR/tube_station_modbus_tcp.py
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
