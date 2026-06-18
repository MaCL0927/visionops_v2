#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="visionops-robot-protocol.service"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VISIONOPS_PYTHON_BIN:-/opt/visionops/venv/bin/python}"
WORKDIR="${VISIONOPS_ROBOT_GATEWAY_DIR:-/opt/visionops/edge/robot_gateway}"
ENV_FILE="$WORKDIR/vision_robot_protocol.env"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=VisionOps Robot Protocol Modbus TCP Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$WORKDIR
Environment=VISIONOPS_ROBOT_PROTOCOL_ENV=$ENV_FILE
Environment=VISIONOPS_PARTITION_ENV=$WORKDIR/carton_partition_check/partition_check.env
Environment=VISIONOPS_CARTON_TUBE_ENV=$WORKDIR/carton_tube_check/carton_tube_check.env
ExecStart=$PYTHON_BIN $WORKDIR/vision_robot_protocol_modbus_tcp.py
Restart=always
RestartSec=2
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
echo "[OK] installed $SERVICE_NAME"
echo "[NEXT] sudo systemctl restart $SERVICE_NAME"
echo "[NEXT] sudo journalctl -u $SERVICE_NAME -f -o cat"
