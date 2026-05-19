#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/visionops}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
DST_DIR="${PROJECT_ROOT}/edge/robot_gateway/modbus_tcp"
COMMON_SRC_DIR="$(realpath "${SRC_DIR}/../modbus_common" 2>/dev/null || true)"
COMMON_DST_DIR="${PROJECT_ROOT}/edge/robot_gateway/modbus_common"

echo "[INFO] install VisionOps Modbus TCP Server"
echo "[INFO] SRC_DIR=${SRC_DIR}"
echo "[INFO] DST_DIR=${DST_DIR}"
echo "[INFO] COMMON_DST_DIR=${COMMON_DST_DIR}"

mkdir -p "${DST_DIR}"
mkdir -p "${COMMON_DST_DIR}"

if [ "$(realpath "${SRC_DIR}")" != "$(realpath "${DST_DIR}")" ]; then
  echo "[INFO] copy files to ${DST_DIR}"
  cp -f "${SRC_DIR}/modbus_tcp_server.py" "${DST_DIR}/"
  cp -f "${SRC_DIR}/modbus_tcp.env" "${DST_DIR}/"
  cp -f "${SRC_DIR}/register_map.md" "${DST_DIR}/"
  if [ -f "${SRC_DIR}/test_modbus_tcp_client.py" ]; then
    cp -f "${SRC_DIR}/test_modbus_tcp_client.py" "${DST_DIR}/"
  fi
else
  echo "[INFO] SRC_DIR and DST_DIR are the same, skip file copy."
fi


if [ -n "${COMMON_SRC_DIR}" ] && [ -d "${COMMON_SRC_DIR}" ]; then
  echo "[INFO] copy modbus_common to ${COMMON_DST_DIR}"
  cp -f "${COMMON_SRC_DIR}"/*.py "${COMMON_DST_DIR}/"
  cp -f "${COMMON_SRC_DIR}"/register_map_v2.md "${COMMON_DST_DIR}/"
else
  echo "[WARN] modbus_common source dir not found beside ${SRC_DIR}; skip common copy"
fi

cp -f "${SRC_DIR}/visionops-modbus-tcp.service" /etc/systemd/system/visionops-modbus-tcp.service

chmod +x "${DST_DIR}/modbus_tcp_server.py"
if [ -f "${DST_DIR}/test_modbus_tcp_client.py" ]; then
  chmod +x "${DST_DIR}/test_modbus_tcp_client.py"
fi

if [ ! -x "${PROJECT_ROOT}/venv/bin/python" ]; then
  echo "[ERROR] python venv not found: ${PROJECT_ROOT}/venv/bin/python"
  exit 1
fi

echo "[INFO] install python dependencies"
"${PROJECT_ROOT}/venv/bin/python" -m pip install "pymodbus==3.6.9"

echo "[INFO] reload systemd"
systemctl daemon-reload
systemctl enable visionops-modbus-tcp.service

echo "[INFO] done."
echo "Next:"
echo "  1) edit ${DST_DIR}/modbus_tcp.env if needed"
echo "  2) systemctl restart visionops-modbus-tcp.service"
echo "  3) journalctl -u visionops-modbus-tcp.service -f"
echo "  4) from PC: mbpoll -m tcp -a 1 -r 1 -c 20 192.168.1.202 -p 1502"
