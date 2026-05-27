#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/visionops}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
DST_DIR="${PROJECT_ROOT}/edge/robot_gateway/modbus_rtu_master"
COMMON_SRC_DIR="$(realpath "${SRC_DIR}/../modbus_common" 2>/dev/null || true)"
COMMON_DST_DIR="${PROJECT_ROOT}/edge/robot_gateway/modbus_common"

echo "[INFO] install VisionOps Modbus RTU Master Push"
echo "[INFO] SRC_DIR=${SRC_DIR}"
echo "[INFO] DST_DIR=${DST_DIR}"
echo "[INFO] COMMON_DST_DIR=${COMMON_DST_DIR}"

mkdir -p "${DST_DIR}"
mkdir -p "${COMMON_DST_DIR}"

if [ "$(realpath "${SRC_DIR}")" != "$(realpath "${DST_DIR}")" ]; then
  echo "[INFO] copy files to ${DST_DIR}"
  cp -f "${SRC_DIR}/modbus_rtu_master_push.py" "${DST_DIR}/"
  cp -f "${SRC_DIR}/modbus_rtu_master.env" "${DST_DIR}/"
  cp -f "${SRC_DIR}/register_map_master_push.md" "${DST_DIR}/"
else
  echo "[INFO] SRC_DIR and DST_DIR are the same, skip file copy."
fi

if [ -n "${COMMON_SRC_DIR}" ] && [ -d "${COMMON_SRC_DIR}" ]; then
  if [ "$(realpath "${COMMON_SRC_DIR}")" != "$(realpath "${COMMON_DST_DIR}")" ]; then
    echo "[INFO] copy modbus_common to ${COMMON_DST_DIR}"
    cp -f "${COMMON_SRC_DIR}"/*.py "${COMMON_DST_DIR}/"
    if [ -f "${COMMON_SRC_DIR}/register_map_v2.md" ]; then
      cp -f "${COMMON_SRC_DIR}/register_map_v2.md" "${COMMON_DST_DIR}/"
    fi
  else
    echo "[INFO] COMMON_SRC_DIR and COMMON_DST_DIR are the same, skip common copy."
  fi
else
  echo "[WARN] modbus_common source dir not found beside ${SRC_DIR}; skip common copy"
fi

cp -f "${SRC_DIR}/visionops-modbus-rtu-master.service" /etc/systemd/system/visionops-modbus-rtu-master.service
chmod +x "${DST_DIR}/modbus_rtu_master_push.py"

if [ ! -x "${PROJECT_ROOT}/venv/bin/python" ]; then
  echo "[ERROR] python venv not found: ${PROJECT_ROOT}/venv/bin/python"
  exit 1
fi

echo "[INFO] install python dependencies"
"${PROJECT_ROOT}/venv/bin/python" -m pip install pyserial

echo "[INFO] reload systemd"
systemctl daemon-reload
systemctl enable visionops-modbus-rtu-master.service

echo "[INFO] done."
echo "Next:"
echo "  1) edit ${DST_DIR}/modbus_rtu_master.env"
echo "  2) stop RTU slave if using same /dev/ttyS5: systemctl stop visionops-modbus-rtu.service"
echo "  3) systemctl restart visionops-modbus-rtu-master.service"
echo "  4) journalctl -u visionops-modbus-rtu-master.service -f -o cat"
