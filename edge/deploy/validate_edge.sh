#!/usr/bin/env bash
set -euo pipefail

COLLECTOR_PORT="${VISIONOPS_COLLECTOR_PORT:-8090}"
HP60C_PORT="${VISIONOPS_HP60C_PORT:-18181}"
CPP_PORT="${VISIONOPS_CPP_PORT:-18080}"
TUBE_PORT="${VISIONOPS_TUBE_MODBUS_PORT:-1502}"

echo "========== services =========="
systemctl list-units --type=service --all | grep -i visionops || true

echo
echo "========== ports =========="
ss -ltnp | grep -E "${COLLECTOR_PORT}|${HP60C_PORT}|${CPP_PORT}|${TUBE_PORT}" || true

echo
echo "========== hp60c sdk bridge =========="
curl -s "http://127.0.0.1:${HP60C_PORT}/health" | python3 -m json.tool || true

echo
echo "========== collector hp60c proxy =========="
curl -s "http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/hp60c_sdk/health" | python3 -m json.tool || true

echo
echo "========== c++ inference =========="
curl -s "http://127.0.0.1:${CPP_PORT}/health" | python3 -m json.tool || true

echo
echo "========== snapshot =========="
curl -fsS -o /tmp/visionops_hp60c_validate.jpg "http://127.0.0.1:${HP60C_PORT}/stream/snapshot.jpg" && ls -lh /tmp/visionops_hp60c_validate.jpg || true

echo
echo "========== legacy rtsp residue =========="
grep -R "192.168.2.64\|rtsp://admin:Abcd123_" -n \
  /opt/visionops/edge/runtime \
  /opt/visionops/edge/collector \
  /etc/systemd/system \
  2>/dev/null || true
