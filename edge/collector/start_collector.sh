#!/usr/bin/env bash
set -euo pipefail

cd /opt/visionops/edge/collector

if [[ -f /opt/visionops/edge/runtime/collector.env ]]; then
  set -a
  source /opt/visionops/edge/runtime/collector.env
  set +a
fi

exec /opt/visionops/venv/bin/python /opt/visionops/edge/collector/app.py
