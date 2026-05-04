#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-}"
META_PATH="${2:-}"
PORT="${3:-8082}"
SERVICE_NAME="${4:-visionops-inference}"
INSTALL_DIR="${VISIONOPS_INSTALL_DIR:-/opt/visionops}"
EDGE_DIR="${INSTALL_DIR}/edge"
RUNTIME_DIR="${EDGE_DIR}/runtime"
ENV_FILE="${INSTALL_DIR}/.env"
RUNTIME_ENV_FILE="${RUNTIME_DIR}/edge.env"
STOP_SCRIPT="${EDGE_DIR}/deploy/stop_inference.sh"
HEALTH_URL="http://localhost:${PORT}/health"

log() { echo "[switch_model] $*"; }
err() { echo "[switch_model][ERROR] $*" >&2; }

if [[ -z "${MODEL_PATH}" || -z "${META_PATH}" ]]; then
  err "用法: bash switch_model.sh <model.rknn> <meta.yaml> [port] [service_name]"
  exit 2
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  err "模型不存在: ${MODEL_PATH}"
  exit 3
fi
if [[ ! -f "${META_PATH}" ]]; then
  err "meta 不存在: ${META_PATH}"
  exit 4
fi
if ! command -v python3 >/dev/null 2>&1; then
  err "缺少 python3"
  exit 5
fi
if ! command -v sudo >/dev/null 2>&1 || ! sudo -n true 2>/dev/null; then
  err "当前用户没有免密 sudo，无法切换 systemd 推理服务。请先配置 NOPASSWD sudo。"
  exit 11
fi

read_meta_json() {
  python3 - <<'PY' "${META_PATH}" "${PORT}" "${MODEL_PATH}"
import json, sys
from pathlib import Path
try:
    import yaml
except Exception as e:
    raise SystemExit(f"缺少 PyYAML: {e}")

meta_path = Path(sys.argv[1]).resolve()
port = str(sys.argv[2])
model_path = Path(sys.argv[3]).resolve()
data = yaml.safe_load(meta_path.read_text(encoding='utf-8')) or {}
model_meta = data.get('model') if isinstance(data.get('model'), dict) else {}

def first(*vals, default=''):
    for v in vals:
        if v is not None and str(v).strip() != '':
            return v
    return default

def norm_names(v):
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, dict):
        def key(k):
            s = str(k)
            return (0, int(s)) if s.isdigit() else (1, s)
        return [str(v[k]) for k in sorted(v.keys(), key=key)]
    return []

def norm_task(v):
    s = str(v or '').strip().lower()
    if s in {'detection', 'detect', 'yolo_detection', 'object_detection'}:
        return 'detection'
    if s in {'classification', 'classify', 'image_classification', 'cls'}:
        return 'classification'
    if s in {
        'obb', 'obb_detection', 'oriented_detection', 'oriented_bbox_detection',
        'rotated_detection', 'rotated_bbox_detection', 'yolo_obb', 'yolov8_obb',
    }:
        return 'obb_detection'
    if s in {'seg', 'segment', 'segmentation', 'instance_segmentation', 'yolo_seg', 'yolov8_seg', 'mask_segmentation'}:
        return 'segmentation'
    return ''

task = norm_task(first(data.get('task'), model_meta.get('task'), default='detection'))
if not task:
    raw = first(data.get('task'), model_meta.get('task'), default='')
    raise SystemExit(f"meta task 无效: {raw}")

input_size = first(
    data.get('input_size'),
    model_meta.get('input_size'),
    default=[224, 224] if task == 'classification' else [640, 640],
)
if isinstance(input_size, str):
    parts = input_size.replace(',', ' ').split()
else:
    parts = list(input_size)
if len(parts) != 2:
    parts = [224, 224] if task == 'classification' else [640, 640]
input_size = [int(parts[0]), int(parts[1])]

class_names = norm_names(first(data.get('class_names'), model_meta.get('class_names'), default=[]))
num_classes = int(first(data.get('num_classes'), model_meta.get('num_classes'), default=len(class_names) or 1))
if not class_names:
    class_names = [str(i) for i in range(num_classes)]

topk = int(first(data.get('topk'), model_meta.get('topk'), default=min(max(num_classes, 1), 5)))
conf_threshold = float(first(data.get('conf_threshold'), model_meta.get('conf_threshold'), default=0.25))
nms_threshold = float(first(data.get('nms_threshold'), model_meta.get('nms_threshold'), default=0.45))
mask_threshold = float(first(data.get('mask_threshold'), model_meta.get('mask_threshold'), default=0.5))

deploy_meta = data.get('deploy') if isinstance(data.get('deploy'), dict) else {}
dataset_meta = data.get('dataset') if isinstance(data.get('dataset'), dict) else {}
device_id = str(first(
    data.get('device_id'),
    deploy_meta.get('target_device'),
    dataset_meta.get('device_id'),
    default='rk3588-001',
))

out = {
    'device_id': device_id,
    'task': task,
    'input_size': input_size,
    'input_size_env': f"{input_size[0]},{input_size[1]}",
    'num_classes': num_classes,
    'topk': topk,
    'conf_threshold': conf_threshold,
    'nms_threshold': nms_threshold,
    'mask_threshold': mask_threshold,
    'model_path': str(model_path),
    'meta_path': str(meta_path),
    'port': port,
}
print(json.dumps(out, ensure_ascii=False))
PY
}

META_JSON="$(read_meta_json)"
DEVICE_ID="$(python3 - <<'PY' "${META_JSON}"
import json,sys; print(json.loads(sys.argv[1])['device_id'])
PY
)"
TASK="$(python3 - <<'PY' "${META_JSON}"
import json,sys; print(json.loads(sys.argv[1])['task'])
PY
)"
INPUT_SIZE="$(python3 - <<'PY' "${META_JSON}"
import json,sys; print(json.loads(sys.argv[1])['input_size_env'])
PY
)"
NUM_CLASSES="$(python3 - <<'PY' "${META_JSON}"
import json,sys; print(json.loads(sys.argv[1])['num_classes'])
PY
)"
TOPK="$(python3 - <<'PY' "${META_JSON}"
import json,sys; print(json.loads(sys.argv[1])['topk'])
PY
)"
CONF_THRESHOLD="$(python3 - <<'PY' "${META_JSON}"
import json,sys; print(json.loads(sys.argv[1])['conf_threshold'])
PY
)"
NMS_THRESHOLD="$(python3 - <<'PY' "${META_JSON}"
import json,sys; print(json.loads(sys.argv[1])['nms_threshold'])
PY
)"
MASK_THRESHOLD="$(python3 - <<'PY' "${META_JSON}"
import json,sys; print(json.loads(sys.argv[1])['mask_threshold'])
PY
)"

mkdir -p /tmp
cat > /tmp/visionops_switch.env <<EOF_ENV
# Auto generated by switch_model.sh
DEVICE_ID=${DEVICE_ID}
MODEL_PATH=${MODEL_PATH}
INFERENCE_URL=http://localhost:${PORT}
REPORT_INTERVAL=60
TASK=${TASK}
VISIONOPS_TASK=${TASK}
NPU_CORE=auto
NUM_CLASSES=${NUM_CLASSES}
INPUT_SIZE=${INPUT_SIZE}
VISIONOPS_INPUT_SIZE=${INPUT_SIZE}
CLASS_NAMES_FILE=${META_PATH}
PORT=${PORT}
METRICS_PORT=9091
CONF_THRESHOLD=${CONF_THRESHOLD}
NMS_THRESHOLD=${NMS_THRESHOLD}
MASK_THRESHOLD=${MASK_THRESHOLD}
TOPK=${TOPK}
WARMUP_RUNS=3
EOF_ENV

sudo -n mkdir -p "${RUNTIME_DIR}"
sudo -n install -m 644 /tmp/visionops_switch.env "${ENV_FILE}"
sudo -n install -m 644 /tmp/visionops_switch.env "${RUNTIME_ENV_FILE}"

if [[ ! -x "${STOP_SCRIPT}" ]]; then
  err "缺少停止脚本: ${STOP_SCRIPT}"
  exit 6
fi

log "switching to ${TASK} model: ${MODEL_PATH}"
bash "${STOP_SCRIPT}" "${PORT}" "${SERVICE_NAME}"

sudo -n systemctl daemon-reload
sudo -n systemctl restart "${SERVICE_NAME}"

log "waiting health: ${HEALTH_URL}"
for _ in $(seq 1 25); do
  if curl -sf "${HEALTH_URL}" >/tmp/visionops_health.json 2>/dev/null; then
    if python3 - <<'PY' /tmp/visionops_health.json "${MODEL_PATH}" "${TASK}"
import json, sys
from pathlib import Path
h = json.load(open(sys.argv[1], encoding='utf-8'))
target = str(Path(sys.argv[2]).resolve())
task = sys.argv[3]
model = str(Path(str(h.get('model_path',''))).resolve()) if h.get('model_path') else ''
if h.get('status') == 'ok' and str(h.get('task','')).lower() == task and model == target:
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
      cat /tmp/visionops_health.json
      echo
      log "switched successfully"
      exit 0
    fi
  fi
  sleep 1
done

err "health check failed or model_path/task mismatch"
cat /tmp/visionops_health.json 2>/dev/null || true
sudo -n journalctl -u "${SERVICE_NAME}" -n 120 --no-pager 2>/dev/null || true
exit 7