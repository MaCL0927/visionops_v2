#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# VisionOps Edge Deploy Script v6.3-task-aware-seg
# 零参数自动部署 / 版本化模型文件 / 同名 meta YAML
#
# 默认输入（v2 自动识别）：
#   detection:      models/export_detection/model.rknn
#   classification: models/export_classification/model.rknn
#   obb_detection:  models/export_obb/model.rknn
#   segmentation:   models/export_segmentation/model.rknn
#   edge/runtime/class_names.yaml
#   data/model_context/manifest.json（不存在时生成临时 manifest）
#
# 默认输出到边缘端：
#   /opt/visionops/models/{device_id}_{customer_id}_{cls|det|obb|seg}_{timestamp}.rknn
#   /opt/visionops/models/{device_id}_{customer_id}_{cls|det|obb|seg}_{timestamp}.yaml
#
# systemd 仍读取 /opt/visionops/.env，但 MODEL_PATH / CLASS_NAMES_FILE
# 指向具体版本化模型，不再使用 current.rknn / backup_*.rknn，
# 也不再上传 class_names.yaml 或 runtime/class_names.yaml。
# ============================================================

MODEL_PATH="auto"
CLASS_NAMES_FILE="auto"
DATASET_MANIFEST="auto"
CONFIG_FILE="auto"
LOCAL_EDGE_DIR="edge"
LOCAL_EDGE_ENV="edge/runtime/edge.env"

SYNC_EDGE_CODE="false"
CODE_ONLY="false"
NO_RESTART="false"
MODEL_PATH_EXPLICIT="false"
CLASS_NAMES_EXPLICIT="false"
DATASET_MANIFEST_EXPLICIT="false"
CONFIG_EXPLICIT="false"
AUTO_MANIFEST_FILE=""
MODEL_VERSION_OVERRIDE=""
DISPLAY_NAME=""
INPUT_SIZE_OVERRIDE=""
TOPK_OVERRIDE=""
CONF_THRESHOLD_OVERRIDE=""
NMS_THRESHOLD_OVERRIDE=""
ROI_CLASSIFICATION="false"

# 临时部署目标覆盖参数：用于从控制台把同一个模型部署到其他设备。
# 部署目录第一版固定为 /opt/visionops，不对页面开放，避免 systemd 模板和运行目录不一致。
TARGET_DEVICE_ID_OVERRIDE=""
TARGET_HOST_OVERRIDE=""
TARGET_USER_OVERRIDE="ubuntu"
TARGET_PORT_OVERRIDE="22"
TARGET_HEALTH_PORT_OVERRIDE=""
TARGET_HEALTH_URL_OVERRIDE=""

REMOTE_INSTALL_DIR="/opt/visionops"
REMOTE_MODEL_DIR="${REMOTE_INSTALL_DIR}/models"
REMOTE_EDGE_DIR="${REMOTE_INSTALL_DIR}/edge"
REMOTE_RUNTIME_DIR="${REMOTE_EDGE_DIR}/runtime"
REMOTE_ROOT_ENV="${REMOTE_INSTALL_DIR}/.env"
REMOTE_TMP_DIR="/tmp"
REMOTE_INFERENCE_SERVICE="/etc/systemd/system/visionops-inference.service"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

usage() {
  cat <<USAGE
用法：
  bash edge/deploy/push.sh [options]

默认读取：
  模型:       根据 task.yaml / edge/runtime/class_names.yaml 自动选择
              detection      -> models/export_detection/model.rknn
              classification -> models/export_classification/model.rknn
              obb_detection  -> models/export_obb/model.rknn
              segmentation   -> models/export_segmentation/model.rknn
  类别配置:   edge/runtime/class_names.yaml
  数据集信息: data/model_context/manifest.json；不存在时自动生成临时 manifest

常用：
  bash edge/deploy/push.sh
  bash edge/deploy/push.sh --code
  bash edge/deploy/push.sh --code-only

可选参数：
  --model <path>              覆盖默认模型路径
  --class-names <path>        覆盖默认类别配置路径
  --dataset-manifest <path>   覆盖默认数据集 manifest 路径
  --config <path>             覆盖设备配置，默认自动查找 deploy.yaml / task.yaml / legacy mlops
  --edge-env <path>           覆盖默认 edge.env，用于读取端口/阈值等可选默认值
  --model-version <name>      覆盖自动生成的模型版本名，不带后缀
  --display-name <name>       写入 meta 的展示名称
  --input-size "H,W"          覆盖 input_size；默认 classification=224,224，detection/obb_detection/segmentation=640,640
  --topk <N>                  覆盖分类 topk
  --conf-threshold <float>    覆盖检测/OBB/分割 置信度阈值
  --nms-threshold <float>     覆盖检测/OBB/分割 NMS 阈值
  --roi-classification        部署 ROI 分类双模型 bundle：
                              models/export_detection/model.rknn
                              models/export_classification/model.rknn
                              models/metrics_detection/eval_metrics.json
                              models/metrics_classification/eval_metrics.json
                              data/model_context/manifest.json
  --target-device-id <id>     临时指定部署目标设备 ID，例如 rk3588-002
  --target-host <host>        临时指定部署目标设备 IP/Host，例如 192.168.1.202
  --target-user <user>        临时指定 SSH 用户，默认 ubuntu
  --target-port <port>        临时指定 SSH 端口，默认 22
  --target-health-port <port> 临时指定健康检查端口，默认沿用 edge.env PORT
  --target-health-url <url>   临时指定完整健康检查地址，优先级高于 --target-health-port
  --code                      部署模型时同时同步 edge/ 代码到板端
  --code-only                 只同步 edge/ 代码，不上传模型、不改 env/service、不重启
  --no-restart                只上传文件和更新 env/service，不重启推理服务
  -h, --help                  显示帮助
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="${2:-}"; MODEL_PATH_EXPLICIT="true"; shift 2 ;;
    --class-names) CLASS_NAMES_FILE="${2:-}"; CLASS_NAMES_EXPLICIT="true"; shift 2 ;;
    --dataset-manifest) DATASET_MANIFEST="${2:-}"; DATASET_MANIFEST_EXPLICIT="true"; shift 2 ;;
    --config) CONFIG_FILE="${2:-}"; CONFIG_EXPLICIT="true"; shift 2 ;;
    --edge-env) LOCAL_EDGE_ENV="${2:-}"; shift 2 ;;
    --model-version) MODEL_VERSION_OVERRIDE="${2:-}"; shift 2 ;;
    --display-name) DISPLAY_NAME="${2:-}"; shift 2 ;;
    --input-size) INPUT_SIZE_OVERRIDE="${2:-}"; shift 2 ;;
    --topk) TOPK_OVERRIDE="${2:-}"; shift 2 ;;
    --conf-threshold) CONF_THRESHOLD_OVERRIDE="${2:-}"; shift 2 ;;
    --nms-threshold) NMS_THRESHOLD_OVERRIDE="${2:-}"; shift 2 ;;
    --roi-classification) ROI_CLASSIFICATION="true"; shift ;;
    --target-device-id) TARGET_DEVICE_ID_OVERRIDE="${2:-}"; shift 2 ;;
    --target-host) TARGET_HOST_OVERRIDE="${2:-}"; shift 2 ;;
    --target-user) TARGET_USER_OVERRIDE="${2:-ubuntu}"; shift 2 ;;
    --target-port) TARGET_PORT_OVERRIDE="${2:-22}"; shift 2 ;;
    --target-health-port) TARGET_HEALTH_PORT_OVERRIDE="${2:-}"; shift 2 ;;
    --target-health-url) TARGET_HEALTH_URL_OVERRIDE="${2:-}"; shift 2 ;;
    --code) SYNC_EDGE_CODE="true"; shift ;;
    --code-only) CODE_ONLY="true"; SYNC_EDGE_CODE="true"; NO_RESTART="true"; shift ;;
    --no-restart) NO_RESTART="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 不支持的参数: $1" >&2; usage; exit 1 ;;
  esac
done

log_info()  { echo "[INFO] $*"; }
log_ok()    { echo "[OK] $*"; }
log_warn()  { echo "[WARN] $*"; }
log_error() { echo "[ERROR] $*" >&2; }

require_file() { [[ -f "$1" ]] || { log_error "文件不存在: $1"; exit 1; }; }
require_dir()  { [[ -d "$1" ]] || { log_error "目录不存在: $1"; exit 1; }; }
require_cmd()  { command -v "$1" >/dev/null 2>&1 || { log_error "缺少命令: $1"; exit 1; }; }
get_file_md5() { md5sum "$1" | awk '{print $1}'; }
get_file_size_bytes() { stat -c%s "$1"; }

read_env_value() {
  local key="$1" file="$2"
  [[ -f "$file" ]] || return 0
  grep -E "^${key}=" "$file" | tail -n 1 | cut -d= -f2- || true
}

normalize_input_size_comma() {
  local value="$1"
  value="${value//,/ }"
  value="$(echo "$value" | xargs || true)"
  local h="" w="" extra=""
  read -r h w extra <<< "$value"
  if [[ -z "${h}" || -z "${w}" || -n "${extra}" ]]; then
    log_error "INPUT_SIZE 格式错误，应为 '640,640'、'640 640'、'224,224' 或 '224 224'，当前: ${value}"
    exit 1
  fi
  if ! [[ "$h" =~ ^[0-9]+$ && "$w" =~ ^[0-9]+$ ]]; then
    log_error "INPUT_SIZE 必须是整数，当前: ${h},${w}"
    exit 1
  fi
  echo "${h},${w}"
}

normalize_task() {
  local raw="$1"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$raw" in
    detection|detect|yolo_detection|object_detection) echo "detection" ;;
    classification|classify|image_classification|cls) echo "classification" ;;
    obb|obb_detection|oriented_detection|oriented_bbox_detection|rotated_detection|rotated_bbox_detection|yolo_obb|yolov8_obb) echo "obb_detection" ;;
    seg|segment|segmentation|instance_segmentation|yolo_seg|yolov8_seg|mask_segmentation) echo "segmentation" ;;
    *)
      log_error "edge/runtime/class_names.yaml 中 task 必须是 detection、classification、obb_detection 或 segmentation，当前: ${raw}"
      exit 1
      ;;
  esac
}

task_short() {
  local task="$1"
  if [[ "$task" == "classification" ]]; then
    echo "cls"
  elif [[ "$task" == "obb_detection" ]]; then
    echo "obb"
  elif [[ "$task" == "segmentation" ]]; then
    echo "seg"
  else
    echo "det"
  fi
}


normalize_task_family_value() {
  local raw="$1"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | xargs)"
  case "$raw" in
    detection|detect|yolo_detection|object_detection) echo "detection" ;;
    classification|classify|image_classification|cls) echo "classification" ;;
    obb|obb_detection|oriented_detection|oriented_bbox_detection|rotated_detection|rotated_bbox_detection|yolo_obb|yolov8_obb) echo "obb_detection" ;;
    seg|segment|segmentation|instance_segmentation|yolo_seg|yolov8_seg|mask_segmentation) echo "segmentation" ;;
    *) echo "" ;;
  esac
}

infer_task_family() {
  python3 - <<'PY' "$CLASS_NAMES_FILE" "pipeline/configs/task.yaml"
import sys
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None

def norm(v):
    s = str(v or "").strip().lower()
    if s in {"detection", "detect", "yolo_detection", "object_detection"}:
        return "detection"
    if s in {"classification", "classify", "image_classification", "cls"}:
        return "classification"
    if s in {"obb", "obb_detection", "oriented_detection", "oriented_bbox_detection", "rotated_detection", "rotated_bbox_detection", "yolo_obb", "yolov8_obb"}:
        return "obb_detection"
    if s in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg", "yolov8_seg", "mask_segmentation"}:
        return "segmentation"
    return ""

for raw in sys.argv[1:]:
    p = Path(raw)
    if not yaml or not p.exists():
        continue
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        continue
    for key in ["task_type", "task"]:
        got = norm(cfg.get(key))
        if got:
            print(got); raise SystemExit(0)
    task = cfg.get("task") if isinstance(cfg.get("task"), dict) else {}
    for key in ["type", "name"]:
        got = norm(task.get(key))
        if got:
            print(got); raise SystemExit(0)
print("detection")
PY
}

choose_first_existing() {
  local p
  for p in "$@"; do
    [[ -n "$p" && -f "$p" ]] && { echo "$p"; return 0; }
  done
  echo "$1"
}

config_has_device_info() {
  local p="$1"
  python3 - <<'PY' "$p"
import sys
from pathlib import Path
try:
    import yaml
except Exception:
    raise SystemExit(1)
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
try:
    cfg = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
except Exception:
    raise SystemExit(1)

def get_path(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict): return None
        cur = cur.get(key)
    return cur

for path in [
    ['edge_devices'], ['deploy','edge_devices'], ['deploy','devices'],
    ['edge','devices'], ['devices']
]:
    v = get_path(cfg, path)
    if isinstance(v, list) and v:
        raise SystemExit(0)

for path in [['edge','ssh_host'], ['edge','device_host'], ['deploy','host'], ['device','host']]:
    v = get_path(cfg, path)
    if v and str(v) not in {'0.0.0.0', 'localhost', '127.0.0.1'}:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

select_config_file() {
  local task="$1"
  local candidates=()
  candidates+=("pipeline/configs/deploy.yaml")
  candidates+=("pipeline/configs/generated/task.generated.yaml")
  candidates+=("pipeline/configs/task.yaml")

  local p
  for p in "${candidates[@]}"; do
    if [[ -f "$p" ]] && config_has_device_info "$p"; then
      echo "$p"; return 0
    fi
  done
  for p in "${candidates[@]}"; do
    [[ -f "$p" ]] && { echo "$p"; return 0; }
  done
  echo "pipeline/configs/task.yaml"
}

auto_prepare_inputs() {
  if [[ "$CLASS_NAMES_EXPLICIT" != "true" || "$CLASS_NAMES_FILE" == "auto" ]]; then
    if [[ -f "edge/runtime/class_names.yaml" ]]; then
      CLASS_NAMES_FILE="edge/runtime/class_names.yaml"
    else
      CLASS_NAMES_FILE="edge/runtime/class_names.yaml"
    fi
  fi

  local task_family
  task_family="$(infer_task_family)"

  if [[ "$MODEL_PATH_EXPLICIT" != "true" || "$MODEL_PATH" == "auto" ]]; then
    if [[ "$task_family" == "classification" ]]; then
      MODEL_PATH="$(choose_first_existing \
        "models/export_classification/model.rknn" \
        "models/export/model.rknn")"
    elif [[ "$task_family" == "obb_detection" ]]; then
      MODEL_PATH="$(choose_first_existing \
        "models/export_obb/model.rknn" \
        "models/export/model.rknn")"
    elif [[ "$task_family" == "segmentation" ]]; then
      MODEL_PATH="$(choose_first_existing \
        "models/export_segmentation/model.rknn" \
        "models/export/model.rknn")"
    else
      MODEL_PATH="$(choose_first_existing \
        "models/export_detection/model.rknn" \
        "models/export/model.rknn")"
    fi
  fi

  if [[ "$CONFIG_EXPLICIT" != "true" || "$CONFIG_FILE" == "auto" ]]; then
    CONFIG_FILE="$(select_config_file "$task_family")"
  fi

  if [[ "$DATASET_MANIFEST_EXPLICIT" != "true" || "$DATASET_MANIFEST" == "auto" ]]; then
    if [[ -f "data/model_context/manifest.json" ]]; then
      DATASET_MANIFEST="data/model_context/manifest.json"
    else
      AUTO_MANIFEST_FILE="$(mktemp /tmp/visionops_auto_manifest_XXXXXX.json)"
      python3 - <<'PY' "$AUTO_MANIFEST_FILE" "$CONFIG_FILE" "$task_family"
import json, os, sys
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None
out = Path(sys.argv[1]); cfg_path = Path(sys.argv[2]); task = sys.argv[3]

def load_yaml(p):
    if yaml and p.exists():
        try: return yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        except Exception: return {}
    return {}

cfg = load_yaml(cfg_path)

def first_device_id(cfg):
    for key in ['edge_devices', 'devices']:
        v = cfg.get(key)
        if isinstance(v, list) and v:
            return str(v[0].get('id') or v[0].get('device_id') or 'rk3588-001')
    for root in ['deploy', 'edge']:
        sub = cfg.get(root) if isinstance(cfg.get(root), dict) else {}
        for key in ['edge_devices', 'devices']:
            v = sub.get(key)
            if isinstance(v, list) and v:
                return str(v[0].get('id') or v[0].get('device_id') or 'rk3588-001')
        for key in ['device_id', 'id']:
            if sub.get(key):
                return str(sub.get(key))
    return os.environ.get('VISIONOPS_DEVICE_ID') or os.environ.get('DEVICE_ID') or 'rk3588-001'

manifest = {
    'dataset_id': f'auto_{task}',
    'device_id': first_device_id(cfg),
    'customer_id': os.environ.get('VISIONOPS_CUSTOMER_ID') or os.environ.get('CUSTOMER_ID') or 'CUST-000',
    'counts': {},
    'source': 'auto-generated by edge/deploy/push.sh because data/model_context/manifest.json was not found',
}
out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
PY
      DATASET_MANIFEST="$AUTO_MANIFEST_FILE"
      log_warn "data/model_context/manifest.json 不存在，已生成临时 manifest: ${DATASET_MANIFEST}"
    fi
  fi
}

sanitize_name_part() {
  echo "$1" | sed -E 's/[^A-Za-z0-9_-]+/-/g; s/^-+//; s/-+$//'
}

parse_deploy_context() {
  local class_file="$1" manifest_file="$2" input_override="$3" topk_override="$4" conf_override="$5" nms_override="$6" display_name="$7" model_path="$8" model_md5="$9" model_size_bytes="${10}" version_override="${11}" target_device_override="${12}"

  python3 - <<'PY' "$class_file" "$manifest_file" "$input_override" "$topk_override" "$conf_override" "$nms_override" "$display_name" "$model_path" "$model_md5" "$model_size_bytes" "$version_override" "$target_device_override"
import json, re, sys
from pathlib import Path
from datetime import datetime
try:
    import yaml
except Exception as e:
    print(json.dumps({"ok": False, "error": f"缺少 PyYAML: {e}"}, ensure_ascii=False))
    sys.exit(0)

class_file = Path(sys.argv[1])
manifest_file = Path(sys.argv[2])
input_override = sys.argv[3].strip()
topk_override = sys.argv[4].strip()
conf_override = sys.argv[5].strip()
nms_override = sys.argv[6].strip()
display_name = sys.argv[7].strip()
model_path = sys.argv[8]
model_md5 = sys.argv[9]
model_size_bytes = int(sys.argv[10])
version_override = sys.argv[11].strip()
target_device_override = sys.argv[12].strip()

def fail(msg):
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    sys.exit(0)

def load_yaml(p):
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        fail(f"读取类别配置失败: {p}, err={e}")

def load_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"读取数据集 manifest 失败: {p}, err={e}")

def normalize_names(v):
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, dict):
        def key_sort(k):
            try:
                return int(k)
            except Exception:
                return str(k)
        return [str(v[k]) for k in sorted(v.keys(), key=key_sort)]
    return []

def normalize_input(v, default):
    if isinstance(v, str):
        items = v.replace(',', ' ').split()
    elif isinstance(v, (list, tuple)):
        items = list(v)
    else:
        items = default
    if len(items) != 2:
        return default
    try:
        h, w = int(items[0]), int(items[1])
        if h <= 0 or w <= 0:
            raise ValueError
        return [h, w]
    except Exception:
        return default

def first_nonempty(*vals):
    for v in vals:
        if v is not None and str(v).strip() != "":
            return v
    return ""

def find_timestamp(manifest):
    # 优先从 dataset_id / package_id / name 中提取 YYYYMMDD_HHMMSS
    candidates = []
    for key in ["dataset_id", "package_id", "package_name", "name", "source_package", "source_dir"]:
        v = manifest.get(key)
        if v:
            candidates.append(str(v))
    for c in candidates:
        m = re.search(r"(20\d{6}_\d{6})", c)
        if m:
            return m.group(1)
    # 再尝试 created_at / timestamp
    for key in ["created_at", "collected_at", "timestamp", "exported_at"]:
        v = manifest.get(key)
        if not v:
            continue
        s = str(v)
        m = re.search(r"(20\d{2})[-/]?(\d{2})[-/]?(\d{2})[T _-]?(\d{2}):?(\d{2}):?(\d{2})", s)
        if m:
            return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}{m.group(5)}{m.group(6)}"
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def sanitize(s):
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", str(s)).strip("-")
    return s or "UNKNOWN"

class_cfg = load_yaml(class_file)
manifest = load_json(manifest_file)

def normalize_task_value(v):
    s = str(v or "").strip().lower()
    if s in {"detection", "detect", "yolo_detection", "object_detection"}:
        return "detection"
    if s in {"classification", "classify", "image_classification", "cls"}:
        return "classification"
    if s in {"obb", "obb_detection", "oriented_detection", "oriented_bbox_detection", "rotated_detection", "rotated_bbox_detection", "yolo_obb", "yolov8_obb"}:
        return "obb_detection"
    if s in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg", "yolov8_seg", "mask_segmentation"}:
        return "segmentation"
    return ""

task = normalize_task_value(class_cfg.get("task_type")) or normalize_task_value(class_cfg.get("task"))
if not task:
    fail(f"{class_file} 必须包含 task_type/task: classification、detection、obb_detection 或 segmentation")

class_names = normalize_names(class_cfg.get("class_names", class_cfg.get("names")))
if not class_names:
    fail(f"{class_file} 必须包含 class_names 或 names")

num_classes = class_cfg.get("num_classes")
try:
    num_classes = int(num_classes) if num_classes is not None else len(class_names)
except Exception:
    num_classes = len(class_names)
if num_classes != len(class_names):
    fail(f"num_classes 与 class_names 数量不一致: num_classes={num_classes}, len(class_names)={len(class_names)}")

default_input = [224, 224] if task == "classification" else [640, 640]
if input_override:
    input_size = normalize_input(input_override, default_input)
else:
    input_size = normalize_input(class_cfg.get("input_size"), default_input)

default_topk = min(5, num_classes) if task == "classification" else 5
default_topk = max(1, default_topk)
topk = int(topk_override or class_cfg.get("topk") or default_topk)
conf_threshold = float(conf_override or class_cfg.get("conf_threshold") or 0.25)
nms_threshold = float(nms_override or class_cfg.get("nms_threshold") or 0.45)

source_device_id = first_nonempty(manifest.get("device_id"), manifest.get("equipment_id"), manifest.get("edge_device_id"))
device_id = target_device_override or source_device_id
if not device_id:
    fail(f"{manifest_file} 必须包含 device_id，或通过 --target-device-id 指定目标设备 ID")
customer_id = first_nonempty(manifest.get("customer_id"), manifest.get("customer"), manifest.get("cust_id"), "CUST-000")
counts = manifest.get("counts") or {}
if not isinstance(counts, dict):
    counts = {}

timestamp = find_timestamp(manifest)
short = "cls" if task == "classification" else ("obb" if task == "obb_detection" else ("seg" if task == "segmentation" else "det"))
version_name = version_override or f"{sanitize(device_id)}_{sanitize(customer_id)}_{short}_{timestamp}"
version_name = sanitize(version_name)

meta = {
    "schema_version": 1,
    "task": task,
    "input_size": input_size,
    "num_classes": num_classes,
    "class_names": class_names,
    "topk": topk,
    "conf_threshold": conf_threshold,
    "nms_threshold": nms_threshold,
    "model": {
        "name": version_name,
        "file": f"{version_name}.rknn",
        "display_name": display_name or version_name,
        "md5": model_md5,
        "size_bytes": model_size_bytes,
        "source_path": model_path,
    },
    "dataset": {
        "manifest_path": str(manifest_file),
        "dataset_id": manifest.get("dataset_id") or manifest.get("package_id") or "",
        "device_id": str(device_id),
        "source_device_id": str(source_device_id or ""),
        "customer_id": str(customer_id),
        "counts": counts,
        "raw_manifest": manifest,
    },
    "deploy": {
        "deployed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deployed_by": "edge/deploy/push.sh",
        "target_device": str(device_id),
    },
}

print(json.dumps({
    "ok": True,
    "task": task,
    "task_short": short,
    "device_id": str(device_id),
    "customer_id": str(customer_id),
    "version_name": version_name,
    "version_model_file": f"{version_name}.rknn",
    "version_meta_file": f"{version_name}.yaml",
    "input_size": input_size,
    "input_size_env": f"{input_size[0]},{input_size[1]}",
    "num_classes": num_classes,
    "class_names": class_names,
    "topk": topk,
    "conf_threshold": conf_threshold,
    "nms_threshold": nms_threshold,
    "meta": meta,
}, ensure_ascii=False))
PY
}

write_meta_yaml() {
  local context_json="$1" out_file="$2"
  python3 - <<'PY' "$context_json" "$out_file"
import json, sys
from pathlib import Path
try:
    import yaml
except Exception as e:
    raise SystemExit(f"缺少 PyYAML: {e}")
ctx = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
meta = ctx["meta"]
out = Path(sys.argv[2])
out.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY
}

parse_device_from_yaml() {
  python3 - <<'PY' "$1" "$2"
import os, sys
from pathlib import Path
try:
    import yaml
except Exception:
    print("YAML_IMPORT_ERROR"); sys.exit(0)

cfg_path = Path(sys.argv[1]); device_id = sys.argv[2]
if not cfg_path.exists():
    print("CONFIG_NOT_FOUND"); sys.exit(0)

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def get_path(obj, path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur

def emit_device(d):
    host = d.get('host') or d.get('ssh_host') or d.get('device_host') or ''
    port = d.get('port') or d.get('ssh_port') or 22
    user = d.get('user') or d.get('ssh_user') or 'ubuntu'
    deploy_path = d.get('deploy_path') or d.get('model_dir') or '/opt/visionops/models/'
    service_name = d.get('service_name') or 'visionops-inference'
    health_url = d.get('health_url') or 'http://localhost:8080/health'
    print(f"{host}|{port}|{user}|{deploy_path}|{service_name}|{health_url}")
    sys.exit(0)

device_lists = []
for path in [
    ['edge_devices'], ['devices'],
    ['deploy', 'edge_devices'], ['deploy', 'devices'],
    ['edge', 'edge_devices'], ['edge', 'devices'],
]:
    v = get_path(cfg, path)
    if isinstance(v, list) and v:
        device_lists.append(v)

for devices in device_lists:
    for d in devices:
        if str(d.get('id') or d.get('device_id') or '') == str(device_id):
            emit_device(d)
    if len(devices) == 1:
        emit_device(devices[0])

for path in [['deploy'], ['device'], ['edge']]:
    d = get_path(cfg, path)
    if isinstance(d, dict):
        host = d.get('ssh_host') or d.get('device_host') or d.get('deploy_host')
        if host and str(host) not in {'0.0.0.0', 'localhost', '127.0.0.1'}:
            direct = {
                'host': host,
                'port': d.get('ssh_port') or d.get('port') or 22,
                'user': d.get('ssh_user') or d.get('user') or 'ubuntu',
                'deploy_path': d.get('deploy_path') or d.get('model_dir') or '/opt/visionops/models/',
                'service_name': d.get('service_name') or 'visionops-inference',
                'health_url': d.get('health_url') or 'http://localhost:8080/health',
            }
            emit_device(direct)

env_host = os.environ.get('VISIONOPS_EDGE_HOST') or os.environ.get('EDGE_DEVICE_HOST') or os.environ.get('EDGE_HOST')
if env_host:
    direct = {
        'host': env_host,
        'port': os.environ.get('VISIONOPS_EDGE_PORT') or os.environ.get('EDGE_SSH_PORT') or 22,
        'user': os.environ.get('VISIONOPS_EDGE_USER') or os.environ.get('EDGE_USER') or 'ubuntu',
        'deploy_path': os.environ.get('VISIONOPS_EDGE_DEPLOY_PATH') or '/opt/visionops/models/',
        'service_name': os.environ.get('VISIONOPS_EDGE_SERVICE') or 'visionops-inference',
        'health_url': os.environ.get('VISIONOPS_EDGE_HEALTH_URL') or 'http://localhost:8080/health',
    }
    emit_device(direct)

print("DEVICE_NOT_FOUND")
PY
}
remote_run() { local user="$1" host="$2" port="$3" cmd="$4"; ssh ${SSH_OPTS} -p "$port" "${user}@${host}" "$cmd"; }
remote_sudo_cmd() { local user="$1" host="$2" port="$3"; shift 3; ssh ${SSH_OPTS} -p "$port" "${user}@${host}" sudo -n "$@"; }
do_health_check() { remote_run "$1" "$2" "$3" "curl -sf '$4'"; }

resolve_health_url() {
  local configured_url="$1"
  local env_file="$2"
  local env_port
  env_port="$(read_env_value PORT "$env_file")"
  [[ -n "$env_port" ]] || env_port="8080"

  # deploy.yaml 里如果没有显式配置 health_url，parse_device_from_yaml 会给出默认 8080。
  # classification 常用 8082，这里根据当前 edge.env 的 PORT 自动修正默认健康检查端口。
  if [[ -z "$configured_url" \
        || "$configured_url" == "http://localhost:8080/health" \
        || "$configured_url" == "http://127.0.0.1:8080/health" ]]; then
    echo "http://localhost:${env_port}/health"
  else
    echo "$configured_url"
  fi
}

build_deploy_env() {
  local out_file="$1" target_device="$2" remote_model_path="$3" remote_meta_path="$4" task="$5" input_size_env="$6" num_classes="$7" topk="$8" conf_threshold="$9" nms_threshold="${10}"

  local env_npu_core env_port env_metrics_port env_warmup env_report_interval env_mask_threshold
  env_npu_core="$(read_env_value NPU_CORE "$LOCAL_EDGE_ENV")"; [[ -n "$env_npu_core" ]] || env_npu_core="auto"
  env_port="$(read_env_value PORT "$LOCAL_EDGE_ENV")"; [[ -n "$env_port" ]] || env_port="8080"
  env_metrics_port="$(read_env_value METRICS_PORT "$LOCAL_EDGE_ENV")"; [[ -n "$env_metrics_port" ]] || env_metrics_port="9091"
  env_warmup="$(read_env_value WARMUP_RUNS "$LOCAL_EDGE_ENV")"; [[ -n "$env_warmup" ]] || env_warmup="3"
  env_report_interval="$(read_env_value REPORT_INTERVAL "$LOCAL_EDGE_ENV")"; [[ -n "$env_report_interval" ]] || env_report_interval="60"
  env_mask_threshold="$(read_env_value MASK_THRESHOLD "$LOCAL_EDGE_ENV")"; [[ -n "$env_mask_threshold" ]] || env_mask_threshold="0.5"

  cat > "$out_file" <<EOF_ENV
# Auto generated by edge/deploy/push.sh v6.3-task-aware-seg
DEVICE_ID=${target_device}
MODEL_PATH=${remote_model_path}
INFERENCE_URL=http://localhost:${env_port}
REPORT_INTERVAL=${env_report_interval}
TASK=${task}
VISIONOPS_TASK=${task}
NPU_CORE=${env_npu_core}
NUM_CLASSES=${num_classes}
INPUT_SIZE=${input_size_env}
VISIONOPS_INPUT_SIZE=${input_size_env}
CLASS_NAMES_FILE=${remote_meta_path}
PORT=${env_port}
METRICS_PORT=${env_metrics_port}
CONF_THRESHOLD=${conf_threshold}
NMS_THRESHOLD=${nms_threshold}
MASK_THRESHOLD=${env_mask_threshold}
TOPK=${topk}
WARMUP_RUNS=${env_warmup}
EOF_ENV
}

build_inference_service_file() {
  local out_file="$1"
  cat > "$out_file" <<'EOF_SERVICE'
[Unit]
Description=VisionOps RK3588 Edge Inference Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/visionops
EnvironmentFile=/opt/visionops/.env

ExecStart=/opt/visionops/venv/bin/python /opt/visionops/edge/inference/engine.py \
    --model ${MODEL_PATH} \
    --task ${TASK} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --npu-core ${NPU_CORE} \
    --num-classes ${NUM_CLASSES} \
    --class-names-file ${CLASS_NAMES_FILE} \
    --metrics-port ${METRICS_PORT} \
    --conf-threshold ${CONF_THRESHOLD} \
    --nms-threshold ${NMS_THRESHOLD} \
    --mask-threshold ${MASK_THRESHOLD} \
    --topk ${TOPK} \
    --warmup-runs ${WARMUP_RUNS}

Restart=always
RestartSec=5
TimeoutStartSec=30
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-inference

[Install]
WantedBy=multi-user.target
EOF_SERVICE
}


# -----------------------------------------------------------------------------
# ROI classification 双模型 bundle 部署
# -----------------------------------------------------------------------------

build_roi_pipeline_env() {
  local out_file="$1" target_device="$2" pipeline_config="$3"

  local env_npu_core env_port env_metrics_port env_warmup env_report_interval
  env_npu_core="$(read_env_value NPU_CORE "$LOCAL_EDGE_ENV")"; [[ -n "$env_npu_core" ]] || env_npu_core="auto"
  env_port="$(read_env_value PORT "$LOCAL_EDGE_ENV")"; [[ -n "$env_port" ]] || env_port="8082"
  env_metrics_port="$(read_env_value METRICS_PORT "$LOCAL_EDGE_ENV")"; [[ -n "$env_metrics_port" ]] || env_metrics_port="9091"
  env_warmup="$(read_env_value WARMUP_RUNS "$LOCAL_EDGE_ENV")"; [[ -n "$env_warmup" ]] || env_warmup="3"
  env_report_interval="$(read_env_value REPORT_INTERVAL "$LOCAL_EDGE_ENV")"; [[ -n "$env_report_interval" ]] || env_report_interval="60"

  cat > "$out_file" <<EOF_ENV
# Auto generated by edge/deploy/push.sh roi_classification bundle mode
DEVICE_ID=${target_device}
TASK=roi_classification
VISIONOPS_TASK=roi_classification
PIPELINE_CONFIG=${pipeline_config}
INFERENCE_URL=http://localhost:${env_port}
REPORT_INTERVAL=${env_report_interval}
NPU_CORE=${env_npu_core}
PORT=${env_port}
METRICS_PORT=${env_metrics_port}
WARMUP_RUNS=${env_warmup}
EOF_ENV
}

build_roi_pipeline_service_file() {
  local out_file="$1"
  cat > "$out_file" <<'EOF_SERVICE'
[Unit]
Description=VisionOps RK3588 ROI Classification Pipeline Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/visionops
EnvironmentFile=/opt/visionops/.env

ExecStart=/opt/visionops/venv/bin/python /opt/visionops/edge/inference/pipeline_engine.py \
    --pipeline-config ${PIPELINE_CONFIG} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --metrics-port ${METRICS_PORT}

Restart=always
RestartSec=5
TimeoutStartSec=30
TimeoutStopSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=visionops-inference

[Install]
WantedBy=multi-user.target
EOF_SERVICE
}

build_roi_bundle_context() {
  local det_model="$1" cls_model="$2" det_metrics="$3" cls_metrics="$4" manifest_file="$5" version_override="$6" target_device_override="$7" conf_override="$8" nms_override="$9" topk_override="${10}"

  python3 - <<'PY' "$det_model" "$cls_model" "$det_metrics" "$cls_metrics" "$manifest_file" "$version_override" "$target_device_override" "$conf_override" "$nms_override" "$topk_override"
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

det_model = Path(sys.argv[1])
cls_model = Path(sys.argv[2])
det_metrics = Path(sys.argv[3])
cls_metrics = Path(sys.argv[4])
manifest_file = Path(sys.argv[5])
version_override = sys.argv[6].strip()
target_device_override = sys.argv[7].strip()
conf_override = sys.argv[8].strip()
nms_override = sys.argv[9].strip()
topk_override = sys.argv[10].strip()

def fail(msg):
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    raise SystemExit(0)

def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"读取 JSON 失败: {path}, err={exc}")

def md5(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def first_nonempty(*vals):
    for v in vals:
        if v is not None and str(v).strip():
            return v
    return ""

def sanitize(value):
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-")
    return value or "UNKNOWN"

def find_timestamp(manifest):
    for key in ["dataset_id", "package_id", "package_name", "batch_id", "name", "source_package", "source_dir"]:
        value = manifest.get(key)
        if value:
            m = re.search(r"(20\d{6}_\d{6})", str(value))
            if m:
                return m.group(1)
    for key in ["created_at", "collected_at", "timestamp", "exported_at"]:
        value = manifest.get(key)
        if not value:
            continue
        s = str(value)
        m = re.search(r"(20\d{2})[-/]?(\d{2})[-/]?(\d{2})[T _-]?(\d{2}):?(\d{2}):?(\d{2})", s)
        if m:
            return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}{m.group(5)}{m.group(6)}"
    return datetime.now().strftime("%Y%m%d_%H%M%S")

for path, label in [
    (det_model, "检测 RKNN"),
    (cls_model, "分类 RKNN"),
    (det_metrics, "检测 eval_metrics.json"),
    (cls_metrics, "分类 eval_metrics.json"),
    (manifest_file, "manifest.json"),
]:
    if not path.exists() or not path.is_file():
        fail(f"{label} 不存在: {path}")

det_eval = load_json(det_metrics)
cls_eval = load_json(cls_metrics)
manifest = load_json(manifest_file)

det_names = det_eval.get("class_names") or det_eval.get("names")
cls_names = cls_eval.get("class_names") or cls_eval.get("names")
if not isinstance(det_names, list) or not det_names:
    fail(f"{det_metrics} 中缺少有效 class_names")
if not isinstance(cls_names, list) or not cls_names:
    fail(f"{cls_metrics} 中缺少有效 class_names")

det_names = [str(x) for x in det_names]
cls_names = [str(x) for x in cls_names]

det_num = int(det_eval.get("num_classes") or len(det_names))
cls_num = int(cls_eval.get("num_classes") or len(cls_names))
if det_num != len(det_names):
    fail(f"检测 num_classes 与 class_names 数量不一致: {det_num} vs {len(det_names)}")
if cls_num != len(cls_names):
    fail(f"分类 num_classes 与 class_names 数量不一致: {cls_num} vs {len(cls_names)}")

source_device_id = first_nonempty(manifest.get("device_id"), manifest.get("equipment_id"), manifest.get("edge_device_id"))
device_id = target_device_override or source_device_id
if not device_id:
    fail(f"{manifest_file} 必须包含 device_id，或通过 --target-device-id 指定目标设备 ID")
customer_id = first_nonempty(manifest.get("customer_id"), manifest.get("customer"), manifest.get("cust_id"), "CUST-000")
timestamp = find_timestamp(manifest)

version_name = version_override or f"{sanitize(device_id)}_{sanitize(customer_id)}_roi_cls_{timestamp}"
version_name = sanitize(version_name)

det_conf = float(conf_override or det_eval.get("conf_threshold") or 0.25)
det_nms = float(nms_override or det_eval.get("nms_threshold") or 0.45)
cls_topk = int(topk_override or min(5, cls_num))
cls_topk = max(1, min(cls_topk, cls_num))
roi_padding = float(first_nonempty(os.environ.get("VISIONOPS_ROI_CLS_PADDING"), 0.05))

det_md5 = md5(det_model)
cls_md5 = md5(cls_model)

ctx = {
    "ok": True,
    "task": "roi_classification",
    "version_name": version_name,
    "device_id": str(device_id),
    "customer_id": str(customer_id),
    "detector": {
        "source_model": str(det_model),
        "source_metrics": str(det_metrics),
        "file": "detector.rknn",
        "meta_file": "detector.yaml",
        "md5": det_md5,
        "size_bytes": det_model.stat().st_size,
        "num_classes": det_num,
        "class_names": det_names,
        "input_size": [640, 640],
        "conf_threshold": det_conf,
        "nms_threshold": det_nms,
        "metrics": det_eval,
    },
    "classifier": {
        "source_model": str(cls_model),
        "source_metrics": str(cls_metrics),
        "file": "classifier.rknn",
        "meta_file": "classifier.yaml",
        "md5": cls_md5,
        "size_bytes": cls_model.stat().st_size,
        "num_classes": cls_num,
        "class_names": cls_names,
        "input_size": [224, 224],
        "topk": cls_topk,
        "metrics": cls_eval,
    },
    "roi": {
        "mode": "full_box",
        "padding_ratio": roi_padding,
    },
    "dataset": {
        "manifest_path": str(manifest_file),
        "device_id": str(device_id),
        "source_device_id": str(source_device_id or ""),
        "customer_id": str(customer_id),
        "dataset_id": manifest.get("dataset_id") or manifest.get("package_id") or manifest.get("batch_id") or "",
        "raw_manifest": manifest,
    },
    "deploy": {
        "deployed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deployed_by": "edge/deploy/push.sh --roi-classification",
        "target_device": str(device_id),
    },
}
print(json.dumps(ctx, ensure_ascii=False))
PY
}

write_roi_bundle_files() {
  local context_json="$1" bundle_dir="$2"
  python3 - <<'PY' "$context_json" "$bundle_dir"
import json
import shutil
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"缺少 PyYAML: {exc}")

ctx = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
bundle_dir = Path(sys.argv[2])
bundle_dir.mkdir(parents=True, exist_ok=True)

det = ctx["detector"]
cls = ctx["classifier"]

shutil.copy2(det["source_model"], bundle_dir / det["file"])
shutil.copy2(cls["source_model"], bundle_dir / cls["file"])

detector_meta = {
    "schema_version": 1,
    "task": "detection",
    "input_size": det["input_size"],
    "num_classes": det["num_classes"],
    "class_names": det["class_names"],
    "conf_threshold": det["conf_threshold"],
    "nms_threshold": det["nms_threshold"],
    "model": {
        "file": det["file"],
        "md5": det["md5"],
        "size_bytes": det["size_bytes"],
        "source_path": det["source_model"],
    },
    "metrics": det["metrics"],
}

classifier_meta = {
    "schema_version": 1,
    "task": "classification",
    "input_size": cls["input_size"],
    "num_classes": cls["num_classes"],
    "class_names": cls["class_names"],
    "topk": cls["topk"],
    "model": {
        "file": cls["file"],
        "md5": cls["md5"],
        "size_bytes": cls["size_bytes"],
        "source_path": cls["source_model"],
        "architecture": cls["metrics"].get("architecture", ""),
    },
    "metrics": cls["metrics"],
}

pipeline = {
    "schema_version": 1,
    "pipeline_type": "roi_classification",
    "task": "roi_classification",
    "pipeline_name": ctx["version_name"],
    "stage1": {
        "name": "detector",
        "task": "detection",
        "model_path": det["file"],
        "meta_path": det["meta_file"],
        "input_size": det["input_size"],
        "num_classes": det["num_classes"],
        "class_names": det["class_names"],
        "conf_threshold": det["conf_threshold"],
        "nms_threshold": det["nms_threshold"],
        "select_policy": "conf_area",
        "target_class_id": 0 if det["num_classes"] == 1 else None,
        "target_class_name": det["class_names"][0] if det["num_classes"] == 1 else "",
    },
    "roi": ctx["roi"],
    "stage2": {
        "name": "classifier",
        "task": "classification",
        "model_path": cls["file"],
        "meta_path": cls["meta_file"],
        "input_size": cls["input_size"],
        "num_classes": cls["num_classes"],
        "class_names": cls["class_names"],
        "topk": cls["topk"],
    },
    "decision": {
        "low_det_conf_policy": "REVIEW",
        "bad_roi_policy": "REVIEW",
        "low_cls_conf_policy": "REVIEW",
    },
    "dataset": ctx["dataset"],
    "deploy": ctx["deploy"],
}

(bundle_dir / det["meta_file"]).write_text(yaml.safe_dump(detector_meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
(bundle_dir / cls["meta_file"]).write_text(yaml.safe_dump(classifier_meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
(bundle_dir / "pipeline.yaml").write_text(yaml.safe_dump(pipeline, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY
}

deploy_roi_classification_bundle() {
  local det_model="models/export_detection/model.rknn"
  local cls_model="models/export_classification/model.rknn"
  local det_metrics="models/metrics_detection/eval_metrics.json"
  local cls_metrics="models/metrics_classification/eval_metrics.json"

  if [[ "$CONFIG_EXPLICIT" != "true" || "$CONFIG_FILE" == "auto" ]]; then
    CONFIG_FILE="$(select_config_file "detection")"
  fi

  if [[ "$DATASET_MANIFEST_EXPLICIT" != "true" || "$DATASET_MANIFEST" == "auto" ]]; then
    DATASET_MANIFEST="data/model_context/manifest.json"
  fi

  require_file "$det_model"
  require_file "$cls_model"
  require_file "$det_metrics"
  require_file "$cls_metrics"
  require_file "$DATASET_MANIFEST"
  require_file "$CONFIG_FILE"
  [[ -f "$LOCAL_EDGE_ENV" ]] || log_warn "edge.env 不存在，将使用默认端口和默认推理参数: ${LOCAL_EDGE_ENV}"
  [[ "$SYNC_EDGE_CODE" == "true" ]] && require_dir "$LOCAL_EDGE_DIR"

  local context_file context_json
  context_file="$(mktemp /tmp/visionops_roi_context_XXXXXX.json)"
  context_json="$(build_roi_bundle_context "$det_model" "$cls_model" "$det_metrics" "$cls_metrics" "$DATASET_MANIFEST" "$MODEL_VERSION_OVERRIDE" "$TARGET_DEVICE_ID_OVERRIDE" "$CONF_THRESHOLD_OVERRIDE" "$NMS_THRESHOLD_OVERRIDE" "$TOPK_OVERRIDE")"
  echo "$context_json" > "$context_file"

  local ok err
  ok="$(python3 - <<'PY' "$context_file"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("ok"))
PY
)"
  if [[ "$ok" != "True" && "$ok" != "true" ]]; then
    err="$(python3 - <<'PY' "$context_file"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("error", "未知错误"))
PY
)"
    log_error "$err"
    rm -f "$context_file"
    exit 1
  fi

  local version_name device_id customer_id
  version_name="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["version_name"])
PY
)"
  device_id="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["device_id"])
PY
)"
  customer_id="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["customer_id"])
PY
)"

  local local_bundle_dir
  local_bundle_dir="$(mktemp -d /tmp/visionops_roi_bundle_XXXXXX)"
  write_roi_bundle_files "$context_file" "$local_bundle_dir"

  log_info "准备部署 ROI 分类双模型 bundle"
  log_info "检测模型: ${det_model}"
  log_info "分类模型: ${cls_model}"
  log_info "检测指标: ${det_metrics}"
  log_info "分类指标: ${cls_metrics}"
  log_info "数据集信息: ${DATASET_MANIFEST}"
  log_info "目标设备 ID: ${device_id}"
  log_info "客户 ID: ${customer_id}"
  log_info "bundle 名称: ${version_name}"

  local parsed host port user deploy_path service_name health_url
  if [[ -n "$TARGET_HOST_OVERRIDE" ]]; then
    log_warn "使用临时目标设备参数部署到其他设备: device_id=${device_id}, host=${TARGET_HOST_OVERRIDE}"
    host="$TARGET_HOST_OVERRIDE"
    port="${TARGET_PORT_OVERRIDE:-22}"
    user="${TARGET_USER_OVERRIDE:-ubuntu}"
    deploy_path="${REMOTE_MODEL_DIR}"
    service_name="visionops-inference"
    if [[ -n "$TARGET_HEALTH_URL_OVERRIDE" ]]; then
      health_url="$TARGET_HEALTH_URL_OVERRIDE"
    elif [[ -n "$TARGET_HEALTH_PORT_OVERRIDE" ]]; then
      health_url="http://localhost:${TARGET_HEALTH_PORT_OVERRIDE}/health"
    else
      health_url=""
    fi
  else
    parsed="$(parse_device_from_yaml "$CONFIG_FILE" "$device_id")"
    case "$parsed" in
      YAML_IMPORT_ERROR) log_error "缺少 PyYAML，无法解析设备配置: ${CONFIG_FILE}"; rm -f "$context_file"; rm -rf "$local_bundle_dir"; exit 1 ;;
      CONFIG_NOT_FOUND) log_error "设备配置不存在: ${CONFIG_FILE}"; rm -f "$context_file"; rm -rf "$local_bundle_dir"; exit 1 ;;
      DEVICE_NOT_FOUND) log_error "在 ${CONFIG_FILE} 中找不到设备: ${device_id}。请在 pipeline/configs/task.yaml 或 pipeline/configs/deploy.yaml 中增加 edge_devices，或设置 EDGE_DEVICE_HOST/EDGE_USER 环境变量。"; rm -f "$context_file"; rm -rf "$local_bundle_dir"; exit 1 ;;
    esac
    IFS='|' read -r host port user deploy_path service_name health_url <<< "$parsed"
  fi

  [[ -n "$host" ]] || { log_error "设备 host 为空"; rm -f "$context_file"; rm -rf "$local_bundle_dir"; exit 1; }
  if ! [[ "$port" =~ ^[0-9]+$ ]]; then
    log_error "SSH 端口必须是数字，当前: ${port}"
    rm -f "$context_file"; rm -rf "$local_bundle_dir"
    exit 1
  fi

  local resolved_health_url
  resolved_health_url="$(resolve_health_url "$health_url" "$LOCAL_EDGE_ENV")"
  if [[ "$resolved_health_url" != "$health_url" ]]; then
    log_warn "健康检查地址根据 edge.env PORT 自动调整: ${health_url} -> ${resolved_health_url}"
    health_url="$resolved_health_url"
  fi

  REMOTE_MODEL_DIR="${deploy_path%/}"
  REMOTE_RUNTIME_DIR="${REMOTE_EDGE_DIR}/runtime"

  local remote_bundle_dir remote_pipeline_config
  remote_bundle_dir="${REMOTE_MODEL_DIR}/${version_name}"
  remote_pipeline_config="${remote_bundle_dir}/pipeline.yaml"

  local local_deploy_env local_service_file ts remote_tmp_edge_env remote_tmp_service
  local remote_tmp_detector remote_tmp_detector_meta remote_tmp_classifier remote_tmp_classifier_meta remote_tmp_pipeline
  local_deploy_env="$(mktemp /tmp/visionops_roi_env_XXXXXX.env)"
  local_service_file="$(mktemp /tmp/visionops_roi_service_XXXXXX.service)"
  ts="$(date +%Y%m%d_%H%M%S)"
  remote_tmp_edge_env="${REMOTE_TMP_DIR}/visionops_roi_${ts}.env"
  remote_tmp_service="${REMOTE_TMP_DIR}/visionops-roi-inference_${ts}.service"
  remote_tmp_detector="${REMOTE_TMP_DIR}/visionops_${version_name}_detector_${ts}.rknn"
  remote_tmp_detector_meta="${REMOTE_TMP_DIR}/visionops_${version_name}_detector_${ts}.yaml"
  remote_tmp_classifier="${REMOTE_TMP_DIR}/visionops_${version_name}_classifier_${ts}.rknn"
  remote_tmp_classifier_meta="${REMOTE_TMP_DIR}/visionops_${version_name}_classifier_${ts}.yaml"
  remote_tmp_pipeline="${REMOTE_TMP_DIR}/visionops_${version_name}_pipeline_${ts}.yaml"

  build_roi_pipeline_env "$local_deploy_env" "$device_id" "$remote_pipeline_config"
  build_roi_pipeline_service_file "$local_service_file"

  log_info "本地 ROI bundle 内容:"
  ls -lh "$local_bundle_dir" | sed 's/^/[INFO]   /'

  log_info "目标设备: ${user}@${host}:${port}"
  remote_run "$user" "$host" "$port" "echo connected >/dev/null" || { log_error "无法连接设备"; rm -f "$context_file" "$local_deploy_env" "$local_service_file"; rm -rf "$local_bundle_dir"; exit 1; }
  log_ok "设备连通性验证通过"

  remote_sudo_cmd "$user" "$host" "$port" mkdir -p "${REMOTE_MODEL_DIR}" "${REMOTE_RUNTIME_DIR}"

  log_info "上传 ROI 分类 bundle 到远端临时目录 /tmp"
  scp ${SSH_OPTS} -P "$port" "$local_bundle_dir/detector.rknn" "${user}@${host}:${remote_tmp_detector}"
  scp ${SSH_OPTS} -P "$port" "$local_bundle_dir/detector.yaml" "${user}@${host}:${remote_tmp_detector_meta}"
  scp ${SSH_OPTS} -P "$port" "$local_bundle_dir/classifier.rknn" "${user}@${host}:${remote_tmp_classifier}"
  scp ${SSH_OPTS} -P "$port" "$local_bundle_dir/classifier.yaml" "${user}@${host}:${remote_tmp_classifier_meta}"
  scp ${SSH_OPTS} -P "$port" "$local_bundle_dir/pipeline.yaml" "${user}@${host}:${remote_tmp_pipeline}"
  scp ${SSH_OPTS} -P "$port" "$local_deploy_env" "${user}@${host}:${remote_tmp_edge_env}"
  scp ${SSH_OPTS} -P "$port" "$local_service_file" "${user}@${host}:${remote_tmp_service}"

  log_info "使用 sudo 安装 ROI 分类 bundle 到 ${remote_bundle_dir}"
  remote_run "$user" "$host" "$port" "sudo -n rm -rf '${remote_bundle_dir}' && sudo -n mkdir -p '${remote_bundle_dir}' && sudo -n install -m 644 '${remote_tmp_detector}' '${remote_bundle_dir}/detector.rknn' && sudo -n install -m 644 '${remote_tmp_detector_meta}' '${remote_bundle_dir}/detector.yaml' && sudo -n install -m 644 '${remote_tmp_classifier}' '${remote_bundle_dir}/classifier.rknn' && sudo -n install -m 644 '${remote_tmp_classifier_meta}' '${remote_bundle_dir}/classifier.yaml' && sudo -n install -m 644 '${remote_tmp_pipeline}' '${remote_bundle_dir}/pipeline.yaml' && sudo -n chown -R root:root '${remote_bundle_dir}' && rm -f '${remote_tmp_detector}' '${remote_tmp_detector_meta}' '${remote_tmp_classifier}' '${remote_tmp_classifier_meta}' '${remote_tmp_pipeline}'"

  log_info "远端 ROI bundle 内容:"
  remote_run "$user" "$host" "$port" "sudo -n ls -lh '${remote_bundle_dir}'"

  if ! remote_run "$user" "$host" "$port" "sudo -n test -f '${remote_bundle_dir}/detector.rknn' && sudo -n test -f '${remote_bundle_dir}/classifier.rknn' && sudo -n test -f '${remote_bundle_dir}/pipeline.yaml'"; then
    log_error "ROI bundle 安装校验失败：远端目录缺少 detector.rknn / classifier.rknn / pipeline.yaml"
    rm -f "$context_file" "$local_deploy_env" "$local_service_file"
    rm -rf "$local_bundle_dir"
    exit 1
  fi

  if [[ "$SYNC_EDGE_CODE" == "true" ]]; then
    log_info "同步整个 edge/ 目录..."

    # 推理服务通常以 root 运行，Python 会在 /opt/visionops/edge 下生成 root-owned __pycache__/*.pyc。
    # 如果直接 rsync --delete，ubuntu 用户无法删除这些缓存文件，会导致 --code 部署失败。
    # 这里先用 sudo 清理 pycache，再 rsync 时排除缓存目录，避免部署被运行时缓存文件阻塞。
    remote_run "$user" "$host" "$port" "sudo -n find '${REMOTE_EDGE_DIR}' -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true"
    remote_run "$user" "$host" "$port" "sudo -n find '${REMOTE_EDGE_DIR}' -type f -name '*.pyc' -delete 2>/dev/null || true"

    rsync -az --delete \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      -e "ssh ${SSH_OPTS} -p ${port}" \
      "${LOCAL_EDGE_DIR}/" "${user}@${host}:${REMOTE_EDGE_DIR}/"

    log_ok "edge/ 代码同步完成"
  fi

  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_edge_env}" "${REMOTE_ROOT_ENV}"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_edge_env}" "${REMOTE_RUNTIME_DIR}/edge.env"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_service}" "${REMOTE_INFERENCE_SERVICE}"
  remote_run "$user" "$host" "$port" "rm -f '${remote_tmp_edge_env}' '${remote_tmp_service}'" || true

  log_ok "已部署 ROI 分类 bundle: ${remote_bundle_dir}"
  log_ok "已更新 /opt/visionops/.env，TASK=roi_classification，PIPELINE_CONFIG=${remote_pipeline_config}"

  if [[ "$NO_RESTART" == "true" ]]; then
    log_warn "--no-restart 已启用，跳过 systemd 重启与健康检查"
    rm -f "$context_file" "$local_deploy_env" "$local_service_file"
    rm -rf "$local_bundle_dir"
    exit 0
  fi

  log_info "重启 ${service_name}。注意：边缘端需要存在 edge/inference/pipeline_engine.py"
  remote_sudo_cmd "$user" "$host" "$port" systemctl daemon-reload
  remote_sudo_cmd "$user" "$host" "$port" systemctl enable "${service_name}" >/dev/null || true
  remote_sudo_cmd "$user" "$host" "$port" systemctl restart "${service_name}"

  log_info "等待健康检查通过: ${health_url}"
  local health_ok="false"
  for i in $(seq 1 10); do
    if do_health_check "$user" "$host" "$port" "$health_url" >/dev/null; then
      health_ok="true"
      break
    fi
    sleep 1
  done

  if [[ "$health_ok" != "true" ]]; then
    log_error "服务健康检查失败，最近日志如下："
    remote_run "$user" "$host" "$port" "sudo -n journalctl -u ${service_name} -n 120 --no-pager || journalctl -u ${service_name} -n 120 --no-pager || true" || true
    rm -f "$context_file" "$local_deploy_env" "$local_service_file"
    rm -rf "$local_bundle_dir"
    exit 1
  fi

  log_info "当前健康检查结果："
  remote_run "$user" "$host" "$port" "curl -sf '${health_url}'" || true

  rm -f "$context_file" "$local_deploy_env" "$local_service_file"
  rm -rf "$local_bundle_dir"

  log_ok "ROI 分类双模型部署完成"
  log_ok "bundle=${remote_bundle_dir}"
  log_ok "pipeline=${remote_pipeline_config}"
}

main() {
  require_cmd ssh; require_cmd scp; require_cmd rsync; require_cmd md5sum; require_cmd python3

  if [[ "$ROI_CLASSIFICATION" == "true" ]]; then
    deploy_roi_classification_bundle
    exit 0
  fi

  auto_prepare_inputs
  [[ -n "$AUTO_MANIFEST_FILE" ]] && trap 'rm -f "$AUTO_MANIFEST_FILE"' EXIT

  if [[ "$CODE_ONLY" == "true" ]]; then
    require_dir "$LOCAL_EDGE_DIR"
    require_file "$CONFIG_FILE"
    local code_device_id code_parsed code_host code_port code_user code_deploy_path code_service code_health
    code_device_id="${VISIONOPS_DEVICE_ID:-${DEVICE_ID:-rk3588-001}}"
    code_parsed="$(parse_device_from_yaml "$CONFIG_FILE" "$code_device_id")"
    case "$code_parsed" in
      YAML_IMPORT_ERROR) log_error "缺少 PyYAML，无法解析设备配置: ${CONFIG_FILE}"; exit 1 ;;
      CONFIG_NOT_FOUND) log_error "设备配置不存在: ${CONFIG_FILE}"; exit 1 ;;
      DEVICE_NOT_FOUND) log_error "在 ${CONFIG_FILE} 中找不到设备: ${code_device_id}。可在 task.yaml/deploy.yaml 增加 edge_devices，或设置 EDGE_DEVICE_HOST/EDGE_USER。"; exit 1 ;;
    esac
    IFS='|' read -r code_host code_port code_user code_deploy_path code_service code_health <<< "$code_parsed"
    log_info "仅同步 edge/ 代码到 ${code_user}@${code_host}:${code_port}${REMOTE_EDGE_DIR}"
    remote_run "$code_user" "$code_host" "$code_port" "mkdir -p '${REMOTE_EDGE_DIR}'" || true
    rsync -az --delete -e "ssh ${SSH_OPTS} -p ${code_port}" "${LOCAL_EDGE_DIR}/" "${code_user}@${code_host}:${REMOTE_EDGE_DIR}/"
    log_ok "edge/ 代码同步完成"
    exit 0
  fi

  require_file "$MODEL_PATH"
  require_file "$CLASS_NAMES_FILE"
  require_file "$DATASET_MANIFEST"
  require_file "$CONFIG_FILE"
  [[ -f "$LOCAL_EDGE_ENV" ]] || log_warn "edge.env 不存在，将使用默认端口和默认推理参数: ${LOCAL_EDGE_ENV}"
  [[ "$SYNC_EDGE_CODE" == "true" ]] && require_dir "$LOCAL_EDGE_DIR"

  local model_md5 model_size_bytes context_file context_json
  model_md5="$(get_file_md5 "$MODEL_PATH")"
  model_size_bytes="$(get_file_size_bytes "$MODEL_PATH")"
  context_file="$(mktemp /tmp/visionops_deploy_context_XXXXXX.json)"

  context_json="$(parse_deploy_context "$CLASS_NAMES_FILE" "$DATASET_MANIFEST" "$INPUT_SIZE_OVERRIDE" "$TOPK_OVERRIDE" "$CONF_THRESHOLD_OVERRIDE" "$NMS_THRESHOLD_OVERRIDE" "$DISPLAY_NAME" "$MODEL_PATH" "$model_md5" "$model_size_bytes" "$MODEL_VERSION_OVERRIDE" "$TARGET_DEVICE_ID_OVERRIDE")"
  echo "$context_json" > "$context_file"

  local ok err
  ok="$(python3 - <<'PY' "$context_file"
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8')).get('ok'))
PY
)"
  if [[ "$ok" != "True" && "$ok" != "true" ]]; then
    err="$(python3 - <<'PY' "$context_file"
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8')).get('error', '未知错误'))
PY
)"
    log_error "$err"
    rm -f "$context_file"
    exit 1
  fi

  local task device_id customer_id version_name version_model_file version_meta_file input_size_env num_classes topk conf_threshold nms_threshold
  task="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['task'])
PY
)"
  device_id="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['device_id'])
PY
)"
  customer_id="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['customer_id'])
PY
)"
  version_name="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['version_name'])
PY
)"
  version_model_file="${version_name}.rknn"
  version_meta_file="${version_name}.yaml"
  input_size_env="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['input_size_env'])
PY
)"
  num_classes="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['num_classes'])
PY
)"
  topk="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['topk'])
PY
)"
  conf_threshold="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['conf_threshold'])
PY
)"
  nms_threshold="$(python3 - <<'PY' "$context_file"
import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['nms_threshold'])
PY
)"

  log_info "准备自动部署 VisionOps 模型"
  log_info "模型源文件: ${MODEL_PATH}"
  log_info "类别配置: ${CLASS_NAMES_FILE}"
  log_info "数据集信息: ${DATASET_MANIFEST}"
  log_info "目标设备 ID: ${device_id}"
  log_info "客户 ID: ${customer_id}"
  log_info "任务类型: ${task}"
  log_info "模型版本名: ${version_name}"
  log_info "输入尺寸: ${input_size_env}, 类别数: ${num_classes}"
  log_info "模型 MD5: ${model_md5}, size=${model_size_bytes} bytes"

  local parsed host port user deploy_path service_name health_url
  if [[ -n "$TARGET_HOST_OVERRIDE" ]]; then
    log_warn "使用临时目标设备参数部署到其他设备: device_id=${device_id}, host=${TARGET_HOST_OVERRIDE}"
    host="$TARGET_HOST_OVERRIDE"
    port="${TARGET_PORT_OVERRIDE:-22}"
    user="${TARGET_USER_OVERRIDE:-ubuntu}"
    deploy_path="${REMOTE_MODEL_DIR}"
    service_name="visionops-inference"
    if [[ -n "$TARGET_HEALTH_URL_OVERRIDE" ]]; then
      health_url="$TARGET_HEALTH_URL_OVERRIDE"
    elif [[ -n "$TARGET_HEALTH_PORT_OVERRIDE" ]]; then
      health_url="http://localhost:${TARGET_HEALTH_PORT_OVERRIDE}/health"
    else
      health_url=""
    fi
  else
    parsed="$(parse_device_from_yaml "$CONFIG_FILE" "$device_id")"
    case "$parsed" in
      YAML_IMPORT_ERROR) log_error "缺少 PyYAML，无法解析设备配置: ${CONFIG_FILE}"; rm -f "$context_file"; exit 1 ;;
      CONFIG_NOT_FOUND) log_error "设备配置不存在: ${CONFIG_FILE}"; rm -f "$context_file"; exit 1 ;;
      DEVICE_NOT_FOUND) log_error "在 ${CONFIG_FILE} 中找不到设备: ${device_id}。请在 pipeline/configs/task.yaml 或 pipeline/configs/deploy.yaml 中增加 edge_devices，或设置 EDGE_DEVICE_HOST/EDGE_USER 环境变量。"; rm -f "$context_file"; exit 1 ;;
    esac
    IFS='|' read -r host port user deploy_path service_name health_url <<< "$parsed"
  fi

  [[ -n "$host" ]] || { log_error "设备 host 为空"; rm -f "$context_file"; exit 1; }
  if ! [[ "$port" =~ ^[0-9]+$ ]]; then
    log_error "SSH 端口必须是数字，当前: ${port}"
    rm -f "$context_file"
    exit 1
  fi

  local resolved_health_url
  resolved_health_url="$(resolve_health_url "$health_url" "$LOCAL_EDGE_ENV")"
  if [[ "$resolved_health_url" != "$health_url" ]]; then
    log_warn "健康检查地址根据 edge.env PORT 自动调整: ${health_url} -> ${resolved_health_url}"
    health_url="$resolved_health_url"
  fi

  REMOTE_MODEL_DIR="${deploy_path%/}"
  REMOTE_RUNTIME_DIR="${REMOTE_EDGE_DIR}/runtime"
  local remote_version_model remote_version_meta
  remote_version_model="${REMOTE_MODEL_DIR}/${version_model_file}"
  remote_version_meta="${REMOTE_MODEL_DIR}/${version_meta_file}"

  local ts remote_tmp_model remote_tmp_meta remote_tmp_edge_env remote_tmp_service remote_tmp_stop remote_tmp_switch local_meta_file local_deploy_env local_service_file
  ts="$(date +%Y%m%d_%H%M%S)"
  remote_tmp_model="${REMOTE_TMP_DIR}/visionops_${version_model_file}.${ts}.tmp"
  remote_tmp_meta="${REMOTE_TMP_DIR}/visionops_${version_meta_file}.${ts}.tmp"
  remote_tmp_edge_env="${REMOTE_TMP_DIR}/visionops_edge_${task}_${ts}.env"
  remote_tmp_service="${REMOTE_TMP_DIR}/visionops-inference_${ts}.service"
  remote_tmp_stop="${REMOTE_TMP_DIR}/visionops_stop_inference_${ts}.sh"
  remote_tmp_switch="${REMOTE_TMP_DIR}/visionops_switch_model_${ts}.sh"
  local_meta_file="$(mktemp /tmp/visionops_meta_XXXXXX.yaml)"
  local_deploy_env="$(mktemp /tmp/visionops_edge_env_${task}_XXXXXX.env)"
  local_service_file="$(mktemp /tmp/visionops_inference_XXXXXX.service)"

  write_meta_yaml "$context_file" "$local_meta_file"
  build_deploy_env "$local_deploy_env" "$device_id" "$remote_version_model" "$remote_version_meta" "$task" "$input_size_env" "$num_classes" "$topk" "$conf_threshold" "$nms_threshold"
  build_inference_service_file "$local_service_file"

  log_info "目标设备: ${user}@${host}:${port}"
  remote_run "$user" "$host" "$port" "echo connected >/dev/null" || { log_error "无法连接设备"; rm -f "$context_file" "$local_meta_file" "$local_deploy_env" "$local_service_file"; exit 1; }
  log_ok "设备连通性验证通过"

  remote_sudo_cmd "$user" "$host" "$port" mkdir -p "${REMOTE_MODEL_DIR}" "${REMOTE_RUNTIME_DIR}"

  local backup_env backup_service
  backup_env="${REMOTE_TMP_DIR}/visionops_env_backup_${ts}.env"
  backup_service="${REMOTE_TMP_DIR}/visionops_service_backup_${ts}.service"
  remote_run "$user" "$host" "$port" "if [ -f '${REMOTE_ROOT_ENV}' ]; then sudo -n cp '${REMOTE_ROOT_ENV}' '${backup_env}'; fi; if [ -f '${REMOTE_INFERENCE_SERVICE}' ]; then sudo -n cp '${REMOTE_INFERENCE_SERVICE}' '${backup_service}'; fi" || true

  log_info "上传版本化模型、同名 meta、env、service..."
  scp ${SSH_OPTS} -P "$port" "$MODEL_PATH" "${user}@${host}:${remote_tmp_model}"
  scp ${SSH_OPTS} -P "$port" "$local_meta_file" "${user}@${host}:${remote_tmp_meta}"
  scp ${SSH_OPTS} -P "$port" "$local_deploy_env" "${user}@${host}:${remote_tmp_edge_env}"
  scp ${SSH_OPTS} -P "$port" "$local_service_file" "${user}@${host}:${remote_tmp_service}"
  scp ${SSH_OPTS} -P "$port" "${LOCAL_EDGE_DIR}/deploy/stop_inference.sh" "${user}@${host}:${remote_tmp_stop}"
  scp ${SSH_OPTS} -P "$port" "${LOCAL_EDGE_DIR}/deploy/switch_model.sh" "${user}@${host}:${remote_tmp_switch}"

  local remote_md5
  remote_md5="$(remote_run "$user" "$host" "$port" "md5sum '${remote_tmp_model}' | awk '{print \$1}'")"
  if [[ "$remote_md5" != "$model_md5" ]]; then
    log_error "模型 MD5 校验失败: local=${model_md5}, remote=${remote_md5}"
    rm -f "$context_file" "$local_meta_file" "$local_deploy_env" "$local_service_file"
    exit 1
  fi
  log_ok "模型 MD5 校验通过"

  if [[ "$SYNC_EDGE_CODE" == "true" ]]; then
    log_info "同步整个 edge/ 目录..."

    # 推理服务通常以 root 运行，Python 会在 /opt/visionops/edge 下生成 root-owned __pycache__/*.pyc。
    # 如果直接 rsync --delete，ubuntu 用户无法删除这些缓存文件，会导致 --code 部署失败。
    # 这里先用 sudo 清理 pycache，再 rsync 时排除缓存目录，避免部署被运行时缓存文件阻塞。
    remote_run "$user" "$host" "$port" "sudo -n find '${REMOTE_EDGE_DIR}' -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true"
    remote_run "$user" "$host" "$port" "sudo -n find '${REMOTE_EDGE_DIR}' -type f -name '*.pyc' -delete 2>/dev/null || true"

    rsync -az --delete \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      -e "ssh ${SSH_OPTS} -p ${port}" \
      "${LOCAL_EDGE_DIR}/" "${user}@${host}:${REMOTE_EDGE_DIR}/"

    log_ok "edge/ 代码同步完成"
  fi

  remote_sudo_cmd "$user" "$host" "$port" mkdir -p "${REMOTE_EDGE_DIR}/deploy"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_model}" "${remote_version_model}"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_meta}" "${remote_version_meta}"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_edge_env}" "${REMOTE_ROOT_ENV}"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_edge_env}" "${REMOTE_RUNTIME_DIR}/edge.env"
  remote_sudo_cmd "$user" "$host" "$port" install -m 644 "${remote_tmp_service}" "${REMOTE_INFERENCE_SERVICE}"
  remote_sudo_cmd "$user" "$host" "$port" install -m 755 "${remote_tmp_stop}" "${REMOTE_EDGE_DIR}/deploy/stop_inference.sh"
  remote_sudo_cmd "$user" "$host" "$port" install -m 755 "${remote_tmp_switch}" "${REMOTE_EDGE_DIR}/deploy/switch_model.sh"
  remote_run "$user" "$host" "$port" "rm -f '${remote_tmp_model}' '${remote_tmp_meta}' '${remote_tmp_edge_env}' '${remote_tmp_service}' '${remote_tmp_stop}' '${remote_tmp_switch}'" || true

  log_ok "已部署版本化模型与 meta：${remote_version_model}, ${remote_version_meta}"
  log_ok "已更新 /opt/visionops/.env，生产服务将指向版本化模型"

  if [[ "$NO_RESTART" == "true" ]]; then
    log_warn "--no-restart 已启用，跳过 systemd 重启与健康检查"
    rm -f "$context_file" "$local_meta_file" "$local_deploy_env" "$local_service_file"
    exit 0
  fi

  deploy_port="$(read_env_value PORT "$local_deploy_env")"
  [[ -n "$deploy_port" ]] || deploy_port="8082"

  log_info "通过 switch_model.sh 切换当前运行模型，避免端口残留和多进程冲突..."
  if ! remote_run "$user" "$host" "$port" "bash '${REMOTE_EDGE_DIR}/deploy/switch_model.sh' '${remote_version_model}' '${remote_version_meta}' '${deploy_port}' '${service_name}'"; then
    log_error "模型切换失败，尝试恢复旧 env/service..."
    remote_run "$user" "$host" "$port" "if [ -f '${backup_env}' ]; then sudo -n cp '${backup_env}' '${REMOTE_ROOT_ENV}'; fi; if [ -f '${backup_service}' ]; then sudo -n cp '${backup_service}' '${REMOTE_INFERENCE_SERVICE}'; fi" || true
    remote_sudo_cmd "$user" "$host" "$port" systemctl daemon-reload || true
    remote_sudo_cmd "$user" "$host" "$port" systemctl restart "${service_name}" || true
    rm -f "$context_file" "$local_meta_file" "$local_deploy_env" "$local_service_file"
    exit 1
  fi

  log_info "等待健康检查通过: ${health_url}"
  local health_ok="false"
  for i in $(seq 1 10); do
    if do_health_check "$user" "$host" "$port" "$health_url" >/dev/null; then
      health_ok="true"
      break
    fi
    sleep 1
  done

  if [[ "$health_ok" != "true" ]]; then
    log_error "服务健康检查失败，最近日志如下："
    remote_run "$user" "$host" "$port" "sudo -n journalctl -u ${service_name} -n 120 --no-pager || journalctl -u ${service_name} -n 120 --no-pager || true" || true
    rm -f "$context_file" "$local_meta_file" "$local_deploy_env" "$local_service_file"
    exit 1
  fi

  log_info "当前健康检查结果："
  local health_json
  health_json="$(remote_run "$user" "$host" "$port" "curl -sf '${health_url}'" || true)"
  echo "$health_json"
  if [[ "$health_json" != *"\"task\":\"${task}\""* && "$health_json" != *"\"task\": \"${task}\""* ]]; then
    log_error "健康检查 task 与部署任务不一致，请检查 /opt/visionops/.env"
    rm -f "$context_file" "$local_meta_file" "$local_deploy_env" "$local_service_file"
    exit 1
  fi

  remote_run "$user" "$host" "$port" "rm -f '${backup_env}' '${backup_service}'" || true
  rm -f "$context_file" "$local_meta_file" "$local_deploy_env" "$local_service_file"

  log_ok "部署完成"
  log_ok "task=${task}"
  log_ok "model=${remote_version_model}"
  log_ok "meta=${remote_version_meta}"
}

main "$@"