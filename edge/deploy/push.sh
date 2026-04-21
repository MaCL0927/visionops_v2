#!/usr/bin/env bash
# ============================================================
# VisionOps 边缘部署脚本
# 将RKNN模型推送到RK3588设备并重载推理服务
# 
# 用法:
#   ./edge/deploy/push.sh [MODEL_PATH] [DEVICE_ID]
#   ./edge/deploy/push.sh models/export/model.rknn rk3588-001
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/pipeline/configs/mlops.yaml"

# ── 颜色输出 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 参数 ──
MODEL_PATH="${1:-models/export/model.rknn}"
DEVICE_FILTER="${2:-}"  # 空=部署到所有设备

# ── 检查模型文件 ──
if [ ! -f "$MODEL_PATH" ]; then
    log_error "模型文件不存在: $MODEL_PATH"
    exit 1
fi

MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
MODEL_MD5=$(md5sum "$MODEL_PATH" | cut -d' ' -f1)
log_info "准备部署模型: $MODEL_PATH (大小: $MODEL_SIZE, MD5: $MODEL_MD5)"

# ── 读取设备列表（从mlops.yaml） ──
read_devices() {
    python3 - <<'EOF'
import yaml, json, sys
with open("pipeline/configs/mlops.yaml") as f:
    cfg = yaml.safe_load(f)
devices = cfg.get("edge_devices", [])
print(json.dumps(devices))
EOF
}

DEVICES=$(read_devices)

if [ -z "$DEVICES" ] || [ "$DEVICES" = "[]" ]; then
    log_warn "mlops.yaml 中没有配置 edge_devices，跳过部署"
    exit 0
fi

# ── 部署到单台设备 ──
deploy_to_device() {
    local device_id="$1"
    local host="$2"
    local port="${3:-22}"
    local user="${4:-root}"
    local deploy_path="${5:-/opt/visionops/models/}"
    local service_name="${6:-visionops-inference}"

    log_info "─────────────────────────────────────────"
    log_info "部署到设备: $device_id ($user@$host:$port)"

    # SSH选项
    SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $port"
    SCP_OPTS="-o StrictHostKeyChecking=no -P $port"

    # 1. 测试连接
    if ! ssh $SSH_OPTS "$user@$host" "echo connected" &>/dev/null; then
        log_error "无法连接到设备 $device_id ($host:$port)"
        return 1
    fi
    log_success "设备连通性验证通过"

    # 2. 确保目标目录存在
    ssh $SSH_OPTS "$user@$host" "mkdir -p $deploy_path"

    # 3. 备份当前模型
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ssh $SSH_OPTS "$user@$host" \
        "[ -f ${deploy_path}current.rknn ] && \
         cp ${deploy_path}current.rknn ${deploy_path}backup_${TIMESTAMP}.rknn || true"
    log_info "已备份当前模型"

    # 4. 传输新模型
    log_info "上传模型文件..."
    scp $SCP_OPTS "$MODEL_PATH" "$user@$host:${deploy_path}model_${TIMESTAMP}.rknn"
    log_success "模型上传完成"

    # 5. 验证MD5
    REMOTE_MD5=$(ssh $SSH_OPTS "$user@$host" \
        "md5sum ${deploy_path}model_${TIMESTAMP}.rknn | cut -d' ' -f1")
    if [ "$REMOTE_MD5" != "$MODEL_MD5" ]; then
        log_error "MD5校验失败！本地: $MODEL_MD5, 远端: $REMOTE_MD5"
        ssh $SSH_OPTS "$user@$host" "rm -f ${deploy_path}model_${TIMESTAMP}.rknn"
        return 1
    fi
    log_success "MD5校验通过"

    # 6. 原子切换（移动到current.rknn）
    ssh $SSH_OPTS "$user@$host" \
        "mv ${deploy_path}model_${TIMESTAMP}.rknn ${deploy_path}current.rknn"

    # 7. 重载推理服务（通过API热重载，无需重启进程）
    RELOAD_RESULT=$(ssh $SSH_OPTS "$user@$host" \
        "curl -sf -X POST http://localhost:8080/reload?model_path=${deploy_path}current.rknn || echo 'api_unavailable'")

    if echo "$RELOAD_RESULT" | grep -q '"success": true'; then
        log_success "推理服务热重载成功"
    elif echo "$RELOAD_RESULT" | grep -q 'api_unavailable'; then
        # API不可用，尝试systemd重启
        log_warn "API热重载不可用，尝试重启systemd服务..."
        ssh $SSH_OPTS "$user@$host" "systemctl restart $service_name || true"
        sleep 3
    fi

    # 8. 验证服务健康
    HEALTH=$(ssh $SSH_OPTS "$user@$host" \
        "curl -sf http://localhost:8080/health || echo 'unhealthy'")

    if echo "$HEALTH" | grep -q '"status": "ok"'; then
        log_success "设备 $device_id 部署成功！推理服务运行正常"
    else
        log_error "服务健康检查失败，回滚中..."
        ssh $SSH_OPTS "$user@$host" \
            "[ -f ${deploy_path}backup_${TIMESTAMP}.rknn ] && \
             mv ${deploy_path}backup_${TIMESTAMP}.rknn ${deploy_path}current.rknn && \
             systemctl restart $service_name || true"
        return 1
    fi

    return 0
}

# ── 主循环：部署到所有设备 ──
SUCCESS_COUNT=0
FAIL_COUNT=0

while IFS= read -r device_json; do
    device_id=$(echo "$device_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id','unknown'))")
    
    # 如果指定了设备过滤器，跳过不匹配的设备
    if [ -n "$DEVICE_FILTER" ] && [ "$device_id" != "$DEVICE_FILTER" ]; then
        continue
    fi

    host=$(echo "$device_json"     | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('host',''))")
    port=$(echo "$device_json"     | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('port',22))")
    user=$(echo "$device_json"     | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('user','root'))")
    deploy_path=$(echo "$device_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('deploy_path','/opt/visionops/models/'))")
    service=$(echo "$device_json"  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('service_name','visionops-inference'))")

    if deploy_to_device "$device_id" "$host" "$port" "$user" "$deploy_path" "$service"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi

done < <(echo "$DEVICES" | python3 -c "import sys,json; [print(json.dumps(d)) for d in json.load(sys.stdin)]")

# ── 汇总 ──
echo ""
log_info "═══════════════════════════════════════════"
log_info "部署汇总:"
log_success "  成功: $SUCCESS_COUNT 台设备"
[ $FAIL_COUNT -gt 0 ] && log_error "  失败: $FAIL_COUNT 台设备"
log_info "═══════════════════════════════════════════"

[ $FAIL_COUNT -gt 0 ] && exit 1 || exit 0
