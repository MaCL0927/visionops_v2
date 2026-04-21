#!/usr/bin/env bash
# ============================================================
# VisionOps RK3588 边缘设备初始化安装脚本
# 在全新 RK3588 设备上一键部署推理环境
#
# 用法:
#   chmod +x edge/deploy/setup_edge.sh
#   sudo ./edge/deploy/setup_edge.sh [SERVER_URL] [DEVICE_ID]
#
# 例:
#   sudo ./edge/deploy/setup_edge.sh http://192.168.1.1:8000 rk3588-001
# ============================================================

set -euo pipefail

SERVER_URL="${1:-http://192.168.1.1:8000}"
DEVICE_ID="${2:-rk3588-001}"
INSTALL_DIR="/opt/visionops"
VENV_DIR="$INSTALL_DIR/venv"
PYTHON_BIN="python3"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 检查 root ──
[ "$EUID" -ne 0 ] && { log_error "请以 root 运行"; exit 1; }

log_info "=== VisionOps 边缘初始化 ==="
log_info "设备ID: $DEVICE_ID"
log_info "服务器: $SERVER_URL"
log_info "安装目录: $INSTALL_DIR"

# ── 1. 系统依赖 ──
log_info "安装系统依赖..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libopencv-dev \
    curl wget git \
    libgomp1 \
    systemd

# ── 2. 创建目录结构 ──
mkdir -p "$INSTALL_DIR"/{models,logs,edge}
mkdir -p "$INSTALL_DIR/models"

# ── 3. 创建Python虚拟环境 ──
log_info "创建Python虚拟环境..."
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ── 4. 安装Python依赖 ──
log_info "安装Python依赖..."
# rknnlite2 需要手动安装（ARM架构专用包）
pip install -q \
    fastapi \
    "uvicorn[standard]" \
    numpy \
    opencv-python-headless \
    requests \
    psutil \
    pyyaml

# 尝试安装 rknnlite2（ARM设备专用，x86上会失败，属正常）
log_info "尝试安装 rknnlite2..."
pip install rknn-toolkit-lite2 2>/dev/null \
    || log_warn "rknnlite2 安装失败（非ARM环境），推理将运行模拟模式"

# ── 5. 复制推理代码 ──
log_info "部署推理代码..."
# 如果是从开发机推送过来的包，则解压；否则假设代码已在 $INSTALL_DIR
if [ -f "/tmp/visionops_edge.tar.gz" ]; then
    tar -xzf /tmp/visionops_edge.tar.gz -C "$INSTALL_DIR/edge/"
    log_success "代码包已解压"
fi

# ── 6. 写入环境配置 ──
cat > "$INSTALL_DIR/.env" <<EOF
DEVICE_ID=$DEVICE_ID
SERVER_URL=$SERVER_URL
MODEL_PATH=$INSTALL_DIR/models/current.rknn
INFERENCE_URL=http://localhost:8080
REPORT_INTERVAL=60
NPU_CORE=auto
EOF
log_success "环境配置写入: $INSTALL_DIR/.env"

# ── 7. 安装 systemd 服务 ──
log_info "安装 systemd 服务..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 注入设备 ID 到 service 文件
sed "s/rk3588-001/$DEVICE_ID/g; s|http://192.168.1.1:8000|$SERVER_URL|g" \
    "$SCRIPT_DIR/visionops-inference.service" \
    > /etc/systemd/system/visionops-inference.service

sed "s/rk3588-001/$DEVICE_ID/g; s|http://192.168.1.1:8000|$SERVER_URL|g" \
    "$SCRIPT_DIR/visionops-monitor.service" \
    > /etc/systemd/system/visionops-monitor.service

systemctl daemon-reload
systemctl enable visionops-inference visionops-monitor

log_success "systemd 服务已注册（将在下次重启时自动启动）"

# ── 8. 创建占位模型（等待第一次部署） ──
if [ ! -f "$INSTALL_DIR/models/current.rknn" ]; then
    echo "PLACEHOLDER - awaiting first deployment" > "$INSTALL_DIR/models/current.rknn"
    log_warn "占位模型已创建，等待服务器首次部署真实模型"
fi

echo ""
log_success "=========================================="
log_success "VisionOps 边缘初始化完成！"
log_success "  设备ID: $DEVICE_ID"
log_success "  安装路径: $INSTALL_DIR"
log_success ""
log_success "后续操作："
log_info "  1. 从服务器部署模型: ./edge/deploy/push.sh models/export/model.rknn $DEVICE_ID"
log_info "  2. 手动启动服务: systemctl start visionops-inference visionops-monitor"
log_info "  3. 查看日志: journalctl -u visionops-inference -f"
log_success "=========================================="
