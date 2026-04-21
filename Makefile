# VisionOps Makefile — 常用命令快捷方式

.PHONY: help up down train pipeline deploy monitor retrain check-triggers test-api

PYTHON ?= python
MODEL_VERSION ?= latest
DEVICE ?=
SERVICE ?=

## ── 帮助 ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "VisionOps 命令清单"
	@echo "────────────────────────────────────────────────────"
	@echo "  make up              启动所有MLOps服务（Docker）"
	@echo "  make down            停止所有服务"
	@echo "  make init            初始化项目（DVC + MinIO）"
	@echo ""
	@echo "  make train           运行训练流水线（DVC repro）"
	@echo "  make pipeline        运行完整MLOps流水线"
	@echo "  make pipeline-force  强制重跑所有stage"
	@echo "  make deploy          部署RKNN模型到所有边缘设备"
	@echo "  make deploy DEVICE=rk3588-001  部署到指定设备"
	@echo ""
	@echo "  make retrain         手动触发再训练检查（一次）"
	@echo "  make retrain-force   强制重训练（跳过条件检查）"
	@echo "  make check-triggers  查看当前再训练触发状态"
	@echo ""
	@echo "  make test-api        快速测试 API 健康状态"
	@echo "  make diff            查看DVC pipeline变更"
	@echo "  make logs            查看服务日志 (SERVICE=api)"
	@echo ""
	@echo "  make mlflow          打开MLflow UI"
	@echo "  make monitor         打开Grafana监控面板"
	@echo "  make minio           打开MinIO控制台"
	@echo ""
	@echo "  make install         安装Python依赖"
	@echo "  make install-edge    安装边缘端依赖（RK3588上运行）"
	@echo "────────────────────────────────────────────────────"

## ── 服务管理 ──────────────────────────────────────────────────
up:
	docker compose up -d
	@echo "✓ 所有服务已启动"
	@echo "  MLflow:    http://localhost:5000"
	@echo "  API:       http://localhost:8000"
	@echo "  MinIO:     http://localhost:9001"
	@echo "  Grafana:   http://localhost:3000  (admin/visionops123)"
	@echo "  Prometheus:http://localhost:9090"

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f $(SERVICE)

## ── 项目初始化 ────────────────────────────────────────────────
init:
	@echo "初始化DVC..."
	dvc init || true
	dvc remote add -d minio s3://visionops-data --force
	dvc remote modify minio endpointurl http://localhost:9000
	dvc remote modify minio access_key_id minioadmin
	dvc remote modify minio secret_access_key minioadmin123
	@echo "创建数据目录..."
	mkdir -p data/raw data/processed models/checkpoints models/export models/metrics logs/retrain
	@echo "✓ 初始化完成"

## ── 训练流水线 ────────────────────────────────────────────────
train:
	$(PYTHON) pipeline/stages/train.py

pipeline:
	@echo "运行完整MLOps流水线..."
	dvc repro
	@echo "✓ 流水线执行完毕"

pipeline-force:
	dvc repro --force

diff:
	dvc diff

params:
	dvc params diff

## ── 再训练调度 ────────────────────────────────────────────────
retrain:
	@echo "执行一次再训练条件检查..."
	$(PYTHON) server/mlops/retrain_scheduler.py --mode once

retrain-force:
	@echo "强制触发重训练..."
	$(PYTHON) server/mlops/retrain_scheduler.py --force

retrain-loop:
	@echo "启动持续再训练监控（Ctrl+C 停止）..."
	$(PYTHON) server/mlops/retrain_scheduler.py --mode loop --interval 300

check-triggers:
	@echo "当前再训练触发状态："
	@$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from server.mlops.retrain_scheduler import RetrainScheduler; \
s = RetrainScheduler(); \
should, reason = s.check_triggers(); \
print(f'  触发: {should}'); print(f'  原因: {reason}')" 2>/dev/null || \
	echo "  (需要先 make install)"

## ── 部署 ──────────────────────────────────────────────────────
deploy:
	@echo "部署RKNN模型到边缘设备..."
	bash edge/deploy/push.sh models/export/model.rknn $(DEVICE)

## ── API 测试 ──────────────────────────────────────────────────
test-api:
	@echo "测试 API 健康状态..."
	@curl -sf http://localhost:8000/health | python3 -m json.tool 2>/dev/null || \
	 echo "API 未响应，请先 make up"
	@echo ""
	@echo "测试实验列表..."
	@curl -sf http://localhost:8000/api/v1/experiments | python3 -m json.tool 2>/dev/null || true
	@echo ""
	@echo "测试模型列表..."
	@curl -sf http://localhost:8000/api/v1/models | python3 -m json.tool 2>/dev/null || true

## ── 监控 ──────────────────────────────────────────────────────
mlflow:
	@echo "打开 http://localhost:5000"

monitor:
	@echo "打开 http://localhost:3000 (admin/visionops123)"

minio:
	@echo "打开 http://localhost:9001 (minioadmin/minioadmin123)"

## ── 安装依赖 ──────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-edge:
	pip install -r edge/requirements-edge.txt

## ── 工具 ──────────────────────────────────────────────────────
clean-checkpoints:
	find models/checkpoints -name "*.pt" ! -name "best.pt" -delete
	@echo "✓ 清理旧checkpoint完成"

dvc-push:
	dvc push

dvc-pull:
	dvc pull
