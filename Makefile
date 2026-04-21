# VisionOps Makefile — 常用命令快捷方式

.PHONY: help up down restart logs init \
        train train-detection \
        pipeline pipeline-force \
        pipeline-detection pipeline-detection-force \
        pipeline-classification pipeline-classification-force \
        deploy deploy-detection deploy-classification \
        retrain retrain-force retrain-loop check-triggers \
        test-api mlflow monitor minio \
        install install-edge \
        clean-checkpoints dvc-push dvc-pull

PYTHON ?= python
MODEL_VERSION ?= latest
DEVICE ?=
SERVICE ?=

## ── 帮助 ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "VisionOps 命令清单"
	@echo "────────────────────────────────────────────────────"
	@echo " make up                     启动所有 MLOps 服务（Docker）"
	@echo " make down                   停止所有服务"
	@echo " make restart                重启所有服务"
	@echo " make logs SERVICE=api       查看指定服务日志"
	@echo " make init                   初始化项目（DVC + MinIO + 目录）"
	@echo ""
	@echo " make train                  运行 detection 训练脚本（默认主线）"
	@echo " make train-detection        运行 detection 训练脚本"
	@echo " make pipeline               运行 detection 完整 MLOps 流水线（默认主线）"
	@echo " make pipeline-force         强制重跑 detection 所有 stage"
	@echo " make deploy                 部署 detection RKNN 模型到边缘设备（默认主线）"
	@echo " make deploy DEVICE=rk3588-001  部署 detection 模型到指定设备"
	@echo ""
	@echo " make pipeline-classification        运行 legacy/classification 流水线"
	@echo " make pipeline-classification-force  强制重跑 legacy/classification 所有 stage"
	@echo " make deploy-classification          部署 legacy/classification RKNN 模型"
	@echo ""
	@echo " make retrain               手动触发再训练检查（一次）"
	@echo " make retrain-force         强制重训练（跳过条件检查）"
	@echo " make retrain-loop          启动持续再训练监控"
	@echo " make check-triggers        查看当前再训练触发状态"
	@echo ""
	@echo " make test-api              快速测试 API 健康状态"
	@echo " make mlflow                打开 MLflow UI"
	@echo " make monitor               打开 Grafana 监控面板"
	@echo " make minio                 打开 MinIO 控制台"
	@echo ""
	@echo " make install               安装 Python 依赖"
	@echo " make install-edge          安装边缘端依赖（RK3588 上运行）"
	@echo " make dvc-push              推送 DVC 数据到远端"
	@echo " make dvc-pull              从远端拉取 DVC 数据"
	@echo "────────────────────────────────────────────────────"

## ── 服务管理 ──────────────────────────────────────────────────
up:
	docker compose up -d
	@echo "✓ 所有服务已启动"
	@echo "  MLflow:    http://localhost:5000"
	@echo "  API:       http://localhost:8000"
	@echo "  MinIO:     http://localhost:9001"
	@echo "  Grafana:   http://localhost:3000 (admin/visionops123)"
	@echo "  Prometheus:http://localhost:9090"

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f $(SERVICE)

## ── 项目初始化 ────────────────────────────────────────────────
init:
	@echo "初始化 DVC..."
	dvc init || true
	dvc remote add -d minio s3://visionops-data --force
	dvc remote modify minio endpointurl http://localhost:9000
	dvc remote modify minio access_key_id minioadmin
	dvc remote modify minio secret_access_key minioadmin123
	@echo "创建项目目录..."
	mkdir -p \
		data/raw_detection data/processed_detection \
		models/checkpoints_detection models/export_detection models/metrics_detection \
		models/runs/detect \
		logs/retrain
	@echo "如需兼容 legacy/classification 流程，请手动创建："
	@echo "  data/raw data/processed"
	@echo "  models/checkpoints models/export models/metrics"
	@echo "✓ 初始化完成"

## ── 训练流水线（默认主线：detection）──────────────────────────
train: train-detection

train-detection:
	$(PYTHON) pipeline/stages/train_detection.py

pipeline: pipeline-detection

pipeline-force: pipeline-detection-force

pipeline-detection:
	@echo "运行 detection 完整 MLOps 流水线..."
	dvc repro preprocess_detection train_detection evaluate_detection export_onnx_detection convert_rknn_detection register_model_detection
	@echo "✓ detection 流水线执行完毕"

pipeline-detection-force:
	dvc repro --force preprocess_detection train_detection evaluate_detection export_onnx_detection convert_rknn_detection register_model_detection

## ── legacy/classification 流水线 ─────────────────────────────
train-classification:
	$(PYTHON) pipeline/stages/train.py

pipeline-classification:
	@echo "运行 legacy/classification 完整 MLOps 流水线..."
	dvc repro preprocess train evaluate export_onnx convert_rknn register_model
	@echo "✓ legacy/classification 流水线执行完毕"

pipeline-classification-force:
	dvc repro --force preprocess train evaluate export_onnx convert_rknn register_model

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
print(f'  触发: {should}'); \
print(f'  原因: {reason}')" 2>/dev/null || \
	echo "  (需要先 make install)"

## ── 部署（默认主线：detection）────────────────────────────────
deploy: deploy-detection

deploy-detection:
	@echo "部署 detection RKNN 模型到边缘设备..."
	bash edge/deploy/push.sh models/export_detection/model.rknn $(DEVICE)

deploy-classification:
	@echo "部署 legacy/classification RKNN 模型到边缘设备..."
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
	find models/checkpoints -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true
	find models/checkpoints_detection -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true
	@echo "✓ 清理旧 checkpoint 完成"

dvc-push:
	dvc push

dvc-pull:
	dvc pull