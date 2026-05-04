# VisionOps Makefile

.PHONY: help up down restart logs init render-task clean-generated \
        pipeline pipeline-force pipeline-task pipeline-dvc \
        pipeline-detection pipeline-classification \
		pipeline-obb pipeline-segmentation \
        train deploy deploy-code deploy-code-only \
        test-api mlflow monitor minio install install-edge \
        ingest-collected ingest-collected-force clean-checkpoints dvc-push dvc-pull

PYTHON ?= python
SERVICE ?=

help:
	@echo "VisionOps 命令清单"
	@echo "  make up                         启动 Docker Compose 服务"
	@echo "  make init                       初始化目录/DVC/MinIO"
	@echo "  make render-task                根据 pipeline/configs/task.yaml 生成运行配置"
	@echo "  make pipeline                   按当前 task.yaml 运行完整 Python 主链"
	@echo "  make pipeline-dvc               按当前 task.yaml 运行 DVC 主链"
	@echo "  make deploy                     上传当前 task 模型到边缘端"
	@echo "  make deploy-code                上传当前 task 模型并同步 edge/ 代码"
	@echo "  make deploy-code-only           仅同步 edge/ 代码"

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f $(SERVICE)

ingest-collected:
	$(PYTHON) server/data_ingest/ingest_uploaded_package.py --incoming-dir data/incoming

ingest-collected-force:
	$(PYTHON) server/data_ingest/ingest_uploaded_package.py --incoming-dir data/incoming --overwrite

init:
	@echo "初始化 DVC..."
	dvc init || true
	dvc remote add -d minio s3://visionops-data --force
	dvc remote modify minio endpointurl http://localhost:9000
	dvc remote modify minio access_key_id minioadmin
	dvc remote modify minio secret_access_key minioadmin123
	@echo "创建项目目录..."
	mkdir -p data/raw_detection data/processed_detection \
		data/raw_classification data/processed_classification \
		data/raw_obb data/processed_obb \
		data/raw_segmentation data/processed_segmentation \
		models/checkpoints_detection models/export_detection models/metrics_detection \
		models/checkpoints_classification models/export_classification models/metrics_classification \
		models/checkpoints_obb models/export_obb models/metrics_obb \
		models/checkpoints_segmentation models/export_segmentation models/metrics_segmentation \
		models/runs/detect models/runs/obb models/runs/segment \
		logs/retrain edge/runtime
	@echo "✓ 初始化完成"

render-task:
	$(PYTHON) -m pipeline.utils.render_task_config

clean-generated:
	rm -rf pipeline/configs/generated
	rm -f pipeline/configs/*.generated.yaml
	rm -f edge/runtime/class_names.yaml edge/runtime/edge.env
	@echo "✓ 已清理 generated 配置"

pipeline pipeline-task: render-task
	$(PYTHON) -m pipeline.stages.preprocess
	$(PYTHON) -m pipeline.stages.train
	$(PYTHON) -m pipeline.stages.evaluate
	$(PYTHON) -m pipeline.stages.export_onnx
	$(PYTHON) -m pipeline.stages.convert_rknn
	$(PYTHON) -m pipeline.stages.register_model

pipeline-force:
	dvc repro --force register_model

pipeline-dvc:
	dvc repro register_model

pipeline-detection: pipeline

pipeline-classification: pipeline

pipeline-obb: pipeline

pipeline-segmentation: pipeline

train: render-task
	$(PYTHON) -m pipeline.stages.train

deploy:
	bash edge/deploy/push.sh

deploy-code:
	bash edge/deploy/push.sh --code

deploy-code-only:
	bash edge/deploy/push.sh --code-only

test-api:
	@curl -sf http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "API 未响应，请先 make up"

mlflow:
	@echo "打开 http://localhost:5000"

monitor:
	@echo "打开 http://localhost:3000"

minio:
	@echo "打开 http://localhost:9001"

install:
	pip install -r requirements.txt

install-edge:
	pip install -r edge/requirements-edge.txt

clean-checkpoints:
	find models/checkpoints_detection models/checkpoints_classification -name "*.pt" ! -name "best.pt" -delete 2>/dev/null || true

dvc-push:
	dvc push

dvc-pull:
	dvc pull

workflow-ui:
	$(PYTHON) -m uvicorn server.workflow.control_panel_app:app --host 0.0.0.0 --port 8091 --reload

accept-reviewed-detection:
	$(PYTHON) server/workflow/accept_reviewed_detection.py --move
