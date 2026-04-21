# VisionOps

> 端到端视觉AI软件平台 — 服务器训练 × RK3588 NPU边缘推理 × 全自动 MLOps 管理

```
服务器端                        边缘端 (RK3588)
┌─────────────────────────┐    ┌──────────────────────────┐
│  数据预处理              │    │  rknnlite2 推理引擎       │
│  YOLOv8/MobileNetV3训练  │───▶│  INT8量化 NPU加速        │
│  ONNX → RKNN 转换        │    │  FastAPI 推理服务 :8080   │
│  MLflow 实验追踪         │    │  Prometheus 指标 :9091   │
│  MLflow Model Registry  │◀───│  数据漂移检测 + 上报      │
│  DVC 数据版本管理        │    │  systemd 自启动服务       │
│  MinIO 模型/数据存储     │    └──────────────────────────┘
│  FastAPI 后端 :8000      │
│  Grafana 监控 :3000      │
│  自动再训练调度器        │
└─────────────────────────┘
           │
           ▼
   GitHub Actions CI/CD
   Lint → 测试 → 训练 → 质量门禁 → 边缘部署
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 模型训练 | PyTorch 2.x, YOLOv8 (ultralytics), MobileNetV3, EfficientNet |
| 模型格式 | PyTorch → ONNX (opset 12) → RKNN (INT8量化) |
| 边缘推理 | rknnlite2, RK3588 NPU (3-core, 6 TOPS) |
| 实验追踪 | MLflow 2.13 + PostgreSQL 后端 |
| 数据版本 | DVC + MinIO (S3兼容) |
| 服务后端 | FastAPI + uvicorn |
| 监控告警 | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| 容器化 | Docker Compose |

---

## 快速启动

### 前置条件

- Docker + Docker Compose
- Python 3.11+
- （可选）NVIDIA GPU + CUDA（加速训练）

### 1. 启动 MLOps 基础服务

```bash
cd visionops
make up
```

服务端口：

| 服务 | 地址 | 凭证 |
|------|------|------|
| MLflow UI | http://localhost:5000 | — |
| VisionOps API | http://localhost:8000/docs | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |
| Grafana | http://localhost:3000 | admin / visionops123 |
| Prometheus | http://localhost:9090 | — |

### 2. 初始化项目

```bash
make init          # 初始化 DVC + MinIO buckets + 目录结构
```

### 3. 准备数据

将原始图像放入 `data/raw/`，支持以下结构：

```
data/raw/
├── class_a/
│   ├── img001.jpg
│   └── ...
└── class_b/
    └── ...
```

或直接放入图像文件（无类别结构）。

### 4. 配置训练参数

编辑 `pipeline/configs/train.yaml`：

```yaml
model:
  architecture: yolov8n   # yolov8n/s/m, mobilenetv3, efficientnet_b0
  num_classes: 10

train:
  epochs: 100
  batch_size: 16
  lr: 0.001
```

### 5. 运行完整流水线

```bash
make pipeline
# 等价于: dvc repro
```

流水线自动执行：
1. **数据预处理** — resize, 增强, 训练/验证集划分
2. **模型训练** — 自动记录实验到 MLflow
3. **模型评估** — 计算 mAP/accuracy
4. **ONNX 导出** — opset 12, 静态 shape
5. **RKNN 转换** — INT8量化（需要 rknn-toolkit2 x86环境）
6. **模型注册** — 满足阈值自动晋升 Production

### 6. 部署到 RK3588 设备

```bash
# 部署到所有已配置设备
make deploy

# 部署到指定设备
make deploy DEVICE=rk3588-001
```

---

## 配置参考

### 模型晋升阈值 (`pipeline/configs/mlops.yaml`)

```yaml
registry:
  promotion_threshold:
    accuracy: 0.85      # mAP50 >= 0.85 才自动晋升 Production
    latency_ms: 50      # NPU推理延迟 <= 50ms
```

### RKNN 转换 (`pipeline/configs/rknn.yaml`)

```yaml
target_platform: rk3588
quantization:
  do_quantization: true
  dataset_size: 100     # 校准图像数量
```

### 边缘设备注册 (`pipeline/configs/mlops.yaml`)

```yaml
edge_devices:
  - id: rk3588-001
    host: 192.168.1.100
    port: 22
    user: root
    deploy_path: /opt/visionops/models/
```

---

## 边缘设备部署

### 初始化新设备

```bash
# 在 RK3588 设备上执行
sudo bash edge/deploy/setup_edge.sh http://<服务器IP>:8000 rk3588-001
```

### 验证推理服务

```bash
# 健康检查
curl http://<设备IP>:8080/health

# 单张图片推理
curl -X POST http://<设备IP>:8080/infer \
  -F "file=@test.jpg"

# 查看性能指标
curl http://<设备IP>:8080/stats
```

### NPU 核心配置

RK3588 有 3 个 NPU 核心，通过 `npu_core` 参数控制：

| 值 | 说明 |
|----|------|
| `auto` | 自动调度（默认推荐） |
| `core_0_1_2` | 三核并行（最高吞吐） |
| `core_0` | 单核（低功耗） |

---

## MLOps 自动化

### 自动再训练触发条件

编辑 `pipeline/configs/mlops.yaml`：

```yaml
retraining:
  enabled: true
  triggers:
    accuracy_drop: 0.05     # 精度下降 5%
    data_drift_threshold: 0.1
    new_data_size: 500       # 累积 500 条新数据
```

手动检查：

```bash
make check-triggers   # 查看当前是否需要再训练
make retrain          # 执行一次检查
make retrain-force    # 强制触发重训练
```

### GitHub Actions 集成

在仓库 Settings → Secrets 中配置：

| Secret | 说明 |
|--------|------|
| `MLFLOW_TRACKING_URI` | MLflow 服务地址 |
| `MINIO_ENDPOINT` | MinIO S3 端点 |
| `MINIO_ACCESS_KEY` | MinIO 访问密钥 |
| `MINIO_SECRET_KEY` | MinIO 密钥 |
| `EDGE_SSH_PRIVATE_KEY` | 边缘设备 SSH 私钥 |
| `EDGE_DEVICE_HOST` | 边缘设备 IP |

---

## 项目结构

```
visionops/
├── pipeline/
│   ├── configs/            # 所有配置文件
│   │   ├── train.yaml      # 训练超参数
│   │   ├── rknn.yaml       # RKNN转换参数
│   │   ├── mlops.yaml      # MLOps管理配置
│   │   ├── export.yaml     # ONNX导出配置
│   │   └── prometheus.yml  # Prometheus抓取配置
│   └── stages/             # DVC pipeline stages
│       ├── preprocess.py   # 数据预处理
│       ├── train.py        # 模型训练
│       ├── evaluate.py     # 模型评估
│       ├── export_onnx.py  # ONNX导出
│       ├── convert_rknn.py # RKNN转换
│       └── register_model.py # 模型注册
├── server/
│   ├── api/
│   │   ├── main.py         # FastAPI 后端
│   │   └── Dockerfile
│   ├── training/
│   │   └── model_utils.py  # 模型构建工具
│   └── mlops/
│       └── retrain_scheduler.py  # 自动再训练调度
├── edge/
│   ├── inference/
│   │   └── engine.py       # RK3588推理引擎
│   ├── monitor/
│   │   └── monitor.py      # 边缘监控+漂移检测
│   └── deploy/
│       ├── push.sh         # 模型推送脚本
│       ├── setup_edge.sh   # 边缘设备初始化
│       ├── visionops-inference.service  # systemd 推理服务
│       └── visionops-monitor.service   # systemd 监控服务
├── .github/workflows/
│   ├── mlops_pipeline.yml      # 主CI/CD流水线
│   └── scheduled_retrain.yml  # 定时再训练
├── docker-compose.yml      # MLOps服务栈
├── dvc.yaml               # DVC流水线定义
├── Makefile               # 常用命令快捷方式
└── requirements.txt       # 服务器端依赖
```

---

## 常用命令

```bash
make help            # 查看所有命令
make up              # 启动服务
make pipeline        # 运行训练流水线
make deploy          # 部署到边缘设备
make test-api        # 测试 API
make retrain         # 检查再训练条件
make monitor         # 打开 Grafana
make mlflow          # 打开 MLflow UI
```

## RKNN 转换说明

> ⚠️ `rknn-toolkit2`（用于 ONNX→RKNN 转换）**只能在 x86 Linux 环境安装**。  
> `rknnlite2`（用于 RK3588 NPU 推理）**只在 ARM 设备上安装**。

| 环境 | 工具 | 用途 |
|------|------|------|
| 开发/CI 服务器 (x86) | rknn-toolkit2 | 模型转换 + 量化 |
| RK3588 边缘设备 (ARM) | rknnlite2 | NPU 推理 |

在没有 rknn-toolkit2 的环境，`convert_rknn.py` 会自动切换到**模拟模式**，生成占位符文件，不影响其他 Stage 执行。
