# VisionOps

> 端到端视觉 AI 软件平台  
> 当前主线：**YOLOv8 Detection**  
> 服务器训练 × ONNX 导出 × RK3588 RKNN 边缘部署 × 全自动 MLOps 管理

```text
服务器端                                                边缘端（RK3588）
┌──────────────────────────────────────┐              ┌──────────────────────────────┐
│ 检测数据预处理                       │              │ rknnlite2 推理引擎           │
│ YOLOv8 Detection 训练                │ ───────────▶ │ INT8 量化 NPU 加速           │
│ ONNX → RKNN 转换                     │              │ FastAPI 推理服务 :8080       │
│ MLflow 实验追踪                      │              │ Prometheus 指标 :9091        │
│ MLflow Model Registry                │ ◀─────────── │ 数据漂移检测 + 上报          │
│ DVC 数据版本管理                     │              │ systemd 自启动服务           │
│ MinIO 模型/数据存储                  │              └──────────────────────────────┘
│ FastAPI 后端 :8000                   │
│ Grafana 监控 :3000                   │
│ 自动再训练调度器                     │
└──────────────────────────────────────┘
                      │
                      ▼
              GitHub Actions CI/CD
        Lint → 测试 → 训练 → 质量门禁 → 边缘部署
        

## 技术栈

| 层级    | 技术                                |
| ----- | --------------------------------- |
| 模型训练  | PyTorch 2.x, YOLOv8 (Ultralytics) |
| 模型格式  | PyTorch → ONNX → RKNN             |
| 边缘推理  | rknnlite2, RK3588 NPU             |
| 实验追踪  | MLflow                            |
| 数据版本  | DVC + MinIO                       |
| 服务后端  | FastAPI + Uvicorn                 |
| 监控告警  | Prometheus + Grafana              |
| CI/CD | GitHub Actions                    |
| 容器化   | Docker Compose                    |

---

## 快速启动

### 前置条件

Docker + Docker Compose
Python 3.11+
可选：NVIDIA GPU + CUDA（用于训练加速）
可选：独立的 RKNN 转换环境（推荐，不建议污染训练主环境）

### 1. 启动 MLOps 基础服务

```bash
cd visionops_v2
make up
```

服务端口：

| 服务            | 地址                                                       | 凭证                         |
| ------------- | -------------------------------------------------------- | -------------------------- |
| MLflow UI     | [http://localhost:5000](http://localhost:5000)           | —                          |
| VisionOps API | [http://localhost:8000/docs](http://localhost:8000/docs) | —                          |
| MinIO Console | [http://localhost:9001](http://localhost:9001)           | minioadmin / minioadmin123 |
| Grafana       | [http://localhost:3000](http://localhost:3000)           | admin / visionops123       |
| Prometheus    | [http://localhost:9090](http://localhost:9090)           | —                          |


### 2. 初始化项目

```bash
make init          # 初始化 DVC + MinIO buckets + 目录结构
```

### 3. 准备数据

将 YOLO 检测数据放入：：

```
data/raw_detection/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```
其中：

images/train、images/val 为训练/验证图像
labels/train、labels/val 为对应 YOLO 标签
data.yaml 中的 names 需与检测任务类别一致

说明：仓库默认已将 data/ 与 models/ 加入 .gitignore，因此 GitHub 上不会包含你的本地数据与训练产物；克隆仓库后需要在本地自行准备。

### 4. 配置 detection 流水线参数

主要配置文件位于：
```
pipeline/configs/
├── detection_data.yaml
├── detection_train.yaml
├── detection_export.yaml
├── detection_rknn.yaml
└── detection_mlops.yaml
```

你通常需要修改的是：

detection_data.yaml：数据路径、类别数、预处理开关
detection_train.yaml：模型结构、训练超参数
detection_export.yaml：ONNX 导出参数
detection_rknn.yaml：RKNN 转换目标平台与量化参数
detection_mlops.yaml：模型注册、晋升阈值、边缘设备信息

### 5. 运行完整 detection 流水线

```bash
make pipeline
```
等价于
```bash
dvc repro \
  preprocess_detection \
  train_detection \
  evaluate_detection \
  export_onnx_detection \
  convert_rknn_detection \
  register_model_detection
```

流水线自动执行：
1. **检测数据预处理** — resize, 增强, 训练/验证集划分
2. **YOLOv8 检测模型训练** — 自动记录实验到 MLflow
3. **检测指标评估** — 计算 mAP/accuracy
4. **ONNX 导出** — opset 12, 静态 shape
5. **RKNN 转换** — INT8量化（需要 rknn-toolkit2 x86环境）
6. **MLflow Registry 注册** — 满足阈值自动晋升 Production

强制重跑所有 detection stage：

```bash
make pipeline-force
```

### 6. 部署到 RK3588 设备

```bash
# 部署到所有已配置设备
make deploy

# 部署到指定设备
make deploy DEVICE=rk3588-001
```

Detection 主线输出目录

典型输出如下：
```bash
data/processed_detection/
├── images/
├── labels/
├── data.yaml
└── dataset_stats.json

models/checkpoints_detection/
models/export_detection/
├── model.onnx
├── model.rknn
└── rknn_perf_report.json

models/metrics_detection/
├── eval_metrics.json
├── train_metrics.json
└── registry_result.json
```

---

## 配置参考

### 模型晋升阈值

编辑：

pipeline/configs/detection_mlops.yaml

```yaml
registry:
  promotion_threshold:
    map50: 0.70
    map50_95: 0.45
    latency_ms: 50
```
含义：

map50 达到阈值才允许自动晋升
map50_95 用于更严格的质量控制
latency_ms 用于约束边缘部署时延

### RKNN 转换

编辑：

pipeline/configs/detection_rknn.yaml

```yaml
target_platform: rk3588
quantization:
  do_quantization: true
  dataset_size: 100     # 校准图像数量
```

### 边缘设备注册

编辑：

pipeline/configs/detection_mlops.yaml

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
