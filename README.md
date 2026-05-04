# VisionOps

VisionOps 是一个面向 RK3588 边缘设备的端到端视觉 AI 软件平台，覆盖 **数据采集 → 服务端接收 → 标注与审核 → DVC/MLflow 训练流水线 → ONNX/RKNN 导出 → 边缘端部署 → 模型验证/生产推理** 的完整闭环。

当前主链已经统一为：

```text
task.yaml 单一真源
+ 统一 pipeline stage 分发
+ detection / classification / obb_detection / segmentation 多任务分支
+ MLflow 实验与模型状态追踪
+ RK3588 RKNN 边缘部署
+ 服务端 Web 控制台
+ 边缘端 Web 客户端
```

---

## 1. 当前能力概览

### 已支持任务

| 任务类型 | task.type | 模型类型 | 主要用途 |
|---|---|---|---|
| 目标检测 | `detection` | YOLOv8 Detect | 水平框检测 |
| 图像分类 | `classification` | YOLOv8 Classify | 单图类别判断 |
| 旋转框检测 | `obb_detection` | YOLOv8 OBB | 旋转目标、倾斜目标检测 |
| 实例分割 | `segmentation` | YOLOv8 Seg | 像素级目标区域分割 |

### 已支持流程

```text
边缘端采集图片
  ↓
上传压缩包到服务端 data/incoming
  ↓
服务端 ingest 解压到 data/raw_collected/<batch_id>
  ↓
服务端 Web 标注器审核/标注
  ↓
确认审核完成并同步到 data/raw_detection / data/raw_obb / data/raw_segmentation
  ↓
根据 task.yaml 或 UI preset 生成任务配置
  ↓
DVC 主链训练、评估、导出 ONNX、转换 RKNN、注册模型
  ↓
push.sh 自动部署到 RK3588
  ↓
边缘端 Web 模型验证 / 生产模式推理
```

---

## 2. 项目结构

```text
pipeline/
├── configs/
│   ├── task.yaml                         # 唯一人工配置入口
│   ├── generated/task.generated.yaml      # 自动生成，不手改
│   └── presets/                          # 服务端 UI 使用的任务模板
├── core/                                 # 配置读取、路径、IO 工具
├── stages/                               # 统一 stage 入口，根据 task.type 分发
└── tasks/
    ├── detection/                        # 目标检测分支
    ├── classification/                   # 图像分类分支
    ├── obb/                              # 旋转框检测分支
    └── segmentation/                     # 实例分割分支

server/
├── workflow/
│   ├── control_panel_app.py              # 服务端 Web 控制台
│   └── accept_reviewed_detection.py      # 审核结果同步到训练数据目录
├── annotation/
│   ├── annotation_app.py                 # 服务端标注器
│   └── label_io.py                       # YOLO 标签读写
└── data_ingest/
    └── ingest_uploaded_package.py        # 接收边缘端上传包

edge/
├── collector/                            # 边缘端 Web 客户端
├── inference/
│   └── engine.py                         # RKNN 推理服务
├── deploy/
│   ├── push.sh                           # 自动部署脚本
│   ├── switch_model.sh                   # 板端模型切换
│   └── stop_inference.sh                 # 停止旧推理进程
└── runtime/
    ├── class_names.yaml                  # 当前任务类别配置，自动生成
    └── edge.env                          # 当前任务边缘运行配置，自动生成

data/
├── incoming/                             # 边缘端上传包落点
├── raw_collected/                        # 解压后的采集批次
├── raw_detection/                        # 检测训练原始数据
├── raw_classification/                   # 分类训练原始数据
├── raw_obb/                              # OBB 训练原始数据
├── raw_segmentation/                     # 分割训练原始数据
└── processed_*/                          # 各任务预处理后数据

models/
├── checkpoints_detection/
├── checkpoints_classification/
├── checkpoints_obb/
├── checkpoints_segmentation/
├── export_detection/
├── export_classification/
├── export_obb/
├── export_segmentation/
├── metrics_detection/
├── metrics_classification/
├── metrics_obb/
└── metrics_segmentation/
```

---

## 3. 环境与服务

### 启动基础服务

```bash
make up
```

常用页面：

```text
MLflow:     http://localhost:5000
MinIO:      http://localhost:9001
Grafana:    http://localhost:3000
服务端 UI:  http://localhost:8091
```

### 启动服务端控制台

```bash
make workflow-ui
```

等价于：

```bash
python -m uvicorn server.workflow.control_panel_app:app   --host 0.0.0.0   --port 8091   --reload
```

---

## 4. task.yaml 单一配置入口

`pipeline/configs/task.yaml` 是训练和部署的唯一人工配置入口。修改任务类型时，只需要修改：

```yaml
task:
  type: detection
```

或：

```yaml
task:
  type: classification
```

或：

```yaml
task:
  type: obb_detection
```

或：

```yaml
task:
  type: segmentation
```

然后重新生成配置：

```bash
make render-task
```

生成文件包括：

```text
pipeline/configs/generated/task.generated.yaml
edge/runtime/class_names.yaml
edge/runtime/edge.env
```

这些 generated 文件由系统自动生成，不建议手动修改。

---

## 5. 数据格式

### 5.1 Detection

```text
data/raw_detection/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
└── data.yaml
```

YOLO detection 标签格式：

```text
class_id cx cy w h
```

### 5.2 Classification

```text
data/raw_classification/
├── train/class_a/*.jpg
├── train/class_b/*.jpg
├── val/class_a/*.jpg
└── val/class_b/*.jpg
```

也支持：

```text
data/raw_classification/class_name/*.jpg
```

预处理阶段会自动划分 train / val。

### 5.3 OBB Detection

```text
data/raw_obb/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
└── data.yaml
```

YOLO OBB 标签格式：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

坐标为归一化坐标。

### 5.4 Segmentation

```text
data/raw_segmentation/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
└── data.yaml
```

YOLO segmentation 标签格式：

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

坐标为归一化多边形点坐标。

---

## 6. 服务端 UI 工作流

服务端控制台主要包含三类功能：

### 6.1 数据接收与解压

边缘端上传的压缩包进入：

```text
data/incoming/
```

服务端接收解压：

```bash
make ingest-collected
```

强制覆盖：

```bash
make ingest-collected-force
```

解压后数据位于：

```text
data/raw_collected/<batch_id>/
```

### 6.2 标注与审核

服务端标注器支持：

```text
detection       水平框标注
obb_detection   四点旋转框标注
segmentation    多边形分割标注
```

典型流程：

```text
1. 选择采集批次
2. 打开 VisionOps 标注器
3. 选择任务类型
4. 标注或审核自动预标注结果
5. 保存标签
6. 点击“确认审核完成”
```

确认审核完成后，数据会同步到对应训练目录：

```text
detection       -> data/raw_detection
obb_detection   -> data/raw_obb
segmentation    -> data/raw_segmentation
```

### 6.3 训练与模型状态

服务端 UI 可从数据目录读取类别，生成 `pipeline/configs/task.yaml`，并触发 DVC 训练流水线。

支持的训练任务：

```text
目标检测
图像分类
旋转框检测
实例分割
```

MLflow 实验名建议：

```text
visionops-detection
visionops-classification
visionops-obb
visionops-segmentation
```

注意：MLflow 实验不是 UI 手动创建的，而是在对应任务的 `train.py` 第一次执行 `mlflow.set_experiment(...)` 和 `mlflow.start_run(...)` 时自动创建。若 UI 提示未找到实验，通常说明该任务还未跑过训练，或对应 `train.py` 没有写入 MLflow。

---

## 7. 运行流水线

### Python 主链

```bash
make pipeline
```

等价于：

```bash
python -m pipeline.utils.render_task_config
python -m pipeline.stages.preprocess
python -m pipeline.stages.train
python -m pipeline.stages.evaluate
python -m pipeline.stages.export_onnx
python -m pipeline.stages.convert_rknn
python -m pipeline.stages.register_model
```

### DVC 主链

```bash
make pipeline-dvc
```

或直接：

```bash
dvc repro register_model
```

### 单独运行某一阶段

```bash
python -m pipeline.stages.preprocess
python -m pipeline.stages.train
python -m pipeline.stages.evaluate
python -m pipeline.stages.export_onnx
python -m pipeline.stages.convert_rknn
python -m pipeline.stages.register_model
```

---

## 8. 各任务输出目录

### Detection

```text
data/processed_detection/
models/checkpoints_detection/
models/export_detection/model.onnx
models/export_detection/model.rknn
models/metrics_detection/
```

### Classification

```text
data/processed_classification/
models/checkpoints_classification/
models/export_classification/model.onnx
models/export_classification/model.rknn
models/metrics_classification/
```

### OBB Detection

```text
data/processed_obb/
models/checkpoints_obb/
models/export_obb/model.onnx
models/export_obb/model.rknn
models/metrics_obb/
```

### Segmentation

```text
data/processed_segmentation/
models/checkpoints_segmentation/
models/export_segmentation/model.onnx
models/export_segmentation/model.rknn
models/metrics_segmentation/
```

---

## 9. RKNN 转换说明

RKNN 转换阶段支持自动切换到独立 RKNN Python 环境，例如：

```yaml
rknn:
  python_exec: /home/pc/anaconda3/envs/rknn311/bin/python
```

如果当前 Python 不是 RKNN 环境，`convert_rknn.py` 会自动重新调用 RKNN 环境执行转换。

转换阶段若出现：

```text
I Target is None, use simulator!
```

这是因为脚本在 PC 端调用：

```python
rknn.init_runtime(target=None)
```

检查 RKNN 输出 shape。它表示使用 RKNN Toolkit 的 PC 模拟器做 shape 验证，不代表生成了模拟模型，也不代表转换失败。

可通过配置关闭输出 shape 检查：

```yaml
rknn:
  check_output_shapes: false
```

---

## 10. 部署到 RK3588

### 常用命令

```bash
make deploy           # 只上传当前 task 模型
make deploy-code      # 上传当前 task 模型并同步 edge/ 代码
make deploy-code-only # 仅同步 edge/ 代码
```

`push.sh` 会根据 `edge/runtime/class_names.yaml` 自动判断任务，并选择对应模型：

```text
detection       -> models/export_detection/model.rknn
classification  -> models/export_classification/model.rknn
obb_detection   -> models/export_obb/model.rknn
segmentation    -> models/export_segmentation/model.rknn
```

### 设备配置

建议新增或维护：

```text
pipeline/configs/deploy.yaml
```

示例：

```yaml
edge_devices:
  - id: rk3588-001
    host: 192.168.1.200
    port: 22
    user: ubuntu
    deploy_path: /opt/visionops/models/
    service_name: visionops-inference
    health_url: http://localhost:8080/health
```

`push.sh --code` 会覆盖板端：

```text
/opt/visionops/edge/
```

只有确认本地 edge 代码是最新版时再使用。

---

## 11. 边缘端推理服务

RKNN 推理服务：

```text
edge/inference/engine.py
```

支持：

```text
detection
classification
obb_detection
segmentation
```

典型 systemd 启动参数由 `push.sh` 自动生成：

```bash
/opt/visionops/venv/bin/python /opt/visionops/edge/inference/engine.py   --model ${MODEL_PATH}   --task ${TASK}   --host 0.0.0.0   --port ${PORT}   --npu-core ${NPU_CORE}   --num-classes ${NUM_CLASSES}   --class-names-file ${CLASS_NAMES_FILE}   --metrics-port ${METRICS_PORT}   --conf-threshold ${CONF_THRESHOLD}   --nms-threshold ${NMS_THRESHOLD}   --topk ${TOPK}   --warmup-runs ${WARMUP_RUNS}
```

Segmentation 任务额外支持：

```bash
--mask-threshold ${MASK_THRESHOLD}
```

---

## 12. 边缘端测试

### 健康检查

在 RK3588 上：

```bash
curl -s http://localhost:8082/health | python3 -m json.tool
```

期望看到：

```json
{
  "status": "ok",
  "task": "segmentation",
  "simulate_mode": false
}
```

### 单图推理

```bash
curl -s -X POST   -F "file=@/home/ubuntu/Desktop/test.jpg"   http://localhost:8082/infer | python3 -m json.tool
```

Detection / OBB 返回 `bbox` 或 `obb.points`。

Segmentation 返回：

```json
{
  "task": "segmentation",
  "predictions": [
    {
      "class_id": 0,
      "class_name": "object",
      "confidence": 0.9,
      "bbox": [0, 0, 100, 100],
      "mask": {
        "area": 1234,
        "segments": [[[10, 10], [20, 10], [20, 20]]],
        "polygon": [[10, 10], [20, 10], [20, 20]]
      }
    }
  ]
}
```

---

## 13. 边缘端 Web 客户端

边缘端 Web 客户端位于：

```text
edge/collector/
```

当前支持：

```text
模型校验
图像采集
采集打包上传
模型验证
生产模式推理
```

模型验证支持：

```text
classification top-k 显示
detection bbox 显示
obb rotated box / polygon 显示
segmentation mask polygon 显示
```

Segmentation 可视化默认显示：

```text
半透明 mask 区域
mask 轮廓
类别名与置信度
```

---

## 14. 常用 Make 命令

```bash
make up                         # 启动 Docker Compose 服务
make down                       # 停止服务
make restart                    # 重启服务
make logs SERVICE=mlflow        # 查看服务日志

make init                       # 初始化目录/DVC/MinIO
make render-task                # 根据 task.yaml 生成运行配置
make pipeline                   # Python 主链
make pipeline-dvc               # DVC 主链
make pipeline-force             # 强制重跑 DVC 主链

make ingest-collected           # 接收并解压上传包
make ingest-collected-force     # 覆盖式接收并解压上传包
make workflow-ui                # 启动服务端控制台

make deploy                     # 部署当前 task 模型
make deploy-code                # 部署模型并同步 edge 代码
make deploy-code-only           # 仅同步 edge 代码
```

---

## 15. 常见问题

### 15.1 MLflow 页面没有 OBB 或 Segmentation 实验

MLflow 实验由训练脚本第一次运行时自动创建。确认对应任务的 `train.py` 中有：

```python
mlflow.set_tracking_uri(...)
mlflow.set_experiment(...)
mlflow.start_run(...)
```

然后重新运行：

```bash
dvc repro train
```

检查实验：

```bash
python - <<'PY'
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()
for exp in client.search_experiments():
    print(exp.experiment_id, exp.name)
PY
```

### 15.2 Segmentation 量化模型没有预测

如果 `engine.py` 返回：

```json
{
  "message": "no predictions above confidence threshold",
  "debug": {
    "score_max": 0.0
  }
}
```

通常是 INT8 量化导致 score 分支异常。优先检查：

```text
1. export.mode 是否为 rockchip
2. 校准图片数量是否足够
3. rknn_report.json 中输出 shape 和数值范围
4. 非量化 RKNN 是否正常
```

Segmentation 任务建议使用更多真实场景图片做量化校准。

### 15.3 Rockchip segmentation 输出结构变化

Rockchip 导出的 YOLOv8-seg 常见输出：

```text
box / cls / sum / mask_coeff 三个尺度
+ proto
```

例如：

```text
[1, 64, 80, 80]
[1, 80, 80, 80]
[1, 1, 80, 80]
[1, 32, 80, 80]
...
[1, 32, 160, 160]
```

`engine.py` 需要使用 Rockchip segmentation 多输出后处理，而不能按标准 `[1, 116, 8400] + proto` 解析。

### 15.4 健康检查通过但 infer 报错

健康检查只说明模型加载和服务启动成功，不代表后处理匹配当前模型输出。排查：

```bash
sudo journalctl -u visionops-inference -n 200 --no-pager
```

重点看：

```text
[DEBUG] output[i] shape=...
Traceback
ValueError
```

---

## 16. 当前推荐开发顺序

```text
1. 修改 task.yaml
2. make render-task
3. 检查 generated/task.generated.yaml、edge/runtime/class_names.yaml、edge/runtime/edge.env
4. dvc repro register_model
5. 检查 MLflow 实验和 models/metrics_*/registry_result.json
6. make deploy-code
7. 边缘端 /health 检查
8. 边缘端 /infer 单图验证
9. 边缘端 Web 客户端验证
10. 生产模式测试
```