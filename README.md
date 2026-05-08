# VisionOps

VisionOps 是一个面向 RK3588 边缘设备的端到端视觉 AI 软件平台，覆盖 **数据采集 → 服务端接收 → 标注与审核 → 训练流水线 → ONNX/RKNN 导出 → 边缘端部署 → 模型验证 → 生产推理 → 结果上传** 的完整闭环。

当前系统主线已经从最早的单一 detection/segmentation 分支，扩展为：

```text
task.yaml 单一真源
+ 统一 pipeline stage 分发
+ detection / classification / obb_detection / segmentation 多任务训练分支
+ ROI Classification 双模型推理分支
+ MLflow 实验与模型状态追踪
+ RK3588 RKNN 边缘部署
+ 服务端 Web 控制台
+ 服务端标注器 / ROI 分类数据制作
+ 边缘端 Web 客户端
+ 生产模式 Gateway 检测消息上传
+ 边缘端设置 / 状态 / 诊断 / NTP 同步
```

---

## 1. 当前能力概览

### 1.1 已支持任务

| 任务类型 | task.type / runtime task | 模型类型 | 主要用途 |
|---|---|---|---|
| 目标检测 | `detection` | YOLOv8 Detect | 水平框检测 |
| 图像分类 | `classification` | MobileNet/分类模型 | 单图或 ROI 类别判断 |
| 旋转框检测 | `obb_detection` | YOLOv8 OBB | 旋转目标、倾斜目标检测 |
| 实例分割 | `segmentation` | YOLOv8 Seg | 像素级目标区域分割 |
| ROI 分类双模型 | `roi_classification` | detection + classification | 先检测去背景，再对目标 ROI 分类 |

> `roi_classification` 当前不是单独训练一个新 pipeline，而是复用已有 detection 和 classification 两条训练流水线。它的核心是边缘端运行时和部署形式：一个双模型 bundle 中同时包含 detector 和 classifier。

### 1.2 已支持流程

```text
边缘端采集图片
  ↓
上传压缩包到服务端 data/incoming
  ↓
服务端 ingest 解压到 data/raw_collected/<batch_id>
  ↓
服务端 Web 标注器审核/标注
  ↓
同步到 detection / obb / segmentation / classification 原始训练目录
  ↓
根据 task.yaml 或 UI preset 生成任务配置
  ↓
DVC/MLflow 主链训练、评估、导出 ONNX、转换 RKNN、注册模型
  ↓
push.sh 自动部署到 RK3588
  ↓
边缘端 Web 模型验证 / 实时检测 / 生产模式推理
  ↓
生产结果可上传到 Gateway
```

### 1.3 最近新增重点

```text
1. Gateway 检测消息上传
   - 生产模式支持将实时检测结果推送给 Gateway
   - 支持记录 web / engine / gateway / client 端到端耗时字段

2. 设置界面增强
   - 支持采集数据目录生效
   - 支持相机 RTSP 参数、预览参数
   - 支持算法阈值、TopK、推理间隔
   - 支持视觉盒子状态、诊断信息、服务重启
   - 支持 NTP 时间同步
   - 后续预留 U 盘导入标定文件、双网口配置等能力

3. ROI Classification 双模型检测分支
   - 服务端标注器新增 ROI 分类数据制作
   - 检测模型先定位目标并裁剪 ROI
   - 人工将 ROI 分入多类别分类目录
   - 支持精细 ROI：每个检测类别保存一套相对 ROI 规则
   - 部署时生成双模型 bundle
   - 边缘端 pipeline_engine.py 支持 detection → ROI crop → classification
   - 边缘 Web 模型列表中将双模型 bundle 显示成一个模型
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
│   ├── annotation_app.py                 # 服务端标注器 + ROI 分类数据制作
│   └── label_io.py                       # YOLO 标签读写
└── data_ingest/
    └── ingest_uploaded_package.py        # 接收边缘端上传包

edge/
├── collector/                            # 边缘端 Web 客户端
│   ├── backend/                          # FastAPI 后端
│   └── static/                           # 前端页面与 JS/CSS
├── inference/
│   ├── engine.py                         # 单模型 RKNN 推理服务
│   └── pipeline_engine.py                # ROI Classification 双模型推理服务
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
├── roi_classification_sessions/          # ROI 分类数据制作 session 与 manifest
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

### 3.1 启动基础服务

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

### 3.2 启动服务端控制台

```bash
make workflow-ui
```

等价于：

```bash
python -m uvicorn server.workflow.control_panel_app:app \
  --host 0.0.0.0 \
  --port 8091 \
  --reload
```

### 3.3 边缘端 Web 客户端

边缘端 Web 客户端位于：

```text
edge/collector/
```

它负责：

```text
模型校验
图像采集
采集打包上传
模型验证
实时检测
生产模式推理
设置管理
状态诊断
Gateway 消息上传
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

> ROI Classification 当前不通过 `task.yaml` 训练一个新任务。它通过已有 detection / classification 训练产物组合部署。

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

支持两种输入形式。

已经划分 train / val：

```text
data/raw_classification/
├── train/class_a/*.jpg
├── train/class_b/*.jpg
├── val/class_a/*.jpg
└── val/class_b/*.jpg
```

未划分目录：

```text
data/raw_classification/
├── ok/*.jpg
├── ng/*.jpg
└── other_class/*.jpg
```

预处理阶段会自动划分 train / val。  
ROI Classification 数据制作功能最终也会把样本写入：

```text
data/raw_classification/<class_name>/*.jpg
```

分类类别不限制为 `ok/ng`，可以在服务端 ROI 分类数据制作窗口中新增类别。

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

### 5.5 ROI Classification Session

ROI 分类数据制作会生成：

```text
data/roi_classification_sessions/current/
├── manifest.json
├── candidates/
└── previews/
```

`manifest.json` 记录：

```text
检测模型
检测类别
候选 ROI
人工分类结果
padding_ratio
精细 ROI 规则 roi_policy
```

精细 ROI 采用“相对于检测框 + padding 后 base ROI 的比例框”：

```json
{
  "roi_policy": {
    "mode": "class_relative_box",
    "coordinate": "relative_to_padded_detection_box",
    "by_detector_class": {
      "0:tube": {
        "enabled": true,
        "mode": "relative_box",
        "padding_ratio": 0.0,
        "relative_box": {
          "x1": 0.0,
          "y1": 0.532364,
          "x2": 1.0,
          "y2": 0.793918
        }
      }
    }
  }
}
```

同一个检测类别只保存一套精细 ROI 规则；不同检测类别可以有不同 ROI 规则。

---

## 6. 服务端 UI 工作流

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

### 6.3 ROI 分类数据制作

用于解决这类场景：

```text
单纯分类：背景太多，影响判断
单纯检测：目标外观差异很小，检测模型难以区分
双阶段方案：检测模型去背景 + ROI 小范围分类
```

流程：

```text
1. 先完成 detection 模型训练，得到 models/checkpoints_detection/best.pt
2. 在 /annotator 中打开“ROI 分类数据制作”
3. 选择检测模型
4. 生成 ROI 候选
5. 人工将候选 ROI 分到分类类别
6. 支持新增类别
7. 数据写入 data/raw_classification/<class_name>/
8. 运行 classification pipeline 得到分类模型
```

当前第一版默认使用：

```text
检测模型 bbox + padding
```

作为分类 ROI。

优化版支持精细 ROI：

```text
勾选“启用精细 ROI”
  ↓
在 ROI 图上拖动橙色裁剪框
  ↓
点击“确认精细 ROI”
  ↓
保存该检测类别的一套 relative_box 到 manifest
  ↓
自动重新裁剪该检测类别下已分类样本
```

### 6.4 训练与模型状态

服务端 UI 可从数据目录读取类别，生成 `pipeline/configs/task.yaml`，并触发 DVC 训练流水线。

支持训练任务：

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

注意：MLflow 实验不是 UI 手动创建的，而是在对应任务的 `train.py` 第一次执行 `mlflow.set_experiment(...)` 和 `mlflow.start_run(...)` 时自动创建。

---

## 7. 运行流水线

### 7.1 Python 主链

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

### 7.2 DVC 主链

```bash
make pipeline-dvc
```

或直接：

```bash
dvc repro register_model
```

### 7.3 单独运行某一阶段

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

### ROI Classification

ROI Classification 不产生单独的 `models/export_roi_classification/`。它复用：

```text
models/export_detection/model.rknn
models/export_classification/model.rknn
models/metrics_detection/eval_metrics.json
models/metrics_classification/eval_metrics.json
data/roi_classification_sessions/current/manifest.json
```

部署后边缘端目录为：

```text
/opt/visionops/models/<bundle_name>/
├── detector.rknn
├── detector.yaml
├── classifier.rknn
├── classifier.yaml
└── pipeline.yaml
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

### 10.1 单模型部署

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

### 10.2 ROI Classification 双模型部署

命令：

```bash
bash edge/deploy/push.sh --roi-classification
```

如果边缘端代码也需要同步：

```bash
bash edge/deploy/push.sh --roi-classification --code
```

部署逻辑：

```text
读取 models/export_detection/model.rknn
读取 models/export_classification/model.rknn
读取 models/metrics_detection/eval_metrics.json
读取 models/metrics_classification/eval_metrics.json
读取 data/model_context/manifest.json
读取 data/roi_classification_sessions/current/manifest.json
  ↓
生成 detector.yaml / classifier.yaml / pipeline.yaml
  ↓
上传到 /opt/visionops/models/<bundle_name>/
  ↓
更新 /opt/visionops/.env
  ↓
将 visionops-inference.service 切换为 pipeline_engine.py
```

`pipeline.yaml` 会包含：

```yaml
pipeline_type: roi_classification
task: roi_classification

stage1:
  task: detection
  model_path: detector.rknn

stage2:
  task: classification
  model_path: classifier.rknn

roi:
  mode: class_relative_box
  by_detector_class:
    "0:tube":
      enabled: true
      mode: relative_box
      relative_box:
        x1: 0.0
        y1: 0.532364
        x2: 1.0
        y2: 0.793918
```

### 10.3 `--code` 权限注意

`push.sh --code` 会同步板端：

```text
/opt/visionops/edge/
```

板端推理服务可能以 root 运行，生成 root-owned：

```text
__pycache__/*.pyc
```

为避免 `rsync --delete` 权限问题，部署脚本会先清理远端 `__pycache__` 和 `*.pyc`，并在 rsync 时排除这些缓存文件。

---

## 11. 边缘端推理服务

### 11.1 单模型推理服务

文件：

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

典型 systemd 启动参数：

```bash
/opt/visionops/venv/bin/python /opt/visionops/edge/inference/engine.py \
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
```

### 11.2 ROI Classification 双模型推理服务

文件：

```text
edge/inference/pipeline_engine.py
```

典型启动：

```bash
/opt/visionops/venv/bin/python /opt/visionops/edge/inference/pipeline_engine.py \
  --pipeline-config /opt/visionops/models/<bundle_name>/pipeline.yaml \
  --host 0.0.0.0 \
  --port 8082 \
  --metrics-port 9091
```

推理流程：

```text
原图
  ↓
detector.rknn 检测目标 bbox
  ↓
根据 ROI policy 裁剪 final ROI
  ↓
classifier.rknn 对 final ROI 分类
  ↓
返回 detector bbox + final ROI + classification result
```

支持 ROI 模式：

```text
full_box             检测框 + padding 后直接分类
relative_box         所有检测类别共用一个相对裁剪框
class_relative_box   每个检测类别单独配置一套相对裁剪框
```

---

## 12. 边缘端测试

### 12.1 健康检查

```bash
curl -s http://localhost:8082/health | python3 -m json.tool
```

单模型 segmentation 示例：

```json
{
  "status": "ok",
  "task": "segmentation",
  "simulate_mode": false
}
```

ROI Classification 示例：

```json
{
  "status": "ok",
  "task": "roi_classification",
  "pipeline_name": "rk3588-001_zt_roi_cls_xxx",
  "roi": {
    "mode": "class_relative_box",
    "by_detector_class_keys": ["0:tube"]
  }
}
```

### 12.2 单图推理

```bash
curl -s -X POST \
  -F "file=@/home/ubuntu/Desktop/test.jpg" \
  http://localhost:8082/infer | python3 -m json.tool
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

ROI Classification 返回：

```json
{
  "task": "roi_classification",
  "final_label": "ng",
  "final_confidence": 0.99,
  "predictions": [
    {
      "class_name": "ng",
      "confidence": 0.99,
      "bbox": [951.29, 551.94, 1124.04, 876.42],
      "detector": {
        "class_name": "tube",
        "confidence": 0.95,
        "bbox": [951.29, 551.94, 1124.04, 876.42]
      },
      "roi": {
        "mode": "relative_box",
        "pipeline_mode": "class_relative_box",
        "base_bbox": [951.0, 552.0, 1124.0, 876.0],
        "bbox": [951.0, 725.0, 1124.0, 810.0],
        "relative_box": {
          "x1": 0.0,
          "y1": 0.532364,
          "x2": 1.0,
          "y2": 0.793918
        }
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
选图检测
拍照检测
实时检测
生产模式推理
Gateway 消息上传
设置管理
状态诊断
```

### 13.1 模型选择

模型列表支持混合显示：

```text
普通 .rknn 单模型
ROI Classification 双模型 bundle
```

ROI 双模型在界面中显示为一个模型：

```text
rk3588-001_zt_roi_cls_xxx
最新模型 · ROI分类双模型 · 双模型推理
```

选择普通模型时，服务切换到：

```text
engine.py
```

选择 ROI 双模型时，服务切换到：

```text
pipeline_engine.py
```

切换时会停止旧的 8082 推理进程，避免单模型和双模型抢占端口。

### 13.2 模型验证可视化

支持：

```text
classification top-k 显示
detection bbox 显示
obb rotated box / polygon 显示
segmentation mask polygon 显示
roi_classification detector bbox + final ROI + 分类结果显示
```

ROI Classification 可视化规则：

```text
绿色框：detector bbox
橙色框：final ROI
文字：最终分类结果、分类置信度、检测置信度、ROI mode
```

### 13.3 实时检测画面稳定性

实时检测和生产模式画面刷新使用：

```text
稳定 img/canvas DOM
新图片预加载完成后再替换旧图
```

避免每帧重建 DOM 导致画面黑屏闪烁。

实时图片使用唯一文件名：

```text
realtime_<timestamp>.jpg
```

避免前端读取 `realtime_latest.jpg` 时被下一帧覆盖，导致：

```text
RuntimeError: Response content shorter than Content-Length
```

临时图清理策略：

```text
每个目录最多每 5 秒清理一次
保留最近 30 张实时帧
超过 60 秒的实时帧会被删除
realtime_latest.* 仅作为兼容副本保留
```

### 13.4 采集取图优化

取图按钮应避免重复提交：

```text
captureBusy = true 时禁止再次触发 /api/capture
```

取图成功后优先只插入新图到列表前端，避免每次都刷新整个图片列表并重新加载大量缩略图。

---

## 14. Gateway 检测消息上传

生产模式支持将检测结果上传到 Gateway。典型输出包括：

```text
source
frame_id
count
predictions
client_receive_ms
total_web_start_to_client_recv_ms
web_capture_ms
web_save_ms
web_infer_roundtrip_ms
engine_latency_ms
web_infer_extra_over_engine_ms
web_gateway_post_roundtrip_ms
web_to_gateway_arrive_ms
gateway_total_ms
gateway_to_client_recv_ms
```

用于排查：

```text
相机取流耗时
Web 端保存耗时
推理服务 roundtrip
engine 内部推理耗时
Gateway POST 耗时
端到端客户端接收耗时
```

如果暂时不接 Gateway，可以让生产模式只在本地显示结果；接入 Gateway 后再打开上传配置。

---

## 15. 设置界面

边缘端设置功能按版本逐步扩展，当前已覆盖：

```text
v2.0 设置保存 / 回显框架
v2.1 相机 RTSP 参数 + 预览参数生效
v2.2 算法阈值 + TopK + 推理间隔生效
v2.3 视觉盒子状态 / 诊断包 / 服务重启
v2.3.2 采集数据目录生效
v2.3.3 NTP 时间同步
```

后续预留：

```text
v2.4 U 盘导入标定文件
v2.5 双网口配置
```

常见配置项：

```text
VISIONOPS_CAMERA_SOURCE
VISIONOPS_CAMERA_STREAM_FPS
VISIONOPS_CAMERA_PREVIEW_WIDTH
VISIONOPS_CAMERA_JPEG_QUALITY
推理间隔
conf_threshold
nms_threshold
mask_threshold
topk
采集数据目录
上传目标目录
NTP 服务器
```

---

## 16. 常用 Make 命令

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

ROI Classification：

```bash
bash edge/deploy/push.sh --roi-classification
bash edge/deploy/push.sh --roi-classification --code
```

---

## 17. 常见问题

### 17.1 MLflow 页面没有 OBB 或 Segmentation 实验

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

### 17.2 Segmentation 量化模型没有预测

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

### 17.3 Rockchip segmentation 输出结构变化

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

### 17.4 健康检查通过但 infer 报错

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

### 17.5 ROI 双模型影响普通单模型切换

如果选择 segmentation / detection 等普通模型时日志仍显示：

```text
pipeline_engine.py
```

说明 systemd 服务还停留在 ROI 双模型模式。排查：

```bash
sudo systemctl cat visionops-inference | grep -E "ExecStart|engine.py|pipeline_engine.py"
sudo grep -E "TASK|MODEL_PATH|CLASS_NAMES_FILE|PIPELINE_CONFIG" /opt/visionops/.env
curl -s http://localhost:8082/health | python3 -m json.tool
```

普通单模型应该看到：

```text
ExecStart=.../engine.py
TASK=segmentation / detection / classification / obb_detection
MODEL_PATH=...
CLASS_NAMES_FILE=...
```

ROI 双模型才应该看到：

```text
ExecStart=.../pipeline_engine.py
TASK=roi_classification
PIPELINE_CONFIG=/opt/visionops/models/<bundle>/pipeline.yaml
```

### 17.6 实时图片接口报 Content-Length 错误

报错：

```text
RuntimeError: Response content shorter than Content-Length
```

原因通常是前端正在读取固定图片文件，后端下一帧又覆盖了同一个文件。

解决策略：

```text
1. 实时帧使用唯一文件名 realtime_<timestamp>.jpg
2. 返回图片时一次性 read_bytes 再 Response
3. 定期清理旧实时帧
4. 前端新图预加载完成后再替换旧图
```

### 17.7 `push.sh --code` 删除 `__pycache__` 权限不足

报错：

```text
rsync: delete_file: unlink(inference/__pycache__/engine.cpython-38.pyc) failed: Permission denied
```

原因是推理服务以 root 运行生成了 root-owned pyc。部署脚本应：

```text
1. sudo 删除远端 __pycache__ 和 *.pyc
2. rsync 时排除 __pycache__/ 和 *.pyc
```

---

## 18. 当前推荐开发顺序

### 18.1 普通任务

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

### 18.2 ROI Classification

```text
1. 采集目标图片
2. 训练 detection 模型
3. 在 /annotator 中生成 ROI 分类候选
4. 人工分类 ROI 到 data/raw_classification/<class_name>/
5. 如需要，启用精细 ROI 并确认 ROI policy
6. 训练 classification 模型
7. bash edge/deploy/push.sh --roi-classification --code
8. 边缘端 /health 检查 task=roi_classification
9. 边缘端 Web 选中双模型 bundle
10. 测试选图 / 拍照 / 实时检测 / 生产模式
```

### 18.3 生产模式 + Gateway

```text
1. 先确认本地生产模式检测结果稳定
2. 再打开 Gateway 上传
3. 查看检测结果 count / class / bbox / ROI / latency
4. 对比 engine_latency_ms 和 total_web_start_to_client_recv_ms
5. 根据实际需求调整推理间隔、图片质量、预览分辨率
```