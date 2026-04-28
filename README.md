# VisionOps

端到端视觉 AI 软件平台。当前代码已统一为 **task.yaml 单一入口 + detection/classification 双分支 + RK3588 RKNN 边缘部署**。

## 当前结构

```text
pipeline/configs/task.yaml                 # 唯一人工配置入口
pipeline/configs/*.generated.yaml          # 自动生成，不提交、不手改
pipeline/core/                             # 配置读取与 IO 工具
pipeline/stages/                           # 统一入口，根据 task.type 分发
pipeline/tasks/detection/                  # 检测分支真实实现
pipeline/tasks/classification/             # 分类分支真实实现
edge/runtime/class_names.yaml              # 当前任务类别配置，自动生成
edge/runtime/edge.env                      # 当前任务边缘运行配置，自动生成
```

不再使用旧的 旧版拆分配置文件 作为主链配置。

## 切换任务

检测任务：

```yaml
task:
  type: detection
```

分类任务：

```yaml
task:
  type: classification
```

分类模板：

```text
pipeline/configs/presets/classification_task.example.yaml
```

## 运行流水线

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

DVC 主链保留：

```bash
make pipeline-dvc
```

## 数据格式

Detection:

```text
data/raw_detection/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
└── data.yaml
```

Classification:

```text
data/raw_classification/
├── train/class_a/*.jpg
├── train/class_b/*.jpg
├── val/class_a/*.jpg
└── val/class_b/*.jpg
```

也支持 `data/raw_classification/class_name/*.jpg`，预处理会自动划分 train/val。

## 输出目录

Detection:

```text
data/processed_detection/
models/checkpoints_detection/
models/export_detection/model.onnx
models/export_detection/model.rknn
models/metrics_detection/
```

Classification:

```text
data/processed_classification/
models/checkpoints_classification/
models/export_classification/model.onnx
models/export_classification/model.rknn
models/metrics_classification/
```

## 部署

```bash
make deploy           # 只上传当前 task 模型
make deploy-code      # 上传当前 task 模型并同步 edge/ 代码
make deploy-code-only # 仅同步 edge/ 代码
```

`push.sh` 会根据 `edge/runtime/class_names.yaml` 自动判断当前任务，并自动选择：

```text
models/export_detection/model.rknn
models/export_classification/model.rknn
```

建议新增 `pipeline/configs/deploy.yaml` 保存设备 SSH 信息：

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

注意：`push.sh --code` 会覆盖板端 `/opt/visionops/edge/`，只有确认本地 edge 代码是最新版时再使用。
