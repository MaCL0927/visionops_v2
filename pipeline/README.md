# Pipeline 结构说明

当前 pipeline 已收敛为“单一配置 + 单一 DVC 主链 + 任务实现分发”的结构。

## 入口

- 主配置：`pipeline/configs/task.yaml`
- 派生配置：`pipeline/configs/generated/task.generated.yaml`
- DVC 主链：`render_task_config -> preprocess -> train -> evaluate -> export_onnx -> convert_rknn -> register_model`

## 目录职责

- `pipeline/stages/`：只保留 DVC 入口。每个文件只根据 `task.type` 分发到对应任务实现，不再放具体训练/转换逻辑。
- `pipeline/tasks/classification/`：分类任务的 6 个阶段实现。
- `pipeline/tasks/detection/`：检测任务的 6 个阶段实现。
- `pipeline/configs/presets/`：可选模板，不参与主链运行。

## 配置原则

运行时只维护 `pipeline/configs/task.yaml`。`render_task_config.py` 会生成一个统一派生文件：

```text
pipeline/configs/generated/task.generated.yaml
```

各阶段通过 `pipeline.core.config.load_stage_config(<stage>)` 读取自己的配置段，不再读取 `detection_*.generated.yaml` 或 `classification_*.generated.yaml`。
