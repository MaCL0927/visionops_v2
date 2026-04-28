# VisionOps Edge Collector UI v5.0

本版本基于 v4.8 继续修改，重点是“打包上传”流程：

1. 上传确认弹窗去掉“数据用途”；
2. “联系人（选填）”改为“联系方式（选填）”；
3. 点击“确认上传”后，先将当前固定采集目录打成 `tar.gz` 包；
4. 若配置了电脑端 SSH 目标，则自动通过 `scp` 上传到电脑：

```text
/home/pc/桌面/visionops_v2/data
```

同时保留 v4.7/v4.8 的 RTSP 单例读取、低帧率预览和删除同步逻辑。

## 启动方式

```bash
cd edge/collector
source venv/bin/activate
python app.py
```

访问：

```text
http://板子IP:8090/?v=v5_0
```

## 固定本地目录

默认固定采集目录为：

```text
edge/collector/data/local_dataset/
├── all_images/
├── positive/
├── negative/
└── upload_packages/
```

工人界面不出现数据集下拉框，也不提供新增数据集按钮。

## 采集按钮逻辑

- `取图`：保存到 `all_images/`
- `取正样本`：同时保存到 `all_images/` 和 `positive/`
- `取负样本`：同时保存到 `all_images/` 和 `negative/`

因此，`all_images/` 是完整采集底库，`positive/negative/` 是正负样本子集。

## 图片命名规则

```text
设备ID_用户ID_YYYYMMDD_HHMMSS_微秒.jpg
```

可通过环境变量修改：

```bash
export VISIONOPS_DEVICE_ID=rk3588-001
export VISIONOPS_USER_ID=operator-001
```

## RK3588 + 海康 RTSP 推荐启动

建议先使用子码流，降低 CPU 占用和延迟：

```bash
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|stimeout;5000000"
export VISIONOPS_CAMERA_SOURCE="rtsp://用户名:密码@摄像头IP:554/Streaming/Channels/102"
export VISIONOPS_CAMERA_STREAM_FPS=6
export VISIONOPS_CAMERA_PREVIEW_WIDTH=960
export VISIONOPS_CAMERA_JPEG_QUALITY=75
python app.py
```

说明：当前 OpenCV RTSP/MJPEG 方案在 RK3588 上仍可能有较高 CPU 占用，后续优化方向是进一步减少边缘客户端资源占用。

## 打包上传到电脑端

v5.0 的上传方向与原先 `push.sh` 相反：

```text
原 push.sh：电脑 -> 边缘板
v5.0 上传：边缘板 -> 电脑
```

因此电脑端需要开启 SSH 服务，并建议配置 RK3588 到电脑的免密登录。

### 1. 电脑端准备目录

在电脑上执行：

```bash
mkdir -p /home/pc/桌面/visionops_v2/data
```

确认电脑 SSH 服务可用：

```bash
sudo apt update
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
```

查看电脑 IP：

```bash
ip addr
```

### 2. RK3588 端配置上传目标

假设电脑 IP 是 `192.168.1.10`，电脑用户名是 `pc`：

```bash
export VISIONOPS_UPLOAD_HOST="192.168.1.10"
export VISIONOPS_UPLOAD_USER="pc"
export VISIONOPS_UPLOAD_PORT=22
export VISIONOPS_UPLOAD_TARGET_DIR="/home/pc/桌面/visionops_v2/data"
```

建议配置免密 SSH：

```bash
ssh-keygen -t ed25519 -C "visionops-edge-uploader"
ssh-copy-id -p 22 pc@192.168.1.10
```

测试：

```bash
ssh pc@192.168.1.10 "mkdir -p /home/pc/桌面/visionops_v2/data && echo ok"
```

### 3. 启动客户端

```bash
cd /opt/visionops/edge/collector
source venv/bin/activate
python app.py
```

点击确认上传后，会：

1. 在板端生成本地包：

```text
edge/collector/data/local_dataset/upload_packages/*.tar.gz
```

2. 如果 `VISIONOPS_UPLOAD_HOST` 已配置，则上传到：

```text
pc@电脑IP:/home/pc/桌面/visionops_v2/data/*.tar.gz
```

如果没有配置 `VISIONOPS_UPLOAD_HOST`，系统只会本地打包，不会报错中断。

## 上传包内容

```text
all_images/
positive/
negative/
collector_meta.jsonl    # 如果存在
manifest.json
```

`manifest.json` 包含：

```json
{
  "dataset": "local_dataset",
  "device_id": "rk3588-001",
  "customer_id": "CUST-001",
  "contact_info": "联系方式",
  "remark": "备注",
  "counts": {
    "all": 10,
    "positive": 4,
    "negative": 6
  }
}
```

## 删除同步规则

从“全部图片 / 正样本 / 负样本”任意入口删除一张图片，都会同步删除：

```text
all_images/同名图片
positive/同名图片
negative/同名图片
```

## v6.0 更新说明：模型验证页读取真实模型列表

本版本在 v5.0 打包上传功能基础上，新增模型验证页的真实模型读取能力：

- 默认读取 `/opt/visionops/models` 目录下的所有 `.rknn` 文件。
- 前端“模型验证”页左侧不再显示写死的演示模型，而是显示真实模型列表。
- `current.rknn` 会标记为“当前使用”。
- `backup_*.rknn` 会标记为“历史模型”。
- 文件名包含 `candidate` 的模型会标记为“待测试模型”。
- 支持点击“刷新模型”重新扫描模型目录。

本地调试时可以临时指定模型目录：

```bash
export VISIONOPS_MODELS_DIR=/tmp/visionops_models
mkdir -p "$VISIONOPS_MODELS_DIR"
touch "$VISIONOPS_MODELS_DIR/current.rknn"
touch "$VISIONOPS_MODELS_DIR/det_20260425_001.rknn"
python app.py
```

接口：

```text
GET  /api/models
POST /api/refresh_models
```


## v6.1 更新说明

本版在 v6.0 真实模型列表基础上，修复两个长内容场景的显示问题：

1. 采集标注栏的“确认上传”界面中，图片数量很多时，预览图不再被压缩得很小，而是在预览区域内部上下滚动。
2. 模型验证页中，`/opt/visionops/models` 下模型很多时，模型列表不再把整个页面拉长，而是在模型列表区域内部上下滚动。

本版主要修改 `static/styles.css`，并更新前端资源缓存版本号为 `v6_1`。

## v6.2 更新：分类模型单张图片验证

- 模型验证页默认选择 `current.rknn`，用于代表当前最新模型。
- `backup_时间.rknn` 会显示为历史模型，可手动选择进行对比测试。
- 右侧新增“最近采集图片”列表，默认读取当前固定采集目录的 `all_images/`。
- 点击图片后，点击“开始检测”，后端会使用选中模型启动独立验证推理服务，默认端口为 `8082`。
- 分类验证固定使用 `classification` 任务，并优先读取 `/opt/visionops/edge/runtime/class_names.yaml`。
- 生产推理服务 `8080` 不会被 v6.2 修改；v6.2 只使用独立的验证服务端口。

常用环境变量：

```bash
export VISIONOPS_MODELS_DIR=/opt/visionops/models
export VISIONOPS_CLASSIFICATION_CLASS_NAMES_FILE=/opt/visionops/edge/runtime/class_names.yaml
export VISIONOPS_VALIDATION_INFER_PORT=8082
export VISIONOPS_VALIDATION_ENGINE_PATH=/opt/visionops/edge/inference/engine.py
export VISIONOPS_CLASSIFICATION_INPUT_SIZE=224,224
```

## v6.3：拍照分类检测与验证页布局优化

- 模型验证页右侧布局调整：最近采集图片列表改为单列窄栏，测试结果图片预览区加大。
- 新增“拍照检测”按钮：使用当前选中模型，对摄像头最新画面进行分类验证。
- RTSP/后端摄像头模式下，后端直接读取 latest_frame；浏览器摄像头/模拟模式下，前端截图后上传。
- 拍照检测保存到当前数据集 all_images，并自动加入最近采集图片列表，便于后续复查。


## v6.3.1：模型验证页界面微调

- 最近采集图片列表改为更适合单列缩略图的宽度，避免文件名频繁换行。
- 检测结果文字框与 Top 结果改为左右排列，减少竖向占用。
- 检测结果图片预览区域获得更多高度，在完整显示图片的前提下尽可能放大。
- 按钮顺序调整为：选图检测 / 拍照检测 / 实时检测。
- 三个验证按钮默认统一为白底黑字；实时检测按钮支持点击后黑底白字，再次点击恢复。
- 实时检测本版仅完成界面状态设计，推理功能后续接入。

## v6.4：低频实时分类检测

- 模型验证页的“实时检测”按钮正式接入推理功能。
- 点击“实时检测”后，前端默认每 1 秒请求一次单帧分类；再次点击“停止实时”后停止。
- 仍使用当前选中的模型，并复用 v6.2 的独立验证推理服务，默认端口为 `8082`。
- RTSP/后端摄像头模式下，后端直接读取 `latest_frame`；浏览器摄像头/模拟模式下，前端上传当前截图。
- 实时检测图片只覆盖保存到 `data/<dataset>/validation_tmp/realtime_latest.jpg`，不会写入 `all_images/positive/negative`，避免污染采集数据集。
- 切换模型、刷新模型、切换到其他页面或进入生产模式时，会自动停止实时检测。

可选环境变量：

```bash
export VISIONOPS_VALIDATION_REALTIME_INTERVAL_MS=1000
```

## v6.4.1 更新说明：摄像头资源生命周期优化

- 采集标注页只有处于“拍照采集”子页面时才打开摄像头预览。
- 切换到“确认上传”、校验页、模型验证页、生产模式时，会自动关闭前端摄像头预览。
- RTSP/后端摄像头模式下，离开拍照采集页会调用 `/api/camera/stop` 停止后端读取线程，降低 RK3588 资源占用。
- 模型验证页的“拍照检测”和“实时检测”仍可按需临时打开摄像头；停止实时检测或离开页面后会自动释放摄像头资源。

## v6.5 更新说明：模型验证读取同名 YAML

v6.5 起，模型验证页不再依赖全局 `class_names.yaml`，而是读取 `/opt/visionops/models` 下的版本化模型对：

```text
xxx.rknn
xxx.yaml
```

其中 `xxx.yaml` 同时作为模型 meta 和推理类别配置，至少需要包含：

```yaml
task: classification        # 或 detection
input_size: [224, 224]      # detection 通常为 [640, 640]
num_classes: 2
class_names:
  - 未缠膜
  - 已缠膜
```

模型验证时会根据选中模型的同名 YAML 动态启动 8082 验证推理服务。分类模型显示 Top 结果；检测模型当前先显示目标数量和类别统计，检测框可视化留到后续版本增强。

推荐将客户端运行环境写入：

```text
/opt/visionops/edge/runtime/collector.env
```

本包提供了示例文件：

```text
edge/runtime/collector.env.example
```

开发调试可手动加载：

```bash
set -a
source /opt/visionops/edge/runtime/collector.env
set +a
cd /opt/visionops/edge/collector
python app.py
```

## v6.7 更新说明

本版在 v6.5 的同名 YAML 模型配置基础上，补充检测任务的选图验证能力：

1. 选择 `task: detection` 的模型后，可以对最近采集图片执行“选图检测”；
2. 后端继续根据 `xxx.rknn + xxx.yaml` 动态启动 8082 验证推理服务；
3. 前端会解析检测结果中的 `predictions[].bbox`，在原图上绘制检测框、类别名和置信度；
4. 结果区域显示目标总数、最高置信度和类别统计；
5. 拍照采集页面不做检测任务专门适配，保持原有分类采集界面不变。
