# VisionOps edge C++ v0.3.1

本版本是在当前 v0.3 已经跑通 C++ RKNN 实时推理链路的基础上，增加“检测可视化与验收”能力。目标是在进入 v0.4 RGA 优化前，先确认检测框、类别、置信度、中心点和原图坐标映射是否正确。

## 版本目标

- 保留独立 C++ 服务：`visionops-inference-cpp.service`
- 默认端口：`18080`
- 默认模型：`/opt/visionops/models/rk3588-001_CUST-000_det_20260427_105932.rknn`
- 默认类别配置：`/opt/visionops/models/rk3588-001_CUST-000_det_20260427_105932.yaml`
- 默认类别数：`80`
- 默认取流后端：`opencv`
- 新增带框可视化接口：`GET /stream/annotated.jpg`

## 文件位置

```text
edge/
├── deploy/
│   ├── deploy_cpp.sh
│   ├── visionops-inference-cpp.service
│   └── README_v0.3.1.md
└── inference_cpp/
    ├── CMakeLists.txt
    ├── include/visionops/stream_backend.hpp
    ├── scripts/start_visionops_inference_cpp.sh
    └── src/
        ├── main.cpp
        └── stream_backend.cpp
```

## 一键部署

在训练机/开发机仓库根目录执行：

```bash
bash edge/deploy/deploy_cpp.sh --host 192.168.1.200
```

如需显式指定 RTSP 并自动启动实时流：

```bash
CAMERA_SOURCE='rtsp://admin:Abcd123_@192.168.2.64:554/Streaming/channels/101' \
STREAM_AUTO_START=true \
NUM_CLASSES=80 \
bash edge/deploy/deploy_cpp.sh --host 192.168.1.200
```

USB 相机测试可以使用 OpenCV 相机编号：

```bash
CAMERA_SOURCE='0' STREAM_AUTO_START=true bash edge/deploy/deploy_cpp.sh --host 192.168.1.200
```

## 验收流程

```bash
ssh ubuntu@192.168.1.200

curl http://127.0.0.1:18080/health
curl -X POST http://127.0.0.1:18080/stream/stop
sleep 1
curl -X POST http://127.0.0.1:18080/stream/start
sleep 5

curl http://127.0.0.1:18080/stream/status
curl http://127.0.0.1:18080/stream/latest_result
curl http://127.0.0.1:18080/stream/snapshot.jpg -o /tmp/snapshot.jpg
curl http://127.0.0.1:18080/stream/annotated.jpg -o /tmp/annotated.jpg

file /tmp/snapshot.jpg
file /tmp/annotated.jpg
```

`/stream/status` 中应重点看：

```json
{
  "camera_frames": 1,
  "detect_frames": 1,
  "snapshot_available": true,
  "annotated_available": true,
  "last_error": ""
}
```

`/stream/latest_result` 中新增：

```json
{
  "image_width": 2688,
  "image_height": 1520,
  "input_width": 640,
  "input_height": 640,
  "letterbox": {
    "ratio": 0.238,
    "pad_x": 0.0,
    "pad_y": 139.0
  }
}
```

这些字段用于确认检测框已经从 640x640 letterbox 输入坐标正确映射回原始 RTSP 图像坐标。

## 新增接口

### 原始快照

```bash
curl http://127.0.0.1:18080/stream/snapshot.jpg -o /tmp/snapshot.jpg
```

返回最近一帧原始 BGR 图像编码后的 JPEG。

### 带检测框快照

```bash
curl http://127.0.0.1:18080/stream/annotated.jpg -o /tmp/annotated.jpg
```

返回最近一次推理帧的可视化结果，包含：

- bbox 矩形框
- class name
- confidence
- center point

如果刚启动时还没完成第一轮推理，会返回 JSON：

```json
{"status":"no_annotated_frame","message":"annotated frame is not ready; start stream and wait until annotated_available=true in /stream/status"}
```

## 单图推理

```bash
curl -X POST http://127.0.0.1:18080/infer \
  -F image=@/tmp/test.jpg
```

单图推理 JSON 同样会返回 `image_width / image_height / letterbox` 字段，用于排查坐标映射问题。

## 设计说明

v0.3.1 不做 RGA 优化、不改 RTSP 解码主线、不引入 MPP 依赖。它只做 C++ 检测链路的可视化验收，确保 v0.4 进行 RGA 预处理优化前有一个稳定、可对照的基线版本。
