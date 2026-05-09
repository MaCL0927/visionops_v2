# VisionOps edge C++ v0.3_new

本补丁是在不破坏现有 Python/Web 边缘端的前提下，新增一个旁路 C++ RKNN 推理服务：

- 服务名：`visionops-inference-cpp.service`
- 默认端口：`18080`
- 默认模型：`/opt/visionops/models/current.rknn`
- 默认类别文件：`/opt/visionops/edge/runtime/class_names.yaml`
- 默认取流后端：`opencv`

## 文件位置

```text
edge/
├── deploy/
│   ├── deploy_cpp.sh
│   ├── visionops-inference-cpp.service
│   └── README_v0.3_new.md
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

带 RTSP 自动启动：

```bash
CAMERA_SOURCE='rtsp://admin:密码@192.168.2.64:554/Streaming/channels/101' \
STREAM_AUTO_START=true \
NUM_CLASSES=80 \
bash edge/deploy/deploy_cpp.sh --host 192.168.1.200
```

USB 相机测试可以使用 OpenCV 相机编号：

```bash
CAMERA_SOURCE='0' STREAM_AUTO_START=true bash edge/deploy/deploy_cpp.sh --host 192.168.1.200
```

## 验证

```bash
ssh ubuntu@192.168.1.200
curl http://127.0.0.1:18080/health
curl http://127.0.0.1:18080/stats
curl -X POST http://127.0.0.1:18080/stream/start
curl http://127.0.0.1:18080/stream/status
curl http://127.0.0.1:18080/stream/latest_result
curl http://127.0.0.1:18080/stream/snapshot.jpg -o /tmp/snapshot.jpg
```

单图推理：

```bash
curl -X POST http://127.0.0.1:18080/infer \
  -F image=@/tmp/test.jpg
```

## 设计说明

v0.3_new 只把 C++ 主服务、OpenCV/GStreamer 取流封装、RKNN 推理、systemd 和部署链路补齐。MPP 硬解码不作为主线依赖；`gst-mpp` 只是可选后端，默认仍使用更稳的 `opencv`。后续真正进入性能优化时，再把重点放到 RGA 预处理和后处理优化上。
