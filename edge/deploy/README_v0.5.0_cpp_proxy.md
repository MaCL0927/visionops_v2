# VisionOps v0.5.0：Web/Collector 接入 C++ 推理服务代理层

## 目标

v0.5.0 只做“代理层”：

```text
Web 前端
  ↓
Python Collector /api/cpp/*
  ↓
C++ inference service http://127.0.0.1:18080
  ↓
RTSP / RGA / RKNN / latest_result / snapshot
```

这一步不修改旧的 Python `engine.py`、`pipeline_engine.py`、`validation_infer.py`、`camera_service` 和前端页面。

## 新增文件

```text
edge/collector/backend/services/cpp_inference_client.py
edge/collector/backend/routers/cpp_inference.py
tools/apply_v0_5_0_cpp_proxy.py
edge/deploy/README_v0.5.0_cpp_proxy.md
```

## 自动注册 router 和配置

在仓库根目录执行：

```bash
python tools/apply_v0_5_0_cpp_proxy.py
```

脚本会自动修改：

```text
edge/collector/backend/main.py
edge/collector/backend/config.py
```

增加：

```python
from backend.routers.cpp_inference import router as cpp_inference_router
app.include_router(cpp_inference_router)
```

并在 config.py 中新增：

```python
CPP_INFERENCE_ENABLED
CPP_INFERENCE_URL
CPP_INFERENCE_TIMEOUT_SEC
CPP_INFERENCE_IMAGE_TIMEOUT_SEC
```

## 环境变量

默认 C++ 服务地址：

```bash
VISIONOPS_CPP_SERVICE_URL=http://127.0.0.1:18080
```

可选：

```bash
VISIONOPS_CPP_INFERENCE_ENABLED=1
VISIONOPS_CPP_TIMEOUT_SEC=5
VISIONOPS_CPP_IMAGE_TIMEOUT_SEC=15
```

## 新增 Collector API

```text
GET  /api/cpp/proxy_info
GET  /api/cpp/health
GET  /api/cpp/stats

POST /api/cpp/stream/start
POST /api/cpp/stream/stop
GET  /api/cpp/stream/status
GET  /api/cpp/stream/latest_result

GET  /api/cpp/stream/snapshot.jpg
GET  /api/cpp/stream/annotated.jpg
```

## 验收命令

先确认 C++ 服务自身可用：

```bash
curl -s http://127.0.0.1:18080/health | python3 -m json.tool
```

重启 Collector 后，通过 Collector 代理测试：

```bash
COLLECTOR_PORT=8000

curl -s http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/proxy_info | python3 -m json.tool
curl -s http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/health | python3 -m json.tool

curl -X POST http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/stream/start
sleep 5

curl -s http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/stream/status | python3 -m json.tool
curl -s http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/stream/latest_result | python3 -m json.tool

curl http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/stream/snapshot.jpg -o /tmp/proxy_snapshot.jpg
file /tmp/proxy_snapshot.jpg
```

如果需要调试带框图：

```bash
curl http://127.0.0.1:${COLLECTOR_PORT}/api/cpp/stream/annotated.jpg -o /tmp/proxy_annotated.jpg
file /tmp/proxy_annotated.jpg
```

## 注意

v0.5.0 不让 Python 参与每帧图像处理。Collector 只是代理控制接口和低频图片接口。

后续前端应优先使用：

```text
/api/cpp/stream/status         JSON，高频 1s 一次
/api/cpp/stream/latest_result  JSON，0.5~1s 一次
/api/cpp/stream/snapshot.jpg   图片，低频或手动
```

不要高频请求 `/api/cpp/stream/annotated.jpg`。
