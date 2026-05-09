# VisionOps v0.4.3：降低 CPU 占用诊断与快照解耦

本版本基于 v0.4.2.1，目标不是直接完成最终优化，而是定位 CPU 占用来源。

## 新增能力

- 保留 `--rga-mode off|resize_color|resize_only` 三种预处理实验模式。
- 新增 `--enable-snapshot true|false`，控制是否在实时流循环中缓存原图快照。
- 新增 `--enable-annotated true|false`，控制是否在实时流循环中绘制检测框图。
- `/stream/latest_result` 增加 `timing.stream_detail`：
  - `capture_read_ms`：OpenCV/GStreamer 读帧耗时，包含 RTSP 解码等待。
  - `snapshot_clone_ms`：缓存 snapshot 的 Mat clone 耗时。
  - `annotated_draw_ms`：画框、类别、中心点耗时。
  - `loop_total_ms`：本轮实时流循环总耗时。
- `/stream/status` 增加 `diagnostics`：
  - 最近一次 read/clone/draw/loop 耗时；
  - 最近一次 `/stream/snapshot.jpg` 和 `/stream/annotated.jpg` JPEG 编码耗时；
  - JPEG 请求次数。

## 推荐测试组合

### 1. 最小视觉开销：只取流 + 推理

```bash
sudo sed -i 's/^VISIONOPS_CPP_RGA_MODE=.*/VISIONOPS_CPP_RGA_MODE=resize_color/' /opt/visionops/edge/runtime/cpp.env
sudo sed -i 's/^VISIONOPS_CPP_ENABLE_SNAPSHOT=.*/VISIONOPS_CPP_ENABLE_SNAPSHOT=false/' /opt/visionops/edge/runtime/cpp.env || echo 'VISIONOPS_CPP_ENABLE_SNAPSHOT=false' | sudo tee -a /opt/visionops/edge/runtime/cpp.env
sudo sed -i 's/^VISIONOPS_CPP_ENABLE_ANNOTATED=.*/VISIONOPS_CPP_ENABLE_ANNOTATED=false/' /opt/visionops/edge/runtime/cpp.env || echo 'VISIONOPS_CPP_ENABLE_ANNOTATED=false' | sudo tee -a /opt/visionops/edge/runtime/cpp.env
sudo systemctl restart visionops-inference-cpp
```

### 2. 只打开 snapshot

```bash
sudo sed -i 's/^VISIONOPS_CPP_ENABLE_SNAPSHOT=.*/VISIONOPS_CPP_ENABLE_SNAPSHOT=true/' /opt/visionops/edge/runtime/cpp.env
sudo sed -i 's/^VISIONOPS_CPP_ENABLE_ANNOTATED=.*/VISIONOPS_CPP_ENABLE_ANNOTATED=false/' /opt/visionops/edge/runtime/cpp.env
sudo systemctl restart visionops-inference-cpp
```

### 3. snapshot + annotated 全开

```bash
sudo sed -i 's/^VISIONOPS_CPP_ENABLE_SNAPSHOT=.*/VISIONOPS_CPP_ENABLE_SNAPSHOT=true/' /opt/visionops/edge/runtime/cpp.env
sudo sed -i 's/^VISIONOPS_CPP_ENABLE_ANNOTATED=.*/VISIONOPS_CPP_ENABLE_ANNOTATED=true/' /opt/visionops/edge/runtime/cpp.env
sudo systemctl restart visionops-inference-cpp
```

## 记录命令

```bash
curl -X POST http://127.0.0.1:18080/stream/stop
sleep 1
curl -X POST http://127.0.0.1:18080/stream/start
sleep 8

curl -s http://127.0.0.1:18080/stream/status | python3 -m json.tool
curl -s http://127.0.0.1:18080/stream/latest_result | python3 -m json.tool

pid=$(pgrep -f visionops_inference_cpp | head -n 1)
pidstat -p "$pid" 1 30
```

## 判断逻辑

- 如果关闭 snapshot/annotated 后 CPU 明显下降，说明可视化缓存/绘制是重要开销。
- 如果关闭后 CPU 仍然接近 100%，重点转向 RTSP 读帧/解码和后处理。
- 如果 `capture_read_ms` 很高，说明 OpenCV/FFmpeg 取流解码是主要瓶颈。
- 如果 JPEG 请求后 `last_snapshot_encode_ms` 或 `last_annotated_encode_ms` 很高，说明接口访问时的 JPEG 编码开销较大。
