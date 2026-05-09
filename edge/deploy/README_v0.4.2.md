# VisionOps v0.4.2：RGA resize + color convert 预处理

本版本基于 v0.4.1 的 RGA 检测骨架，启用方案 B：

- OpenCV 继续负责 RTSP/USB 取帧；
- RGA 负责将原始 BGR 帧 resize 到 letterbox 有效区域，并转换为 RGB；
- CPU 负责创建 640x640 letterbox padding 画布；
- CPU 预处理路径完整保留，RGA 失败时自动 fallback。

## 部署

```bash
bash edge/deploy/deploy_cpp.sh \
  --host 192.168.1.200 \
  --preprocess-backend auto
```

强制 CPU 对照：

```bash
bash edge/deploy/deploy_cpp.sh \
  --host 192.168.1.200 \
  --preprocess-backend cpu
```

强制 RGA：

```bash
bash edge/deploy/deploy_cpp.sh \
  --host 192.168.1.200 \
  --preprocess-backend rga
```

## 验收

```bash
curl -s http://127.0.0.1:18080/health | python3 -m json.tool

curl -X POST http://127.0.0.1:18080/stream/stop
sleep 1
curl -X POST http://127.0.0.1:18080/stream/start
sleep 5

curl -s http://127.0.0.1:18080/stream/status | python3 -m json.tool
curl -s http://127.0.0.1:18080/stream/latest_result | python3 -m json.tool

curl http://127.0.0.1:18080/stream/annotated.jpg -o /tmp/annotated_v0_4_2.jpg
file /tmp/annotated_v0_4_2.jpg
```

重点检查：

- `/health` 中 `preprocess_backend_active` 应为 `rga`，`rga_enabled_for_preprocess` 应为 `true`；
- `/stream/latest_result` 中 `timing.preprocess_backend` 应为 `rga`；
- `annotated.jpg` 颜色正常、比例正常、bbox 位置不偏移；
- 与 v0.3.1 基线比较 `preprocess_ms`、`total_ms`、CPU 占用。
