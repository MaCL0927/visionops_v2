# VisionOps v0.4.2.1：RGA 预处理分段计时与实验模式对比版

本版用于排查 v0.4.2 中 RGA 预处理耗时反而高于 CPU 的原因。它不是最终优化版，而是诊断版。

## 目标

- 保留 v0.4.2 的 RGA 接入能力。
- 增加预处理分段耗时输出。
- 支持 3 个 RGA 实验模式：
  - `off`：关闭 RGA，走 CPU 预处理路径。
  - `resize_color`：RGA resize + BGR->RGB，CPU padding。
  - `resize_only`：RGA resize BGR->BGR，CPU padding + CPU BGR->RGB。
- 通过同一套 `/stream/latest_result` 输出公平比较三种路径。

## 部署示例

```bash
bash edge/deploy/deploy_cpp.sh \
  --host 192.168.1.200 \
  --preprocess-backend auto \
  --rga-mode resize_color
```

## 三种模式切换

### 1. CPU baseline

```bash
sudo sed -i 's/^VISIONOPS_CPP_PREPROCESS_BACKEND=.*/VISIONOPS_CPP_PREPROCESS_BACKEND=auto/' /opt/visionops/edge/runtime/cpp.env
sudo sed -i 's/^VISIONOPS_CPP_RGA_MODE=.*/VISIONOPS_CPP_RGA_MODE=off/' /opt/visionops/edge/runtime/cpp.env
sudo systemctl restart visionops-inference-cpp
```

### 2. RGA resize + color convert

```bash
sudo sed -i 's/^VISIONOPS_CPP_PREPROCESS_BACKEND=.*/VISIONOPS_CPP_PREPROCESS_BACKEND=auto/' /opt/visionops/edge/runtime/cpp.env
sudo sed -i 's/^VISIONOPS_CPP_RGA_MODE=.*/VISIONOPS_CPP_RGA_MODE=resize_color/' /opt/visionops/edge/runtime/cpp.env
sudo systemctl restart visionops-inference-cpp
```

### 3. RGA resize only

```bash
sudo sed -i 's/^VISIONOPS_CPP_PREPROCESS_BACKEND=.*/VISIONOPS_CPP_PREPROCESS_BACKEND=auto/' /opt/visionops/edge/runtime/cpp.env
sudo sed -i 's/^VISIONOPS_CPP_RGA_MODE=.*/VISIONOPS_CPP_RGA_MODE=resize_only/' /opt/visionops/edge/runtime/cpp.env
sudo systemctl restart visionops-inference-cpp
```

## 验收命令

```bash
curl -s http://127.0.0.1:18080/health | python3 -m json.tool

curl -X POST http://127.0.0.1:18080/stream/stop
sleep 1
curl -X POST http://127.0.0.1:18080/stream/start
sleep 8

curl -s http://127.0.0.1:18080/stream/status | python3 -m json.tool
curl -s http://127.0.0.1:18080/stream/latest_result | python3 -m json.tool
curl http://127.0.0.1:18080/stream/annotated.jpg -o /tmp/annotated_v0_4_2_1.jpg
file /tmp/annotated_v0_4_2_1.jpg
```

## 重点观察字段

`/stream/latest_result` 中的：

```json
"timing": {
  "preprocess_ms": 8.5,
  "preprocess_backend": "cpu | rga_resize_color | rga_resize_only",
  "preprocess_detail": {
    "backend": "cpu | rga_resize_color | rga_resize_only",
    "rga_mode": "off | resize_color | resize_only",
    "resized_w": 640,
    "resized_h": 362,
    "meta_calc_ms": 0.0,
    "cpu_resize_ms": 0.0,
    "rga_resize_color_ms": 0.0,
    "rga_resize_only_ms": 0.0,
    "cpu_canvas_alloc_ms": 0.0,
    "cpu_padding_copy_ms": 0.0,
    "cpu_cvtcolor_ms": 0.0,
    "continuity_ms": 0.0,
    "total_ms": 0.0
  }
}
```

## 对比表建议

| 模式 | preprocess_backend | rga_mode | preprocess_ms | resize耗时 | cvtColor耗时 | padding耗时 | detect_fps | CPU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CPU | auto/cpu | off | | cpu_resize_ms | cpu_cvtcolor_ms | cpu_padding_copy_ms | | |
| RGA resize_color | auto/rga | resize_color | | rga_resize_color_ms | 0 | cpu_padding_copy_ms | | |
| RGA resize_only | auto/rga | resize_only | | rga_resize_only_ms | cpu_cvtcolor_ms | cpu_padding_copy_ms | | |

如果 `rga_resize_color_ms` 或 `rga_resize_only_ms` 明显高于 `cpu_resize_ms`，说明当前 virtual address RGA 路径本身不划算，后续要么继续优化为 DMA-BUF，要么先保留 CPU 预处理。
