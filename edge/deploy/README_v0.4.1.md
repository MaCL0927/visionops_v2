# VisionOps edge C++ v0.4.1

本版本基于 v0.3.1 检测可视化验收版，加入 RGA 编译检测与预处理后端开关骨架。v0.4.1 的目标是先把 RGA 接入工程、服务配置和状态观测链路打通，为 v0.4.2 的 RGA resize/letterbox 实现做准备。

## 版本边界

v0.4.1 **不替换** 已经稳定的 CPU 预处理路径。即使 `--preprocess-backend rga` 或 `auto` 检测到 RGA 可用，实际图像预处理仍然使用 v0.3.1 的 CPU `letterbox + BGR/RGB` 路径。

也就是说：

- RGA 头文件/库检测：已加入
- CMake 可选链接 `librga`：已加入
- systemd/env/启动脚本参数：已加入
- `/health` 和 `/stream/status` 状态字段：已加入
- 实际 RGA resize：留到 v0.4.2

## 新增参数

```bash
--preprocess-backend cpu|rga|auto
```

对应环境变量：

```bash
VISIONOPS_CPP_PREPROCESS_BACKEND=auto
```

默认值为：

```bash
auto
```

## 一键部署

```bash
bash edge/deploy/deploy_cpp.sh --host 192.168.1.200 --preprocess-backend auto
```

强制 CPU 验收：

```bash
bash edge/deploy/deploy_cpp.sh --host 192.168.1.200 --preprocess-backend cpu
```

验证 RGA 检测链路：

```bash
bash edge/deploy/deploy_cpp.sh --host 192.168.1.200 --preprocess-backend rga
```

注意：v0.4.1 中 `rga` 只代表“请求并检测 RGA”，不会真正启用 RGA resize。

## CMake 检测路径

RGA 头文件会在以下常见路径中查找：

```text
/usr/include
/usr/include/rga
/usr/local/include
/usr/local/include/rga
/opt/rga/include
/opt/visionops/include
```

RGA 库会在以下常见路径中查找：

```text
/usr/lib
/usr/lib/aarch64-linux-gnu
/usr/local/lib
/usr/local/lib/aarch64-linux-gnu
/opt/rga/lib
/opt/visionops/lib
```

如果没有找到 RGA，工程仍然会编译成功，并回退到 CPU 预处理。

## 验收命令

```bash
curl -s http://127.0.0.1:18080/health | python3 -m json.tool
```

重点看这些字段：

```json
{
  "version": "v0.4.1",
  "preprocess_backend_requested": "auto",
  "preprocess_backend_active": "cpu",
  "rga_compiled": true,
  "rga_runtime_available": true,
  "rga_available": true,
  "rga_enabled_for_preprocess": false
}
```

字段解释：

- `preprocess_backend_requested`：用户请求的后端，来自 `--preprocess-backend` 或 `VISIONOPS_CPP_PREPROCESS_BACKEND`
- `preprocess_backend_active`：当前实际生效的预处理后端；v0.4.1 固定为 `cpu`
- `rga_compiled`：编译时是否找到 RGA 头文件和库
- `rga_runtime_available`：运行时是否能加载或发现 `librga.so`
- `rga_available`：`rga_compiled && rga_runtime_available`
- `rga_enabled_for_preprocess`：RGA 是否真正参与预处理；v0.4.1 固定为 `false`

实时流状态也会返回这些字段：

```bash
curl -s http://127.0.0.1:18080/stream/status | python3 -m json.tool
```

单帧结果的 `timing` 中会增加：

```json
{
  "preprocess_ms": 7.392,
  "preprocess_backend": "cpu"
}
```

## 与 v0.3.1 的关系

v0.3.1 已确认：

- RTSP/OpenCV 取流正常
- RKNN 推理正常
- YOLOv8 split-head 后处理正常
- `/stream/annotated.jpg` 检测框位置正常

v0.4.1 在这些能力上只新增 RGA 可用性检测和开关，不应改变检测框、类别、置信度或坐标映射结果。

## v0.4.1 验收标准

1. 服务能正常编译、安装、启动。
2. `/health` 返回 `version=v0.4.1`。
3. `/health` 返回 RGA 状态字段。
4. `/stream/latest_result` 仍有检测结果。
5. `/stream/annotated.jpg` 检测框位置与 v0.3.1 一致。
6. 即使 RGA 不存在，也不会影响 CPU 预处理和推理链路。

## 下一步 v0.4.2

v0.4.2 才开始真正实现：

```text
OpenCV cv::Mat BGR 原图
        ↓
RGA resize / color convert
        ↓
CPU letterbox padding 或 RGA+CPU 混合 letterbox
        ↓
RKNN input
```

v0.4.2 的重点是降低 `preprocess_ms`，并用 `/stream/annotated.jpg` 验证检测框坐标不偏移。
