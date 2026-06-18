# VisionOps runtime split for robot protocol

目标：恢复并保留 Web 端模型切换能力，同时给纸隔板检测提供独立固定的 C++ 推理服务。

## 服务拆分

- `visionops-inference-cpp.service`
  - 继续使用 `/opt/visionops/edge/runtime/cpp.env`
  - 继续由 Web 端模型切换功能控制
  - 机器人纸筒/产品放置检测仍调用 `http://127.0.0.1:8090/api/cpp/infer`

- `visionops-inference-cpp-partition.service`
  - 使用 `/opt/visionops/edge/runtime/cpp-partition.env`
  - 固定加载 `/opt/visionops/models/rk3576-001_paper-cell_det_20260610_155540.rknn`
  - 固定监听 `8091`
  - 机器人纸隔板检测和坐标识别调用 `http://127.0.0.1:8091/api/cpp/infer`

- `visionops-robot-protocol.service`
  - 监听 Modbus-TCP `5045`
  - 机器人写 `101/102/103` 触发
  - 视觉写 `1/2/3` 结果，`20~99` 写纸隔板 40 个小方格中心点坐标

## 安装

在 3576 上执行：

```bash
cd /opt/visionops/edge
# 解压本包后
cd /opt/visionops/edge/robot_gateway
sudo bash install_robot_protocol_runtime_split_services.sh
```

## 检查

```bash
systemctl status visionops-inference-cpp.service --no-pager -l
systemctl status visionops-inference-cpp-partition.service --no-pager -l
systemctl status visionops-robot-protocol.service --no-pager -l
ss -lntp | grep -E '8090|8091|5045|18080'
```

## 注意

不要再给 `visionops-inference-cpp.service` 添加固定模型的 direct drop-in，否则会破坏 Web 端模型切换。安装脚本会删除先前错误生成的：

- `/etc/systemd/system/visionops-inference-cpp.service.d/20-tube-direct.conf`
- `/etc/systemd/system/visionops-inference-cpp-partition.service.d/10-carton-partition-model.conf`
- `/etc/systemd/system/visionops-inference-cpp-partition.service.d/20-partition-direct.conf`
