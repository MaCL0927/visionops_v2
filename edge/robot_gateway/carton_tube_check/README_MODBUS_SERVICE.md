# Carton Tube Check Modbus-TCP Service

该服务用于纸箱蓝色纸筒检测任务：

```text
PLC/上位机写触发寄存器
        ↓
HP60C 抓 RGB snapshot + depth.png
        ↓
调用 C++ OBB 单图推理接口 /api/cpp/infer
        ↓
第一阶段：检测 lying，有 lying 直接 NG
        ↓
第二阶段：对 stand 框中心取深度，按 row_median 判断高出异常
        ↓
写回 Modbus-TCP 寄存器
```

## 与 tube_station 的区别

本服务不修改 `robot_gateway/tube_station`，独立放在：

```text
/opt/visionops/edge/robot_gateway/carton_tube_check
```

systemd 服务名：

```text
visionops-carton-tube-check.service
```

默认 Modbus-TCP 端口：

```text
1503
```

`tube_station` 默认端口通常是 1502。两个服务如果同时运行，端口必须不同。

## 安装

```bash
cd /opt/visionops/edge/robot_gateway/carton_tube_check
bash install_carton_tube_check_service.sh
sudo systemctl restart visionops-carton-tube-check.service
sudo journalctl -u visionops-carton-tube-check.service -f -o cat
```

## 测试触发

```bash
python3 /opt/visionops/edge/robot_gateway/carton_tube_check/test_carton_tube_check_client.py \
  --host 127.0.0.1 \
  --port 1503 \
  --print-matrix
```

## 常用配置

编辑：

```bash
sudo nano /opt/visionops/edge/robot_gateway/carton_tube_check/carton_tube_check.env
```

重点参数：

```bash
VISIONOPS_CARTON_TUBE_MODBUS_PORT=1503
VISIONOPS_CARTON_TUBE_MIN_STAND_COUNT=1
VISIONOPS_CARTON_TUBE_BASELINE_MODE=row_median
VISIONOPS_CARTON_TUBE_EXPECTED_ROWS=5
VISIONOPS_CARTON_TUBE_EXPECTED_COLS=8
VISIONOPS_CARTON_TUBE_HEIGHT_THRESHOLD_MM=35
```

如果满格必须 40 个纸筒，设置：

```bash
VISIONOPS_CARTON_TUBE_MIN_STAND_COUNT=40
```

如果只是先验证高度检测，可以先保持：

```bash
VISIONOPS_CARTON_TUBE_MIN_STAND_COUNT=1
```

每次触发会保存最新调试文件到：

```text
/tmp/carton_tube_check_latest/
```

包含：

```text
rgb.jpg
depth.png
infer.json
result.json
last_seq.txt
```
