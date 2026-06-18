# VisionOps 与机器人/PLC 通讯协议说明

本目录新增统一 Modbus-TCP 服务：

```bash
vision_robot_protocol_modbus_tcp.py
vision_robot_protocol.env
install_vision_robot_protocol_service.sh
test_vision_robot_protocol_client.py
```

它基于现有两个任务：

- `carton_partition_check`：纸隔板/井字隔板 5x8 小方格结构检测；
- `carton_tube_check`：纸筒/产品放置检测。

正式和机器人同事对接时，建议只启动这个统一服务，不要同时启动原来的两个单任务 Modbus 服务，避免端口和寄存器协议混乱。

---

## 1. 角色

- 3576 视觉盒子：Modbus-TCP 从站 / Server / Slave；
- 机器人 PLC / 上位机：Modbus-TCP 主站 / Client / Master。

默认参数：

```text
IP：3576 的实际 IP
端口：5045
站号 / Unit ID：1
寄存器类型：Holding Register
地址基准：0 基 offset
```

---

## 2. 寄存器映射

### 视觉 -> 机器人/PLC

| offset | 名称 | 含义 |
|---:|---|---|
| 0 | 通讯心跳 | 0.5 秒 +1，加到 1000 后清零 |
| 1 | 井字隔板识别判断反馈 | 0=无触发/无结果，1=正常，2=异常 |
| 2 | 产品放置识别判断反馈 | 0=无触发/无结果，1=正常，2=异常 |
| 3 | 产品放置坐标识别反馈 | 0=无触发/无结果，1=正常，2=异常 |
| 20~99 | 放置位 1~40 的 X/Y 坐标 | offset20=slot1_x，21=slot1_y，...，98=slot40_x，99=slot40_y |

### 机器人/PLC -> 视觉

| offset | 名称 | 含义 |
|---:|---|---|
| 100 | 机器人/PLC 通讯心跳 | 视觉侧只读取，不主动写 |
| 101 | 井字隔板识别判断触发 | 0=不触发，1=触发 |
| 102 | 产品放置识别判断触发 | 0=不触发，1=触发 |
| 103 | 产品放置坐标识别触发 | 0=不触发，1=触发 |

说明：

- 上位机写 `101=1` 后，视觉执行纸隔板是否放好检测，并把结果写到 `1`。
- 上位机写 `102=1` 后，视觉执行纸筒/产品放置检测，并把结果写到 `2`。
- 上位机写 `103=1` 后，视觉执行纸隔板小方格中心点识别，并把结果写到 `3`，坐标写到 `20~99`。
- 上位机收到结果后，将对应触发寄存器写回 `0`。
- 当触发寄存器为 `0` 时，视觉侧会把对应结果寄存器清零。
- 坐标寄存器 `20~99` 不会因为触发信号归零而清零，保留最近一次坐标识别结果。

---

## 3. 部署

复制整个 `robot_gateway` 到 3576 的：

```bash
/opt/visionops/edge/robot_gateway
```

安装统一服务：

```bash
cd /opt/visionops/edge/robot_gateway
sudo bash install_vision_robot_protocol_service.sh
```

启动：

```bash
sudo systemctl restart visionops-robot-protocol.service
```

查看状态：

```bash
sudo systemctl status visionops-robot-protocol.service --no-pager
```

查看日志：

```bash
sudo journalctl -u visionops-robot-protocol.service -f -o cat
```

---

## 4. PC 端模拟测试

纸隔板检测：

```bash
python robot_gateway/test_vision_robot_protocol_client.py \
  --host 192.168.213.145 \
  --port 5045 \
  --unit-id 1 \
  --task partition
```

纸筒/产品放置检测：

```bash
python robot_gateway/test_vision_robot_protocol_client.py \
  --host 192.168.213.145 \
  --port 5045 \
  --unit-id 1 \
  --task tube
```

坐标识别：

```bash
python robot_gateway/test_vision_robot_protocol_client.py \
  --host 192.168.213.145 \
  --port 5045 \
  --unit-id 1 \
  --task coord \
  --print-coords
```

---

## 5. 注意事项

1. 如果上位机软件使用 `40001` 显示法，则 offset 0 对应 40001，offset 101 对应 40102。
2. 如果上位机软件直接填写 Modbus 协议地址，则直接填 0、1、2、3、20、101、102、103。
3. 该协议只返回 `1=正常`、`2=异常`，内部的具体 NG 原因仍保存在 `/tmp/vision_robot_protocol_latest/<task>/result.json` 中用于调试。
4. Modbus 03 功能码单次最多读取 125 个 Holding Register；如果要读 0~199，需要分段读取。
