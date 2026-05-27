# VisionOps Modbus RTU Master Push v1.3

## 适用模式

本模式用于 **3576 主动写 PLC**：

```text
3576 = Modbus RTU Master
PLC  = Modbus RTU Slave
```

和已有的 `modbus_rtu` 从站模式不同：

```text
modbus_rtu/        : PLC 主站读取 3576 从站
modbus_rtu_master/ : 3576 主站写入 PLC 从站
```

两种模式不要同时占用同一个 `/dev/ttyS5`。

## 默认写入参数

```text
PLC 从站地址：1
串口参数：9600, 8N1
功能码：16 Write Multiple Registers
目标协议地址：4096
写入数量：120
刷新周期：100ms
```

`mbpoll -r 4097` 默认对应协议地址 4096，因此：

```text
PLC [4097] = VisionOps reg[0]
PLC [4098] = VisionOps reg[1]
PLC [4099] = VisionOps reg[2]
...
```

如果 PLC 表显示 `404097`，通常也对应协议地址 `4096`。

## 关键寄存器

写入 PLC 后，PLC 目标区应看到：

| PLC相对位置 | VisionOps寄存器 | 含义 |
|---:|---:|---|
| 起始+0 | reg[0] | 22096 / 0x5650 |
| 起始+1 | reg[1] | 协议版本 |
| 起始+2 | reg[2] | heartbeat |
| 起始+5 | reg[5] | task_type |
| 起始+6 | reg[6] | result_schema |
| 起始+13 | reg[13] | ng_flag |
| 起始+16 | reg[16] | result_count |
| 起始+100 | reg[100] | 第一个结果起始字段 |

## 验证

用电脑 USB-RS485 读取 PLC：

```bash
mbpoll -m rtu -a 1 -b 9600 -P none -s 1 -t 4 -r 4097 -c 120 /dev/ttyUSB0
```

如果写入成功，应看到：

```text
[4097] = 22096
[4098] = 122
[4099] = 持续变化的 heartbeat
```

具体协议版本取决于当前 `modbus_common/register_mapper.py`。
