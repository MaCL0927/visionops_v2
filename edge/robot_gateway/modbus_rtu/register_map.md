# VisionOps Modbus RTU Register Map

当前版本使用统一寄存器表 v2。

请参考：

```text
edge/robot_gateway/modbus_common/register_map_v2.md
```

RTU 通信参数默认：

```text
Slave ID: 1
Function: 03 Read Holding Registers
Serial: /dev/ttyS5
Baudrate: 9600
Format: 8N1
GPIO: GPIO136, 0=RX, 1=TX
```

测试读取公共区：

```bash
mbpoll -m rtu -a 1 -b 9600 -P none -s 1 -t 4 -r 1 -c 50 /dev/ttyUSB0
```

测试读取更多寄存器：

```bash
mbpoll -m rtu -a 1 -b 9600 -P none -s 1 -t 4 -r 1 -c 120 /dev/ttyUSB0
```
