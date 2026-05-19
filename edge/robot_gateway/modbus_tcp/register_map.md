# VisionOps Modbus TCP Register Map

当前版本使用统一寄存器表 v2。

请参考：

```text
edge/robot_gateway/modbus_common/register_map_v2.md
```

TCP 通信参数默认：

```text
Host: 0.0.0.0
Port: 1502
Unit ID: 1
Function: 03 Read Holding Registers
```

测试读取公共区：

```bash
mbpoll -m tcp -a 1 -r 1 -c 50 192.168.1.202 -p 1502
```

测试读取更多寄存器：

```bash
mbpoll -m tcp -a 1 -r 1 -c 120 192.168.1.202 -p 1502
```
