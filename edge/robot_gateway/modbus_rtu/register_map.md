# VisionOps Modbus Register Map

本版本 RTU/TCP 共用寄存器定义，请查看：

```text
../modbus_common/register_map_v2.md
```

其中 `VISIONOPS_MODBUS_ADDRESS_BASE` 可控制 VisionOps 内部 `reg[0]` 对外映射到哪个 Modbus 协议地址。
例如 PLC 表要求从 404097 读取时，通常设置：

```bash
VISIONOPS_MODBUS_ADDRESS_BASE=4096
```
