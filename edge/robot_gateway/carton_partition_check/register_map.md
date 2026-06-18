# Carton Partition Check Robot Protocol Register Map

本单任务服务已经改为机器人协议寄存器地址。正式同时对接纸隔板 + 纸筒任务时，建议运行上一级目录的 `vision_robot_protocol_modbus_tcp.py` 统一服务。

## Holding Registers, 0 基地址

| offset | 方向 | 名称 | 含义 |
|---:|---|---|---|
| 0 | Vision -> Robot | heartbeat | 0.5s +1，到 1000 后清零 |
| 1 | Vision -> Robot | partition_result | 0=无触发/无结果，1=正常，2=异常 |
| 3 | Vision -> Robot | coord_result | 0=无触发/无结果，1=正常，2=异常 |
| 20~99 | Vision -> Robot | cell_center_xy | slot1_x, slot1_y, ..., slot40_x, slot40_y |
| 101 | Robot -> Vision | partition_trigger | 0=不触发，1=触发纸隔板检测 |
| 103 | Robot -> Vision | coord_trigger | 0=不触发，1=触发坐标识别 |

当 101=0 时，寄存器 1 清零；当 103=0 时，寄存器 3 清零；坐标寄存器 20~99 不清零。
