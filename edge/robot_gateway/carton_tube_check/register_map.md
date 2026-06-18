# Carton Tube/Product Placement Check Robot Protocol Register Map

本单任务服务已经改为机器人协议寄存器地址。正式同时对接纸隔板 + 纸筒任务时，建议运行上一级目录的 `vision_robot_protocol_modbus_tcp.py` 统一服务。

## Holding Registers, 0 基地址

| offset | 方向 | 名称 | 含义 |
|---:|---|---|---|
| 0 | Vision -> Robot | heartbeat | 0.5s +1，到 1000 后清零 |
| 2 | Vision -> Robot | product_result | 0=无触发/无结果，1=正常，2=异常 |
| 102 | Robot -> Vision | product_trigger | 0=不触发，1=触发纸筒/产品放置检测 |

当 102=0 时，寄存器 2 清零。
