# VisionOps Tube Station Modbus TCP 寄存器表

默认监听：`0.0.0.0:1502`，Unit ID：`1`。

本服务使用 Holding Registers，内部地址默认从 0 开始。PLC/HMI 侧如果显示为 40001 格式，需要确认是否存在 +1 或 40001 偏移。

## 状态值定义

### HR2 status

| 值 | 含义 |
|---:|---|
| 0 | idle，空闲 |
| 1 | busy，检测中 |
| 2 | done，检测完成 |
| 3 | error，检测错误 |

### HR4 / HR5 tube_state

| 值 | 含义 |
|---:|---|
| 0 | unknown，未知/未触发 |
| 1 | stand，站立 |
| 2 | lying，放倒 |
| 3 | missing_or_error，缺失/异常 |

### HR6 error_code

| 值 | 含义 |
|---:|---|
| 0 | 无错误 |
| 101 | 没有检测到纸筒 |
| 102 | 只检测到一个纸筒 |
| 103 | 检测结果中没有有效类别 |
| 201 | 获取相机快照失败 |
| 202 | C++ 单图推理失败 |
| 203 | 推理返回 JSON 解析失败 |
| 301 | 服务内部异常 |

## Holding Registers

| 内部地址 | 名称 | 方向 | 含义 |
|---:|---|---|---|
| HR0 | trigger_cmd | 上位机写 | 0=空闲/复位，1=触发检测 |
| HR1 | trigger_seq | 上位机写 | 触发序号；建议每次触发加 1，防重复触发 |
| HR2 | status | 视觉盒子写 | 0=空闲，1=检测中，2=完成，3=错误 |
| HR3 | result_seq | 视觉盒子写 | 当前完成的触发序号 |
| HR4 | left_state | 视觉盒子写 | 左纸筒状态：1=站立，2=放倒，3=异常 |
| HR5 | right_state | 视觉盒子写 | 右纸筒状态：1=站立，2=放倒，3=异常 |
| HR6 | error_code | 视觉盒子写 | 错误码，0=无错误 |
| HR7 | process_time_ms | 视觉盒子写 | 单次触发处理耗时 |
| HR8 | heartbeat | 视觉盒子写 | 心跳计数，每轮递增 |
| HR9 | detection_count | 视觉盒子写 | 本次有效纸筒检测数量 |
| HR10 | left_conf | 视觉盒子写 | 左纸筒置信度 ×10000 |
| HR11 | right_conf | 视觉盒子写 | 右纸筒置信度 ×10000 |
| HR12 | left_class_id | 视觉盒子写 | 左纸筒模型 class_id，65535 表示无效 |
| HR13 | right_class_id | 视觉盒子写 | 右纸筒模型 class_id，65535 表示无效 |
| HR14 | image_width | 视觉盒子写 | 推理图像宽度 |
| HR15 | image_height | 视觉盒子写 | 推理图像高度 |

## 推荐上位机流程

1. 写 HR1 = 新触发序号。
2. 写 HR0 = 1。
3. 轮询 HR2。
4. HR2 = 2 时读取 HR4、HR5。
5. 读取完成后写 HR0 = 0。

如果 HR2 = 3，读取 HR6 查看错误码。
