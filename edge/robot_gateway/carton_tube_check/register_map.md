# VisionOps Carton Tube Check Modbus-TCP Register Map

本服务独立于 `robot_gateway/tube_station`，默认服务名：

```text
visionops-carton-tube-check.service
```

默认端口：

```text
1503
```

如果要使用 1502，需要先停掉其他占用 1502 的 Modbus-TCP 服务，例如 `visionops-tube-station.service`。

## 基本触发与结果寄存器

| Holding Register | 方向 | 含义 |
|---:|---|---|
| HR0 | PLC -> VisionOps | trigger_cmd，写 1 触发一次检测，建议检测完成后写回 0 |
| HR1 | PLC -> VisionOps | trigger_seq，每次触发递增；服务只接受新的 seq |
| HR2 | VisionOps -> PLC | status：0=idle，1=busy，2=done，3=error |
| HR3 | VisionOps -> PLC | result_seq，完成时回写本次 trigger_seq |
| HR4 | VisionOps -> PLC | final_result：0=unknown，1=OK，2=NG，3=ERROR |
| HR5 | VisionOps -> PLC | ng_reason：0=NONE，1=LYING_DETECTED，2=STAND_COUNT_LOW，3=DEPTH_INVALID，4=HEIGHT_HIGH，9=INTERNAL_ERROR |
| HR6 | VisionOps -> PLC | error_code：0=无系统错误；201=取 RGB 失败；202=推理失败；203=深度图失败；204=分析失败；205=JSON 失败；301=内部错误 |
| HR7 | VisionOps -> PLC | process_time_ms，本次检测耗时 |
| HR8 | VisionOps -> PLC | heartbeat，心跳递增 |
| HR9 | VisionOps -> PLC | stand_count |
| HR10 | VisionOps -> PLC | lying_count |
| HR11 | VisionOps -> PLC | high_count，高出异常数量 |
| HR12 | VisionOps -> PLC | max_height_diff_mm，最大高度差，单位 mm |
| HR13 | VisionOps -> PLC | valid_prediction_count，有效预测数量 |
| HR14 | VisionOps -> PLC | raw_prediction_count，原始预测数量 |
| HR15 | VisionOps -> PLC | image_width |
| HR16 | VisionOps -> PLC | image_height |
| HR17 | VisionOps -> PLC | depth_width |
| HR18 | VisionOps -> PLC | depth_height |
| HR19 | VisionOps -> PLC | baseline_mode：1=row_median，2=current_frame_median，3=fixed_env |
| HR20 | VisionOps -> PLC | expected_rows，默认 5 |
| HR21 | VisionOps -> PLC | expected_cols，默认 8 |
| HR22 | VisionOps -> PLC | detected_slot_count，矩阵中检测到的槽位数 |
| HR23 | VisionOps -> PLC | missing_slot_count，矩阵中未检测到的槽位数 |

PLC 最简单判断：

```text
HR2 == 2 且 HR4 == 1  -> OK
HR2 == 2 且 HR4 == 2  -> NG
HR2 == 3 或 HR4 == 3  -> 系统错误，建议按 NG 处理
```

## 5x8 矩阵寄存器

矩阵按行优先展开：

```text
index = row_id * expected_cols + col_id
```

默认 5 行 x 8 列，所以每个矩阵占 40 个寄存器。

| 范围 | 含义 |
|---:|---|
| HR30-HR69 | depth_mm，每个槽位参与计算的纸筒深度，单位 mm；65535=缺失 |
| HR70-HR109 | height_diff_mm，`baseline_depth - current_depth`；按 int16 解析，32767=缺失；正数表示更近/更高 |
| HR110-HR149 | height_high：0=正常，1=高出异常，65535=缺失 |
| HR150-HR189 | baseline_depth_mm，每个槽位对应的行基准深度，单位 mm；65535=缺失 |

注意：HR70-HR109 是有符号 int16 编码。例如寄存器值 65530 按 int16 解析为 -6。
