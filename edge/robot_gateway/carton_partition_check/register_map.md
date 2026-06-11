# VisionOps Carton Partition Cell Check Modbus-TCP Register Map

默认端口：`1504`  
默认 unit id：`1`  
默认地址基准：`0`，即下表 offset=0 对应 40001 或 0 地址，取决于上位机显示方式。

## 触发区

| Offset | 名称 | 方向 | 含义 |
|---:|---|---|---|
| 0 | trigger_cmd | PLC -> VisionOps | 写 1 触发一次检测；检测完成后可写回 0 |
| 1 | trigger_seq | PLC -> VisionOps | 每次触发递增，服务会回写到 result_seq |

## 结果区

| Offset | 名称 | 含义 |
|---:|---|---|
| 2 | status | 0=idle, 1=busy, 2=done, 3=error |
| 3 | result_seq | 完成结果对应的 trigger_seq |
| 4 | final_result | 0=unknown, 1=OK, 2=NG, 3=ERROR |
| 5 | ng_reason | 见下方 reason code |
| 6 | error_code | 0=无错误，201=取图失败，202=推理失败，203=分析失败，204=JSON错误，301=内部错误，401=模板缺失 |
| 7 | process_time_ms | 本次处理耗时 ms |
| 8 | heartbeat | 心跳寄存器，约 0.5s 自增 |
| 9 | cell_count | 当前检测到的小方格数量 |
| 10 | expected_count | 期望数量，默认 40 |
| 11 | matched_count | 成功和模板匹配的格子数 |
| 12 | missing_count | 模板槽位缺失数 |
| 13 | mean_center_err_x10 | 平均中心点误差 px ×10 |
| 14 | p95_center_err_x10 | P95 中心点误差 px ×10 |
| 15 | grid_center_offset_x10 | 整体网格中心偏移 px ×10 |
| 16 | row_angle_diff_x100 | 行方向角度差 deg ×100，有符号 int16 |
| 17 | col_angle_diff_x100 | 列方向角度差 deg ×100，有符号 int16 |
| 18 | affine_rot_x100 | 仿射旋转角 deg ×100，有符号 int16 |
| 19 | affine_shear_x10000 | 仿射剪切 ×10000 |
| 20 | bad_size_count | 框大小比例异常数量，默认不参与硬判定 |
| 21 | image_width | 图像宽度 |
| 22 | image_height | 图像高度 |
| 23 | template_loaded | 1=已加载模板，0=未加载 |
| 24 | grid_assign_ok | 1=5x8 行列排序成功，0=失败 |
| 25 | rows | 期望行数，默认 5 |
| 26 | cols | 期望列数，默认 8 |
| 27 | raw_pred_count | 原始预测数量 |
| 28 | max_center_err_x10 | 单个槽位最大中心点误差 px ×10 |
| 29 | edge_cell_max_err_x10 | 最外圈边缘槽位最大中心点误差 px ×10 |
| 30 | max_row_angle_diff_x100 | 逐行角度最大差异 deg ×100，无符号量按有符号寄存器读取即可 |
| 31 | row_angle_std_diff_x100 | 行角度离散度差异 deg ×100，无符号量按有符号寄存器读取即可 |

## Reason code

| Code | 名称 | 含义 |
|---:|---|---|
| 0 | NONE | 正常 |
| 1 | COUNT_MISMATCH | 小方格数量不是 40 |
| 2 | GRID_ASSIGN_FAILED | 5x8 行列排序失败 |
| 3 | TEMPLATE_MISSING | 未生成/未加载正常模板 |
| 4 | SLOT_MISSING | 模板槽位缺失 |
| 5 | MEAN_CENTER_ERROR | 平均中心点误差超阈值 |
| 6 | P95_CENTER_ERROR | P95 中心点误差超阈值 |
| 7 | GRID_CENTER_OFFSET | 整体网格中心偏移超阈值 |
| 8 | ROW_ANGLE_DIFF | 行方向角度差超阈值 |
| 9 | COL_ANGLE_DIFF | 列方向角度差超阈值 |
| 10 | AFFINE_ROTATION | 仿射旋转角超阈值 |
| 11 | AFFINE_SHEAR | 仿射剪切超阈值 |
| 12 | BOX_SIZE_ANOMALY | 框大小异常数量超阈值 |
| 13 | MAX_CENTER_ERROR | 单个槽位最大中心点误差超阈值 |
| 14 | ROW_ANGLE_MAX_DIFF | 逐行角度最大差异超阈值 |
| 15 | ROW_ANGLE_STD_DIFF | 行角度离散度差异超阈值 |
| 16 | EDGE_CELL_ERROR | 最外圈边缘槽位中心点误差超阈值 |
| 99 | INTERNAL_ERROR | 内部错误 |

## 每个槽位调试区

默认 5×8 共 40 个槽位，按 `slot_id = row * cols + col` 排列。

| Offset | 长度 | 名称 | 含义 |
|---:|---:|---|---|
| 40 | 40 | slot_status | 0=ok, 1=missing, 2=size_bad, 3=other, 65535=无数据 |
| 90 | 40 | slot_center_err_x10 | 每个槽位中心点误差 px ×10，65535=无数据 |


## Modbus TCP Unit ID 兼容性

默认 `VISIONOPS_PARTITION_MODBUS_SINGLE_SLAVE=1`，服务端会接受任意 Unit ID 的 Modbus-TCP 请求，避免上位机使用 0/1/255 不一致时出现 `No Response received from the remote slave`。如现场要求严格 Unit ID，可改为 `0`，并确保上位机 Unit ID 与 `VISIONOPS_PARTITION_MODBUS_UNIT_ID` 一致。
