# VisionOps Modbus Register Map v2

适用版本：

- Modbus RTU v1.2.1：`/dev/ttyS5` + RS485 + GPIO136 方向控制
- Modbus TCP v1.2.1：`0.0.0.0:1502`

RTU 和 TCP **共用同一套寄存器定义**。区别只在通信通道：

- RTU：上位机通过 RS485 读取 Holding Registers，功能码 03。
- TCP：上位机通过网口读取 Holding Registers，功能码 03。

## 基本规则

- Holding Register 为 16-bit unsigned integer。
- 协议地址从 0 开始。部分上位机软件会显示为 40001，对应协议地址 0。
- 浮点数统一放大成整数，例如 `confidence_x10000 = confidence * 10000`。
- 角度等可能为负数的字段按 int16 编码，上位机读取后需要按 int16 还原。
- 32-bit 数值拆成高 16 位和低 16 位。
- v2 默认寄存器数量为 300。

## 任务类型 task_type

`task_type` 来自 `http://127.0.0.1:8090/api/cpp/stream/latest_result` 返回 JSON 的 `task` 字段。

| task_type | 任务 |
|---:|---|
| 0 | unknown |
| 1 | classification |
| 2 | detection |
| 3 | obb_detection |
| 4 | segmentation |
| 5 | roi_classification |

## 结果 schema result_schema

| result_schema | 含义 |
|---:|---|
| 0 | unknown |
| 101 | classification_topk_v1 |
| 201 | detection_xyxy_v1 |
| 301 | obb_cxcywh_angle_points_v1 |
| 401 | segmentation_summary_v1 |
| 501 | roi_det_cls_v1 |

---

# 0~49：公共状态区

| 地址 | 名称 | 含义 |
|---:|---|---|
| 0 | magic | 固定 `0x5650`，十进制 22096 |
| 1 | protocol_version | v2.1 为 121 |
| 2 | heartbeat | 心跳计数，每次刷新 +1 |
| 3 | service_status | 0=未知，1=运行中无结果，2=有结果，3=接口错误 |
| 4 | result_valid | 0=无效，1=有效 |
| 5 | task_type | 任务类型，见上表 |
| 6 | result_schema | 结果解释方式，见上表 |
| 7 | frame_id_hi | frame_id 高 16 位 |
| 8 | frame_id_lo | frame_id 低 16 位 |
| 9 | timestamp_s_hi | 秒级时间戳高 16 位 |
| 10 | timestamp_s_lo | 秒级时间戳低 16 位 |
| 11 | timestamp_ms | 毫秒部分，0~999 |
| 12 | latency_ms | 最新延迟，ms |
| 13 | ng_flag | 0=OK，1=NG |
| 14 | primary_class_id | 主结果类别 ID |
| 15 | primary_conf_x10000 | 主结果置信度 ×10000 |
| 16 | result_count | 当前结果数量 |
| 17 | image_width | 图像宽，未知为 0 |
| 18 | image_height | 图像高，未知为 0 |
| 19 | item_stride | 每条结果占用寄存器数量 |
| 20 | item_base | 结果列表起始地址，固定 100 |
| 21 | max_items | 最多结果数量 |
| 22 | common_size | 公共区大小，固定 50 |
| 23 | primary_base | 主结果区起始地址，固定 50 |
| 24 | control_base | 控制区起始地址，固定 200 |
| 25~49 | reserved | 预留 |

---

# 50~99：主结果区

## classification_topk_v1, result_schema=101

分类任务使用 50~69 存 TopK，每个类别占 2 个寄存器。

| 地址 | 名称 | 含义 |
|---:|---|---|
| 50 | top1_class_id | Top1 类别 ID |
| 51 | top1_conf_x10000 | Top1 置信度 ×10000 |
| 52 | top2_class_id | Top2 类别 ID |
| 53 | top2_conf_x10000 | Top2 置信度 ×10000 |
| 54 | top3_class_id | Top3 类别 ID |
| 55 | top3_conf_x10000 | Top3 置信度 ×10000 |
| 56 | top4_class_id | Top4 类别 ID |
| 57 | top4_conf_x10000 | Top4 置信度 ×10000 |
| 58 | top5_class_id | Top5 类别 ID |
| 59 | top5_conf_x10000 | Top5 置信度 ×10000 |
| 60~99 | reserved | 预留 |

其他任务当前主要使用公共区 `primary_class_id` 和 `primary_conf_x10000` 表示主结果，50~99 预留。

---

# 100~199：结果列表区

不同任务根据 `result_schema` 解释 100 之后的结果列表。实际起始地址由 `reg[20]` 给出，当前固定为 100。

## detection_xyxy_v1, result_schema=201

每个目标占 12 个寄存器。

| 偏移 | 名称 | 含义 |
|---:|---|---|
| +0 | class_id | 类别 ID |
| +1 | conf_x10000 | 置信度 ×10000 |
| +2 | x1 | 左上角 x |
| +3 | y1 | 左上角 y |
| +4 | x2 | 右下角 x |
| +5 | y2 | 右下角 y |
| +6 | center_x | 中心点 x |
| +7 | center_y | 中心点 y |
| +8 | width | 宽 |
| +9 | height | 高 |
| +10 | angle_x100 | 角度 ×100，普通检测通常为 0，按 int16 解释 |
| +11 | reserved | 预留 |

目标 `i` 的起始地址：`100 + i * 12`。

## obb_cxcywh_angle_points_v1, result_schema=301

每个 OBB 目标占 16 个寄存器。

| 偏移 | 名称 | 含义 |
|---:|---|---|
| +0 | class_id | 类别 ID |
| +1 | conf_x10000 | 置信度 ×10000 |
| +2 | cx | 中心 x |
| +3 | cy | 中心 y |
| +4 | width | 旋转框宽 |
| +5 | height | 旋转框高 |
| +6 | angle_x100 | 角度 ×100，按 int16 解释 |
| +7 | p1_x | 四点坐标点1 x |
| +8 | p1_y | 四点坐标点1 y |
| +9 | p2_x | 四点坐标点2 x |
| +10 | p2_y | 四点坐标点2 y |
| +11 | p3_x | 四点坐标点3 x |
| +12 | p3_y | 四点坐标点3 y |
| +13 | p4_x | 四点坐标点4 x |
| +14 | p4_y | 四点坐标点4 y |
| +15 | reserved | 预留 |

目标 `i` 的起始地址：`100 + i * 16`。

## segmentation_summary_v1, result_schema=401

分割任务不通过 Modbus 传完整 mask，只传结构化摘要。每个分割结果占 16 个寄存器。

| 偏移 | 名称 | 含义 |
|---:|---|---|
| +0 | class_id | 分割类别 ID |
| +1 | conf_x10000 | 置信度 ×10000 |
| +2 | area_px_hi | mask 面积高 16 位 |
| +3 | area_px_lo | mask 面积低 16 位 |
| +4 | area_ratio_x10000 | mask 面积占比 ×10000 |
| +5 | x1 | mask 外接框 x1 |
| +6 | y1 | mask 外接框 y1 |
| +7 | x2 | mask 外接框 x2 |
| +8 | y2 | mask 外接框 y2 |
| +9 | center_x | mask 中心 x |
| +10 | center_y | mask 中心 y |
| +11 | contour_points_count | 轮廓点数量，未知为 0 |
| +12~+15 | reserved | 预留 |

结果 `i` 的起始地址：`100 + i * 16`。

## roi_det_cls_v1, result_schema=501

双阶段检测 + 分类任务每个 ROI 占 16 个寄存器。

| 偏移 | 名称 | 含义 |
|---:|---|---|
| +0 | det_class_id | 检测阶段类别 ID |
| +1 | det_conf_x10000 | 检测阶段置信度 ×10000 |
| +2 | x1 | ROI / 检测框 x1 |
| +3 | y1 | ROI / 检测框 y1 |
| +4 | x2 | ROI / 检测框 x2 |
| +5 | y2 | ROI / 检测框 y2 |
| +6 | center_x | 中心 x |
| +7 | center_y | 中心 y |
| +8 | cls_class_id | 第二阶段分类类别 ID |
| +9 | cls_conf_x10000 | 第二阶段分类置信度 ×10000 |
| +10 | final_class_id | 最终业务类别 ID |
| +11 | final_conf_x10000 | 最终置信度 ×10000 |
| +12 | roi_index | ROI 编号 |
| +13 | ng_flag | 当前 ROI 是否 NG |
| +14~+15 | reserved | 预留 |

结果 `i` 的起始地址：`100 + i * 16`。

## classification_topk_v1, result_schema=101

分类结果主要在 50~59。为了统一，也会镜像到 100 之后，每个 TopK 占 4 个寄存器。

| 偏移 | 名称 | 含义 |
|---:|---|---|
| +0 | class_id | 类别 ID |
| +1 | conf_x10000 | 置信度 ×10000 |
| +2 | rank | TopK 排名，从 0 开始 |
| +3 | reserved | 预留 |

结果 `i` 的起始地址：`100 + i * 4`。

---

# 200~299：控制区预留

当前 v1.2.1 仍然只支持上位机读取。200~299 暂时预留给后续功能码 06/16，例如：

| 地址 | 名称 | 计划含义 |
|---:|---|---|
| 200 | command | 1=开始检测，2=停止检测，3=清除结果，4=切换模型 |
| 201 | command_seq | 命令序号 |
| 202 | ack_seq | 已处理命令序号 |
| 203 | command_status | 0=空闲，1=执行中，2=完成，3=失败 |
| 204 | conf_threshold_x10000 | 置信度阈值 |
| 205 | nms_threshold_x10000 | NMS 阈值 |
| 206 | model_id | 模型编号 |

---

# 推荐读取方式

## 先读公共区

上位机先读 0~49：

```text
Function: 03 Read Holding Registers
Start Address: 0
Quantity: 50
```

根据：

```text
reg[5] = task_type
reg[6] = result_schema
reg[16] = result_count
reg[19] = item_stride
reg[20] = item_base
```

决定如何读取 100 之后的结果列表。

## 一次性读取全部

测试阶段可以一次性读取 120 或 200 个寄存器。

RTU 示例：

```bash
mbpoll -m rtu -a 1 -b 9600 -P none -s 1 -t 4 -r 1 -c 120 /dev/ttyUSB0
```

TCP 示例：

```bash
mbpoll -m tcp -a 1 -r 1 -c 120 192.168.1.202 -p 1502
```


## v1.2.1 解析说明

- `task` 从 `latest_result.task` 读取。
- C++ OBB 输出的四点坐标位于 `prediction.obb.points`，v1.2.1 会读取该嵌套字段。
- OBB `prediction.obb.angle` 若带 `angle_unit: radian`，寄存器中 `angle_x100` 会转换为“度 × 100”。
- C++ segmentation 输出的面积位于 `prediction.mask.area`，轮廓位于 `prediction.mask.polygon` / `prediction.mask.segments`，v1.2.1 会读取这些嵌套字段并输出摘要。
- C++ roi_classification 输出的二阶段结果位于 `prediction.detector`、`prediction.classifier`、`prediction.roi`，v1.2.1 会读取这些嵌套字段。
