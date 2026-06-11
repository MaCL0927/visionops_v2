# Carton Partition Cell Check

这个文件夹是一版纸箱隔离板检测的边缘端测试代码，放在：

```bash
edge/robot_gateway/carton_partition_check
```

目标不是直接检测很薄的隔板，而是利用当前 YOLOv8n 小方格检测结果，判断 **5 行 × 8 列 = 40 个小方格** 的数量、位置和整体网格姿态是否和正常模板一致。

## 1. 主要文件

```text
partition_check.env                 配置文件
debug_partition_check_once.py        单次取图+推理+结构判断测试脚本
partition_check_modbus_tcp.py         Modbus-TCP 触发检测服务
test_partition_check_client.py        Modbus-TCP 测试客户端
register_map.md                       寄存器说明
install_partition_check_service.sh    systemd 安装脚本
```

取图接口和推理接口保持不变：

```text
http://127.0.0.1:18181/stream/snapshot.jpg
http://127.0.0.1:8090/api/cpp/infer
```

## 2. 第一次使用：用正常样本生成模板

先确认当前画面是正常放好的隔离板，并且 YOLO 可以检测出 40 个小方格，然后运行：

```bash
cd /opt/visionops/edge/robot_gateway/carton_partition_check
VISIONOPS_PARTITION_ENV=./partition_check.env \
/opt/visionops/venv/bin/python debug_partition_check_once.py \
  --calibrate \
  --save-dir /tmp/carton_partition_calib
```

成功后会生成：

```text
/opt/visionops/edge/robot_gateway/carton_partition_check/partition_template.json
```

同时 `/tmp/carton_partition_calib/overlay.jpg` 会保存检测框和槽位编号，可以用来检查模板是否正确。

## 3. 单次测试

```bash
cd /opt/visionops/edge/robot_gateway/carton_partition_check
VISIONOPS_PARTITION_ENV=./partition_check.env \
/opt/visionops/venv/bin/python debug_partition_check_once.py \
  --save-dir /tmp/carton_partition_check_once
```

输出中重点看：

```text
final_result: OK / NG / ERROR
reason: NONE / COUNT_MISMATCH / ROW_ANGLE_DIFF / P95_CENTER_ERROR ...
valid_cell_count: 当前检测到的小方格数量
metrics.mean_center_error_px
metrics.p95_center_error_px
metrics.max_center_error_px
metrics.edge_cell_max_error_px
metrics.grid_center_offset_px
metrics.row_angle_diff_deg
metrics.max_row_angle_diff_deg
metrics.row_angle_std_diff_deg
metrics.col_angle_diff_deg
```

如果 `reason=TEMPLATE_MISSING`，说明还没有先执行 `--calibrate`。

## 4. Modbus-TCP 服务测试

手动启动服务：

```bash
cd /opt/visionops/edge/robot_gateway/carton_partition_check
VISIONOPS_PARTITION_ENV=./partition_check.env \
/opt/visionops/venv/bin/python partition_check_modbus_tcp.py
```

另一个终端触发：

```bash
cd /opt/visionops/edge/robot_gateway/carton_partition_check
/opt/visionops/venv/bin/python test_partition_check_client.py --host 127.0.0.1 --port 1504 --print-slots
```

## 5. 安装 systemd 服务

```bash
cd /opt/visionops/edge/robot_gateway/carton_partition_check
sudo bash install_partition_check_service.sh
sudo systemctl restart visionops-carton-partition-check.service
sudo journalctl -u visionops-carton-partition-check.service -f -o cat
```

默认端口为 `1504`，避免和：

```text
tube_station: 1502
carton_tube_check: 1503
```

冲突。

## 6. 倾斜判断逻辑

当检测数量等于 40 时，代码会继续做结构判断：

```text
1. 将 40 个检测框中心点排序成 5x8 网格
2. 和 partition_template.json 中的正常中心点模板逐槽位比较
3. 计算中心点误差、边缘格子误差、逐行角度差、行角度离散度、整体网格中心偏移、行列平均角度差、仿射旋转/剪切
4. 任意关键指标超阈值，则输出 NG
```

本版新增了几个更适合识别“检测数量仍然是 40，但隔板已经倾斜”的指标：

```text
max_center_error_px          所有槽位中的最大中心点误差
edge_cell_max_error_px       最外圈槽位中的最大中心点误差
max_row_angle_diff_deg       逐行角度相对模板的最大差异
row_angle_std_diff_deg       当前行角度离散度与模板行角度离散度的差异
```

这几个指标比单纯 `cell_count` 更适合识别倾斜；比直接在原图做 Hough 直线检测更稳定，因为它使用的是 YOLO 检测框中心点形成的“虚拟网格线”，不依赖模糊图像里的真实边缘线。

## 7. 阈值调整建议

先采集 30~100 张正常放置图，逐张运行单次测试，观察这些指标的正常波动：

```text
mean_center_error_px
p95_center_error_px
max_center_error_px
edge_cell_max_error_px
grid_center_offset_px
row_angle_diff_deg
max_row_angle_diff_deg
row_angle_std_diff_deg
col_angle_diff_deg
```

如果正常样本最大 `p95_center_error_px` 是 18 px，可以把阈值设成 25~30 px；如果正常样本最大角度差是 2°，可以把阈值设成 4~5°。

配置项在 `partition_check.env` 中：

```text
VISIONOPS_PARTITION_MAX_MEAN_CENTER_ERR_PX=22
VISIONOPS_PARTITION_MAX_P95_CENTER_ERR_PX=38

# 这几个用于倾斜判断，当前先只收紧它们：
VISIONOPS_PARTITION_MAX_CENTER_ERR_PX=24
VISIONOPS_PARTITION_MAX_EDGE_CELL_ERR_PX=20
VISIONOPS_PARTITION_MAX_ROW_ANGLE_DIFF_MAX_DEG=3.5
VISIONOPS_PARTITION_MAX_ROW_ANGLE_STD_DIFF_DEG=0.70

VISIONOPS_PARTITION_MAX_GRID_CENTER_OFFSET_PX=35
VISIONOPS_PARTITION_MAX_ROW_ANGLE_DIFF_DEG=5
VISIONOPS_PARTITION_MAX_COL_ANGLE_DIFF_DEG=5
VISIONOPS_PARTITION_MAX_AFFINE_ROT_DEG=5
VISIONOPS_PARTITION_MAX_AFFINE_SHEAR=0.18
```


## Modbus TCP Unit ID 兼容性

默认 `VISIONOPS_PARTITION_MODBUS_SINGLE_SLAVE=1`，服务端会接受任意 Unit ID 的 Modbus-TCP 请求，避免上位机使用 0/1/255 不一致时出现 `No Response received from the remote slave`。如现场要求严格 Unit ID，可改为 `0`，并确保上位机 Unit ID 与 `VISIONOPS_PARTITION_MODBUS_UNIT_ID` 一致。
