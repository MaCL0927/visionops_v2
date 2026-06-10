# VisionOps Carton Tube Check - Stage2 Depth Debug

这个文件夹是纸箱蓝色纸筒检测任务的第二阶段调试工具，**不会修改 `robot_gateway/tube_station/`**。

## 已实现内容

1. 通过 HP60C ROS1 C++ bridge 获取 RGB 快照：
   - `http://127.0.0.1:18181/stream/snapshot.jpg`
2. 调用现有 C++ OBB 单图检测接口：
   - `http://127.0.0.1:8090/api/cpp/infer`
3. 获取 HP60C 原始 16UC1 深度图 PNG：
   - `http://127.0.0.1:18181/stream/depth.png`
4. 使用检测到的 `stand` 框中心区域取深度，判断是否存在明显高出的纸筒。
5. 新增 `row_median` 模式：按行计算深度中位数，适配相机斜向下安装导致的不同行正常深度不同的问题。
6. 新增 5×8 矩阵输出：输出 `depth_mm`、`baseline_depth_mm`、`height_diff_mm`、`height_high`，方便观察每个槽位的参与计算深度。

## 使用前提

先确认新版 HP60C bridge 已启动：

```bash
curl -s http://127.0.0.1:18181/health | python3 -m json.tool
curl -o /tmp/hp60c_rgb.jpg http://127.0.0.1:18181/stream/snapshot.jpg
curl -o /tmp/hp60c_depth.png http://127.0.0.1:18181/stream/depth.png
curl -o /tmp/hp60c_depth_vis.jpg http://127.0.0.1:18181/stream/depth_vis.jpg
```

`health` 中应看到：

```text
depth_available: true
depth_width: 640
depth_height: 480
depth_output_encoding: 16UC1_mm_png
```

## 单次调试

```bash
cd /opt/visionops/edge/robot_gateway/carton_tube_check
python3 debug_depth_check_once.py --save-dir /tmp/carton_tube_debug
```

输出重点看：

```text
[MATRIX] depth_mm 5x8
[MATRIX] baseline_depth_mm 5x8
[MATRIX] height_diff_mm 5x8 = baseline - current_depth
[MATRIX] height_high 5x8
[SUMMARY] final=OK/NG reason=... baseline_mode=row_median stand=... lying=... high_count=... max_diff=...mm
```

其中：

```text
height_diff_mm = row_baseline_depth_mm - current_depth_mm
```

数值为正，表示该位置深度更小，也就是离相机更近；超过阈值则认为纸筒高出。

## 关键配置

配置文件：

```text
carton_tube_check.env
```

当前推荐配置：

```bash
VISIONOPS_CARTON_TUBE_BASELINE_MODE=row_median
VISIONOPS_CARTON_TUBE_EXPECTED_ROWS=5
VISIONOPS_CARTON_TUBE_EXPECTED_COLS=8
VISIONOPS_CARTON_TUBE_HEIGHT_THRESHOLD_MM=35
```

你的纸箱是 5 行 8 列，共 40 个槽位。如果只检测到 36 个 `stand`，矩阵中未检测到的位置会显示为 `----`。

如果正常样本 `height_diff_mm` 最大值仍然较大，可以先把阈值提高到 45 或 50 观察；如果人为高出样本的 `height_diff_mm` 明显超过正常波动，再确定最终阈值。
