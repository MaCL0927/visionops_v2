# HP60C ROS1 相机源接入 VisionOps C++ 设置卡片

## 设计原则

本版不再让 Collector 的 Python 相机线程直接 import `rospy/sensor_msgs/cv_bridge`。HP60C 的 ROS 图像读取由独立 C++ 进程完成：

```text
HP60C 官方 ROS1 驱动 roslaunch ascamera hp60c.launch
        ↓
/ascamera_hp60c/rgb0/image
        ↓
visionops_hp60c_ros1_bridge  C++/roscpp/cv_bridge
        ↓
http://127.0.0.1:18181/stream/snapshot.jpg
        ↓
Collector /api/cpp/ros1/stream/snapshot.jpg 只做 HTTP 代理
        ↓
Web 拍照采集页 C++ 预览/取图
```

因此 Web 设置界面只在“C++ 相机类型与输入源”下拉框中新增 `ROS1 图像话题（HP60C）`，不会新增“Python/ROS1 相机源”卡片。

## 安装 C++ ROS1 bridge

确认官方 HP60C ROS 包已经能正常发布图像：

```bash
source /opt/ros/noetic/setup.bash
source ~/ascam_ws/devel/setup.bash
roslaunch ascamera hp60c.launch
```

另开终端：

```bash
source /opt/ros/noetic/setup.bash
source ~/ascam_ws/devel/setup.bash
rostopic hz /ascamera_hp60c/rgb0/image
```

安装 bridge：

```bash
cd /opt/visionops/edge/collector
bash scripts/install_hp60c_ros1_bridge.sh
```

启动 bridge：

```bash
sudo systemctl restart visionops-hp60c-ros1-bridge.service
curl http://127.0.0.1:18181/health | python3 -m json.tool
```

测试快照：

```bash
curl -o /tmp/hp60c_bridge.jpg http://127.0.0.1:18181/stream/snapshot.jpg
ls -lh /tmp/hp60c_bridge.jpg
```

## Web 设置

进入：

```text
设置 → 相机设置 → C++ 相机类型与输入源
```

选择：

```text
ROS1 图像话题（HP60C）
```

默认参数：

```text
ROS1 RGB 图像话题: /ascamera_hp60c/rgb0/image
C++ ROS bridge 地址: http://127.0.0.1:18181
```

点击保存/应用后，配置会写入：

```text
/opt/visionops/edge/runtime/cpp.env
```

关键字段：

```bash
VISIONOPS_CPP_CAMERA_TYPE=ros1
VISIONOPS_CPP_CAMERA_SOURCE=/ascamera_hp60c/rgb0/image
VISIONOPS_HP60C_ROS1_TOPIC=/ascamera_hp60c/rgb0/image
VISIONOPS_CPP_ROS1_BRIDGE_URL=http://127.0.0.1:18181
```

## 接口验证

不再使用旧 Python 相机接口：

```bash
curl http://127.0.0.1:8090/api/camera/status
```

应使用 C++/ROS1 bridge 代理接口：

```bash
curl -s http://127.0.0.1:8090/api/cpp/ros1/health | python3 -m json.tool
curl -s http://127.0.0.1:8090/api/cpp/ros1/stream/status | python3 -m json.tool
curl -o /tmp/visionops_hp60c_ros1.jpg http://127.0.0.1:8090/api/cpp/ros1/stream/snapshot.jpg
```

## 当前边界

这一版解决的是“HP60C ROS1 RGB 图像进入 VisionOps C++ 预览/拍照采集入口”。如果要让 `visionops_inference_cpp` 直接对 ROS topic 做实时检测，仍需要在 C++ 推理服务源码中增加 `RosImageCapture` 后端，或者让推理服务读取 bridge 的 MJPEG/HTTP snapshot。Collector 包本身只包含 Web/设置/代理层，不能替代 C++ 推理二进制内部的取流后端。
