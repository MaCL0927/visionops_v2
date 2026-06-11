# HP60C SDK-only cleanup

本版本移除了 Web 设置页中的 ROS1 图像话题旧入口。HP60C 相机应使用 `visionops-hp60c-sdk-bridge.service`，Web/推理服务继续读取 `http://127.0.0.1:18181/stream.mjpeg`。

清理旧 systemd 残留：

```bash
sudo systemctl disable --now visionops-hp60c-ros1-bridge.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/visionops-hp60c-ros1-bridge.service
sudo find /etc/systemd/system -type l -name visionops-hp60c-ros1-bridge.service -delete
sudo systemctl daemon-reload
sudo systemctl reset-failed visionops-hp60c-ros1-bridge.service 2>/dev/null || true
```
