# VisionOps HP60C Angstrong SDK Bridge

This service replaces the previous ROS1 image bridge with a direct Angstrong C++ SDK bridge.

It keeps the same HTTP interface used by VisionOps and `tube_station`:

- `GET /health`
- `GET /stream/snapshot.jpg`
- `GET /stream.mjpeg`
- `GET /stream/mjpeg`
- `GET /stream/status`
- `POST /stream/start` / `POST /stream/stop` are no-op compatibility endpoints.

Default listen address:

```text
http://127.0.0.1:18181
```

## Why this replaces ROS

The Angstrong SDK callback can provide RGB/MJPEG/YUYV frames directly. This bridge decodes/converts them to OpenCV BGR and re-encodes a normal JPEG snapshot / MJPEG stream. VisionOps can continue using:

```bash
VISIONOPS_CPP_CAMERA_SOURCE=http://127.0.0.1:18181/stream.mjpeg
VISIONOPS_TUBE_SNAPSHOT_URL=http://127.0.0.1:18181/stream/snapshot.jpg
```

## Install

The SDK path on LB3576 is expected to be:

```text
/home/neardi/AngstrongCameraSdk_v1.2.61.20250910/demo/linux_ros
```

If your path differs, edit:

```bash
/opt/visionops/edge/robot_gateway/hp60c_sdk_bridge/hp60c_sdk_bridge.env
```

Install:

```bash
cd /opt/visionops/edge/robot_gateway/hp60c_sdk_bridge
bash install_hp60c_sdk_bridge_service.sh
```

Stop ROS bridge before starting the SDK bridge:

```bash
sudo systemctl stop visionops-hp60c-ros1-bridge.service 2>/dev/null || true
sudo systemctl restart visionops-hp60c-sdk-bridge.service
```

Check:

```bash
curl -s http://127.0.0.1:18181/health | python3 -m json.tool
curl -o /tmp/hp60c_sdk.jpg http://127.0.0.1:18181/stream/snapshot.jpg
ls -lh /tmp/hp60c_sdk.jpg
```

## Notes

- The vendor demo saves RGB/depth as raw `.yuv`; that is normal. This bridge converts available frame data to JPEG for VisionOps.
- It prefers `mjpegImg` when available, then `rgbImg`, then `yuyvImg`.
- If the saved image is vertically inverted, edit `VISIONOPS_HP60C_FLIP_VERTICAL`.
- If colors are swapped, set `VISIONOPS_HP60C_RGB_ORDER=rgb` and restart.

## Service logs

```bash
sudo systemctl status visionops-hp60c-sdk-bridge.service --no-pager -l
sudo journalctl -u visionops-hp60c-sdk-bridge.service -f -o cat
```
