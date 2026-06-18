# VisionOps Orbbec Gemini 336L SDK Bridge

This directory restores the Orbbec Gemini 336L SDK HTTP bridge used by the VisionOps web side.

It provides the same HTTP style as the other VisionOps camera bridges:

- `GET /health`
- `GET /stream/status`
- `GET /stream/snapshot.jpg`
- `GET /stream/depth.png`
- `GET /stream/depth_vis.jpg`
- `GET /stream/depth_meta`
- `GET /stream.mjpeg`, `/stream/mjpeg`, `/stream.mjpg`
- `POST /stream/start`, `POST /stream/stop` compatibility endpoints

Default address:

```text
http://127.0.0.1:18182
```

If your web/runtime settings previously used another port, edit:

```bash
/opt/visionops/edge/robot_gateway/orbbec336l_bridge/orbbec336l_bridge.env
```

## Install

```bash
cd /opt/visionops/edge/robot_gateway/orbbec336l_bridge
sudo bash install_orbbec336l_bridge_service.sh
sudo systemctl restart visionops-orbbec336l-bridge.service
```

## Check

```bash
systemctl status visionops-orbbec336l-bridge.service --no-pager -l
sudo journalctl -u visionops-orbbec336l-bridge.service -f -o cat
curl -s http://127.0.0.1:18182/health | python3 -m json.tool
curl -o /tmp/orbbec336l_rgb.jpg http://127.0.0.1:18182/stream/snapshot.jpg
curl -o /tmp/orbbec336l_depth.png http://127.0.0.1:18182/stream/depth.png
curl -o /tmp/orbbec336l_depth_vis.jpg http://127.0.0.1:18182/stream/depth_vis.jpg
ls -lh /tmp/orbbec336l_*
```

## SDK path

The installer tries common SDK locations. If it cannot find the SDK, edit:

```bash
VISIONOPS_ORBBEC336L_SDK_ROOT
VISIONOPS_ORBBEC336L_SDK_INCLUDE_DIR
VISIONOPS_ORBBEC336L_SDK_LIB_DIR
```

in `orbbec336l_bridge.env`, then rerun the install script.
