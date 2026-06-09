# HP60C ROS1 C++ Bridge Troubleshooting

## Symptom

`/api/cpp/ros1/*` returns 502 and direct `curl -s http://127.0.0.1:18181/health | python3 -m json.tool` prints `Expecting value`.

This means the local C++ ROS1 bridge is not returning JSON, usually because the systemd service is not running or not listening on port 18181.

## Check

```bash
sudo systemctl status visionops-hp60c-ros1-bridge.service --no-pager -l
sudo journalctl -u visionops-hp60c-ros1-bridge.service -n 100 --no-pager
ss -ltnp | grep 18181 || true
curl -v http://127.0.0.1:18181/health
```

## Restart order

```bash
source /opt/ros/noetic/setup.bash
source ~/ascam_ws/devel/setup.bash
rostopic hz /ascamera_hp60c/rgb0/image

sudo systemctl restart visionops-hp60c-ros1-bridge.service
curl -s http://127.0.0.1:18181/health | python3 -m json.tool

cd /opt/visionops/edge/collector
bash start_collector.sh
```

## Notes

For ROS1/HP60C, the bridge should be treated as a shared frame provider. The Web capture page should not stop the bridge when leaving the page; it should only stop refreshing the image in the browser.
