#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${VISIONOPS_INSTALL_DIR:-/opt/visionops}"
COLLECTOR_DIR="${VISIONOPS_COLLECTOR_DIR:-$INSTALL_DIR/edge/collector}"
WS="${VISIONOPS_HP60C_BRIDGE_WS:-$INSTALL_DIR/edge/hp60c_ros1_bridge_ws}"
PKG="$WS/src/visionops_hp60c_bridge"
SRC="$COLLECTOR_DIR/scripts/visionops_hp60c_ros1_bridge.cpp"

if [ ! -f /opt/ros/noetic/setup.bash ]; then
  echo "[ERROR] /opt/ros/noetic/setup.bash not found. Please install ROS Noetic first."
  exit 1
fi
if [ ! -f "$SRC" ]; then
  echo "[ERROR] bridge source not found: $SRC"
  exit 1
fi

sudo apt update
sudo apt install -y ros-noetic-roscpp ros-noetic-sensor-msgs ros-noetic-cv-bridge ros-noetic-image-transport python3-catkin-tools python3-empy python3-dev build-essential

mkdir -p "$PKG/src"
cp "$SRC" "$PKG/src/visionops_hp60c_ros1_bridge.cpp"
cat > "$PKG/package.xml" <<'XML'
<?xml version="1.0"?>
<package format="2">
  <name>visionops_hp60c_bridge</name>
  <version>0.1.0</version>
  <description>VisionOps HP60C ROS1 C++ HTTP bridge</description>
  <maintainer email="visionops@example.com">VisionOps</maintainer>
  <license>Proprietary</license>
  <buildtool_depend>catkin</buildtool_depend>
  <depend>roscpp</depend>
  <depend>sensor_msgs</depend>
  <depend>cv_bridge</depend>
</package>
XML
cat > "$PKG/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.0.2)
project(visionops_hp60c_bridge)
add_compile_options(-std=c++14)
find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs cv_bridge)
find_package(OpenCV REQUIRED)
catkin_package()
include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_executable(visionops_hp60c_ros1_bridge src/visionops_hp60c_ros1_bridge.cpp)
target_link_libraries(visionops_hp60c_ros1_bridge ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
CMAKE

source /opt/ros/noetic/setup.bash
cd "$WS"
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

sudo tee /etc/systemd/system/visionops-hp60c-ros1-bridge.service >/dev/null <<EOF
[Unit]
Description=VisionOps HP60C ROS1 C++ Bridge
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=$WS
Environment=ROS_MASTER_URI=http://127.0.0.1:11311
Environment=ROS_IP=127.0.0.1
EnvironmentFile=-$INSTALL_DIR/edge/runtime/cpp.env
ExecStart=/bin/bash -lc 'source /opt/ros/noetic/setup.bash && source $WS/devel/setup.bash && exec rosrun visionops_hp60c_bridge visionops_hp60c_ros1_bridge'
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable visionops-hp60c-ros1-bridge.service

echo "[OK] HP60C ROS1 bridge installed."
echo "Start official HP60C driver first, then run:"
echo "  sudo systemctl restart visionops-hp60c-ros1-bridge.service"
echo "  curl http://127.0.0.1:18181/health"
