# mmwave_drone_detector — RPi5 Jazzy Headless Setup

## 1. Install ROS2 Jazzy on RPi5 (if not done)

```bash
# On RPi5 via SSH
sudo apt update && sudo apt upgrade -y

# Install ROS2 Jazzy base (no desktop needed — headless)
sudo apt install software-properties-common -y
sudo add-apt-repository universe
sudo apt update
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
sudo apt install ros-jazzy-ros-base -y         # base only, no GUI
sudo apt install python3-colcon-common-extensions -y
sudo apt install ros-jazzy-launch-ros -y
```

## 2. Install Python dependencies

```bash
pip3 install pyserial numpy --break-system-packages
# scipy optional — pure numpy fallback is used if not available
pip3 install scipy --break-system-packages || echo "scipy unavailable, using numpy fallback"
```

## 3. Set up serial port permissions

```bash
sudo usermod -aG dialout $USER
sudo chmod 666 /dev/ttyUSB0 /dev/ttyUSB1
# Disable brltty if it grabs the ports
sudo systemctl stop brltty && sudo systemctl disable brltty
sudo apt remove brltty -y 2>/dev/null || true
```

## 4. Copy package to RPi5

```bash
# From your laptop:
scp -r mmwave_drone_detector_rpi5 pi@<RPI_IP>:~/ros2_ws/src/mmwave_drone_detector
```

## 5. Build

```bash
# On RPi5
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
colcon build --packages-select mmwave_drone_detector
source install/setup.bash
```

## 6. Run

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# Basic run
ros2 launch mmwave_drone_detector detector.launch.py \
    cli_port:=/dev/ttyUSB0 data_port:=/dev/ttyUSB1

# Or run node directly
ros2 run mmwave_drone_detector ros_publisher.py \
    --ros-args \
    -p cli_port:=/dev/ttyUSB0 \
    -p data_port:=/dev/ttyUSB1
```

## 7. Monitor from laptop (SSH tunnel)

```bash
# On laptop — set ROS to talk to RPi5
export ROS_DOMAIN_ID=0   # must match on both machines
export ROS_DISCOVERY_SERVER=<RPI_IP>   # optional for simple networks

# Or just SSH and echo topics on RPi5 directly
ssh pi@<RPI_IP>
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic echo /radar/drone_pose
ros2 topic echo /radar/drone_detections
ros2 topic hz /radar/point_cloud
```

## 8. Auto-start on boot (systemd service)

```bash
# Create service file
sudo tee /etc/systemd/system/mmwave_detector.service << 'EOF'
[Unit]
Description=mmWave Drone Detector ROS2 Node
After=network.target

[Service]
Type=simple
User=pi
Environment="HOME=/home/pi"
ExecStartPre=/bin/bash -c 'chmod 666 /dev/ttyUSB* 2>/dev/null; true'
ExecStart=/bin/bash -c '\
    source /opt/ros/jazzy/setup.bash && \
    source /home/pi/ros2_ws/install/setup.bash && \
    ros2 launch mmwave_drone_detector detector.launch.py \
        cli_port:=/dev/ttyUSB0 data_port:=/dev/ttyUSB1'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mmwave_detector
sudo systemctl start mmwave_detector

# Check status
sudo systemctl status mmwave_detector
journalctl -u mmwave_detector -f   # live logs
```

## 9. Topics published

| Topic | Type | Rate |
|-------|------|------|
| `/radar/point_cloud` | `sensor_msgs/PointCloud2` | ~18 Hz |
| `/radar/drone_pose` | `geometry_msgs/PoseStamped` | on detection |
| `/radar/drone_velocity` | `geometry_msgs/TwistStamped` | on detection |
| `/radar/drone_detections` | `DroneDetection` | on detection |

## 10. Tune detection (edit config/params.yaml)

For in-flight use with real drones, tighten thresholds:

```yaml
min_snr_db: 7.0          # was 4.0 — reduce false positives
min_doppler_ms: 0.3      # was 0.05 — only fast-moving objects
min_height_m: 0.5        # was -5.0 — must be airborne
min_cluster_points: 2    # was 1 — need at least 2 points
min_confidence: 0.3      # was 0.1 — stricter classification
```
