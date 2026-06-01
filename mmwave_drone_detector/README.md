# mmwave_drone_detector

ROS2 Humble package for detecting and tracking drones using the IWR6843ISK mmWave radar.

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/radar/point_cloud` | `sensor_msgs/PointCloud2` | Filtered radar points (x,y,z,doppler,snr fields) |
| `/radar/drone_pose` | `geometry_msgs/PoseStamped` | Closest detected drone position |
| `/radar/drone_velocity` | `geometry_msgs/TwistStamped` | Closest detected drone velocity |
| `/radar/drone_detections` | `DroneDetection` | Per-drone detection with full metadata |

## Detection Pipeline

```
Raw TLV frames (18 Hz)
    ↓
Filter: SNR > 6dB, range 0.3–8m, height > -0.3m,
        azimuth ±60°, |doppler| > 0.15 m/s
    ↓
DBSCAN clustering (eps=0.8m, min_pts=2)
    ↓
Classify: doppler + point count + SNR + compactness score
    ↓
Track: EMA position/velocity smoothing, consistent IDs
    ↓
Publish on ROS2 topics
```

## Build

```bash
# Install Python dependencies
pip3 install pyserial numpy scipy --break-system-packages

# Place package in your ROS2 workspace
cp -r mmwave_drone_detector ~/ros2_ws/src/

# Build
cd ~/ros2_ws
colcon build --packages-select mmwave_drone_detector
source install/setup.bash
```

## Run

```bash
# With default parameters
ros2 launch mmwave_drone_detector detector.launch.py

# Custom ports
ros2 launch mmwave_drone_detector detector.launch.py \
    cli_port:=/dev/ttyUSB0 data_port:=/dev/ttyUSB1

# Tighter confidence threshold
ros2 launch mmwave_drone_detector detector.launch.py min_confidence:=0.5

# Or run node directly
ros2 run mmwave_drone_detector ros_publisher.py
```

## Monitor Topics

```bash
# Watch drone position in real time
ros2 topic echo /radar/drone_pose

# Watch velocity
ros2 topic echo /radar/drone_velocity

# Watch raw detections with full metadata
ros2 topic echo /radar/drone_detections

# Point cloud in rviz2
rviz2  # Add PointCloud2 display → /radar/point_cloud
```

## Tune Detection Parameters

Edit `config/params.yaml` or pass as launch args:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `min_snr_db` | 6.0 | Raise to reduce false positives |
| `min_doppler_ms` | 0.15 | Raise to only detect fast-moving objects |
| `min_cluster_points` | 2 | Raise to require more radar returns per drone |
| `max_cluster_dist_m` | 0.8 | Raise to merge points spread further apart |
| `min_confidence` | 0.3 | Raise for stricter drone classification |
| `min_height_m` | -0.3 | Raise to ignore ground-level detections |

## TF Frame

The sensor publishes in the `radar` frame. Mount the sensor on your drone with:
- **Y forward** (boresight direction)
- **X right**
- **Z up**

Add a static TF from `base_link` to `radar` to match your mounting position.

## DroneDetection Message Fields

```
uint32 drone_id          — consistent ID across frames
Point  position          — x,y,z in metres (sensor frame)
Vector3 velocity         — estimated vx,vy,vz in m/s
float32 speed            — scalar speed magnitude
float32 range            — distance from sensor
float32 azimuth_deg      — horizontal angle (0=forward, +right)
float32 elevation_deg    — vertical angle (0=horizontal, +up)
float32 confidence       — 0.0–1.0
uint32  point_count      — radar points in cluster
float32 snr_mean         — mean signal-to-noise ratio (dB)
float32 doppler_mean     — mean Doppler velocity (m/s, -=approaching)
bool    is_approaching   — true if drone moving toward sensor
```
