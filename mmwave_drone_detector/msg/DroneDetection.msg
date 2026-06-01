# DroneDetection.msg
# Single detected drone with position, velocity and confidence

std_msgs/Header header

uint32 drone_id                    # Unique ID for this detection
geometry_msgs/Point position       # x, y, z in sensor frame (metres)
geometry_msgs/Vector3 velocity     # vx, vy, vz estimated (m/s)
float32 speed                      # scalar speed magnitude (m/s)
float32 range                      # distance from sensor (metres)
float32 azimuth_deg                # horizontal angle (degrees, 0=forward)
float32 elevation_deg              # vertical angle (degrees, 0=horizontal)
float32 confidence                 # 0.0-1.0 detection confidence
uint32 point_count                 # number of radar points in cluster
float32 snr_mean                   # mean SNR of cluster points (dB)
float32 doppler_mean               # mean doppler of cluster (m/s)
bool is_approaching                # true if moving toward sensor
