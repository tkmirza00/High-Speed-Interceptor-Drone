"""Mission configuration for the radar-guided offboard controller."""

# Approximate target location in MAVROS local ENU.
# Put your deliberately error-corrupted target estimate here.
TARGET_X = 0.0       # ENU East [m]
TARGET_Y = -40.0     # ENU North [m]
TARGET_Z = 4.0       # ENU Up [m]

# Candidate must be within this radius of expected target local position.
# Use 15–25 m depending on your injected target-position error.
EXPECTED_TARGET_GATE_M = 15.0

# Coarse approach speed toward approximate target.
COARSE_TARGET_VELOCITY = 5.0  # m/s. Use lower for first real tests.

# Radar acquisition and stopping.
RADAR_ACQUIRE_RANGE_M = 50.0  # switch to radar-guided if accepted candidate is within this range
STOP_RANGE_M = 5.0            # stop if a fresh radar candidate is inside this range

# Candidate freshness.
# Used only to decide whether fresh radar range can trigger an immediate stop.
RADAR_TIMEOUT_S = 0.5

# Committed target behaviour.
# If radar disappears, continue toward the last accepted local target until:
#   1) reached committed target, or
#   2) committed target becomes too old.
COMMITTED_TARGET_REACHED_M = 3.0
MAX_COMMITTED_TARGET_AGE_S = 5.0

# Radar-guided speed schedule toward committed target:
# distance = RADAR_ACQUIRE_RANGE_M -> RADAR_GUIDED_MAX_SPEED
# distance = COMMITTED_TARGET_REACHED_M -> near 0
RADAR_GUIDED_MAX_SPEED = 5.0  # m/s
RADAR_GUIDED_MIN_SPEED = 2.0  # m/s while outside stop/reached distance

# Waypoint behaviour if no radar target is accepted.
DECEL_DISTANCE = 10.0
ARRIVAL_RADIUS = 5.0

# Radar mounting angle.
RADAR_MOUNT_PITCH_DEG = -60.0

RATE = 20  # Hz
