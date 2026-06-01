cmake_minimum_required(VERSION 3.8)
project(mmwave_drone_detector)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Custom message generation
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DroneDetection.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

# Install executable Python nodes
install(PROGRAMS
  mmwave_drone_detector/ros_publisher.py
  mmwave_drone_detector/radar_filter_node.py
  mmwave_drone_detector/range_probe_node.py
  mmwave_drone_detector/radar_range_markers.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python module files so imports work
install(DIRECTORY
  mmwave_drone_detector/
  DESTINATION lib/${PROJECT_NAME}/mmwave_drone_detector
  FILES_MATCHING PATTERN "*.py"
)

# Install launch and config files
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
