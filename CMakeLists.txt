cmake_minimum_required(VERSION 3.0.2)
project(three_wheel_track)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  gazebo_ros
  xacro
)

## Declare this project a catkin package and list its dependencies
catkin_package(
  CATKIN_DEPENDS
    rospy
    geometry_msgs
    nav_msgs
    gazebo_ros
    xacro
)

## Specify additional locations of header files
## (e.g. ${PROJECT_SOURCE_DIR}/include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install Python scripts so rosrun/roslaunch can find them
catkin_install_python(PROGRAMS
  scripts/controllers.py
  scripts/systems.py
  scripts/simulator.py
  scripts/utilities.py
  scripts/loggers.py
  scripts/kinematic_controller_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install the URDF and launch directories
install(DIRECTORY
  urdf
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

