cmake_minimum_required(VERSION 2.8.3)
project(lane_follower)

find_package(catkin REQUIRED)

catkin_package()

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
   scripts/lane_follower.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

