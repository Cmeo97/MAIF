cmake_minimum_required(VERSION 2.8.3)
project(panda_control)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package( CATKIN_DEPENDS
	roscpp
	rospy
	sensor_msgs
	std_msgs
  geometry_msgs
)

###########
## Build ##
###########

# Add AIC library
include_directories(
  include
	${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/classes/AIC.cpp
)


## Declare a C++ executable

add_executable(${PROJECT_NAME}_AIC src/AIC_babbling.cpp)
add_dependencies(${PROJECT_NAME}_AIC ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_AIC
   ${catkin_LIBRARIES} ${PROJECT_NAME}
)


