cmake_minimum_required(VERSION 2.8.3)
project(shadow_regression)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  moveit_core
  moveit_ros_planning
  roscpp
  moveit_ros_move_group
  bio_ik
  moveit_ros_planning_interface
)
find_package( OpenCV REQUIRED )
catkin_package(CATKIN_DEPENDS moveit_core moveit_ros_planning bio_ik moveit_ros_planning roscpp)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(bioik_dataset src/bioik_dataset.cpp)
add_dependencies(bioik_dataset ${catkin_EXPORTED_TARGETS})
target_link_libraries(bioik_dataset ${catkin_LIBRARIES} ${OpenCV_LIBS})
