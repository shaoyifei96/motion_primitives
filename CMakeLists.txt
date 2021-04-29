cmake_minimum_required(VERSION 3.11)
project(motion_primitives)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Find catkin macros and libraries if COMPONENTS list like find_package(catkin
# REQUIRED COMPONENTS xyz) is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp planning_ros_msgs rosbag)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(glog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS timer)
find_package(TBB REQUIRED)

catkin_package(
  INCLUDE_DIRS include LIBRARIES ${PROJECT_NAME} # CATKIN_DEPENDS
                                                 # other_catkin_pkg DEPENDS xcc
)

# ##############################################################################
# Build ##
# ##############################################################################

# Specify libraries to link a library or executable target against
add_library(${PROJECT_NAME} src/motion_primitive_graph.cpp src/graph_search.cpp
                            src/utils.cpp)
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_include_directories(${PROJECT_NAME} PUBLIC include
                                                  ${catkin_INCLUDE_DIRS})
target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  glog::glog
  Boost::timer
  TBB::tbb)

add_executable(${PROJECT_NAME}_graph_search src/graph_search_node.cpp)
target_link_libraries(${PROJECT_NAME}_graph_search PRIVATE ${PROJECT_NAME})

# Mark executables for installation See
# http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS ${PROJECT_NAME}_graph_search
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# Mark libraries for installation See
# http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

# Mark cpp header files for installation
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE)

add_subdirectory(tests)
