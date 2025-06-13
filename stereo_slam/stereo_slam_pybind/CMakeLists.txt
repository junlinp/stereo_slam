cmake_minimum_required(VERSION 3.14)
project(stereo_slam_pybind LANGUAGES CXX)

find_package(pybind11 REQUIRED)
find_package(octomap REQUIRED)

find_package(superray-octomap REQUIRED)

include_directories(${OCTOMAP_INCLUDE_DIRS})
include_directories(${SUPERRAY_OCTOMAP_INCLUDE_DIRS})

pybind11_add_module(stereo_slam_pybind stereo_slam_pybind.cc)

MESSAGE(Status ${SUPERRAY_OCTOMAP_LIBRARIES})

target_link_libraries(stereo_slam_pybind
    PRIVATE ${OCTOMAP_LIBRARIES} ${SUPERRAY_OCTOMAP_LIBRARIES}
)
