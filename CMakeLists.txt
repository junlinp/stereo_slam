
cmake_minimum_required(VERSION 3.10)
project(stereo_slam_pybind)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_NEWPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/stereo_slam/stereo_slam_pybind)



