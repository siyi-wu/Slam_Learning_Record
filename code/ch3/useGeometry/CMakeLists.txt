cmake_minimum_required(VERSION 3.10)

project(Geometry)

find_package(Eigen3 REQUIRED)

add_executable(usegeometry useGeometry.cpp)
target_link_libraries(usegeometry Eigen3::Eigen)

set(CMAKE_CXX_FLAGS "-O3") # 开启release优化