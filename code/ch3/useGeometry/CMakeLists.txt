cmake_minimum_required(VERSION 3.10)

project(Geometry)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_executable(usegeometry useGeometry.cpp)
target_link_libraries(usegeometry Eigen3::Eigen)
target_compile_options(usegeometry PRIVATE -O3)
