cmake_minimum_required(VERSION 3.10)

project(useEigen)

find_package(Eigen3 REQUIRED)

add_executable(eigenmatrix eigenmatrix.cpp)
target_link_libraries(eigenmatrix Eigen3::Eigen)

set(CMAKE_CXX_FLAGS "-O3") # 开启release优化