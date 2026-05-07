cmake_minimum_required(VERSION 3.10)

project(useEigen)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_executable(eigenmatrix eigenMatrix.cpp)
target_link_libraries(eigenmatrix Eigen3::Eigen)
target_compile_options(eigenmatrix PRIVATE -O3)
