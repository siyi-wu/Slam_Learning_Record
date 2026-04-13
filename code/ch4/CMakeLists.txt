cmake_minimum_required(VERSION 3.10)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(usesophus)

find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(usesophus useSophus.cpp)

target_link_libraries(usesophus Sophus::Sophus)
target_link_libraries(usesophus Eigen3::Eigen)

set(CMAKE_CXX_FLAGS "-O3") # 开启release优化