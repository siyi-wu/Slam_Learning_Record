cmake_minimum_required(VERSION 3.10)
project(ch6 LANGUAGES CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(gaussNewton gaussNewton.cpp)
target_include_directories(gaussNewton PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(gaussNewton PRIVATE
    ${OpenCV_LIBS} 
    Eigen3::Eigen
)
