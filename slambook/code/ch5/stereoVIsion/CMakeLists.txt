cmake_minimum_required(VERSION 3.10)
project(stereoVision LANGUAGES CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(stereoVision stereoVision.cpp)

target_include_directories(stereoVision PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(stereoVision PRIVATE
    ${OpenCV_LIBS} 
    ${Pangolin_LIBRARIES}
    GLEW::GLEW
    OpenGL::GL
    Eigen3::Eigen
)
