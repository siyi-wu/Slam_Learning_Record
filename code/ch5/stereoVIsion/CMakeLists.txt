cmake_minimum_required(VERSION 3.10)
project(stereoVision LANGUAGES CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(stereoVision stereoVision.cpp)

target_include_directories(stereoVision PRIVATE 
    ${Pangolin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(stereoVision PRIVATE
    ${OpenCV_LIBS} 
    GLEW::GLEW # 只能这么写
    ${Pangolin_LIBRARIES}
    OpenGL::GL
    Eigen3::Eigen
)