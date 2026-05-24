cmake_minimum_required(VERSION 3.10)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(trajectory)

find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(trajectory trajectoryError.cpp)

# 头文件路径
target_include_directories(trajectory PRIVATE 
    ${EIGEN3_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

target_link_libraries(trajectory PRIVATE
    Sophus::Sophus
    Eigen3::Eigen
    ${Pangolin_LIBRARIES}
    GLEW::GLEW
    OpenGL::GL
)
