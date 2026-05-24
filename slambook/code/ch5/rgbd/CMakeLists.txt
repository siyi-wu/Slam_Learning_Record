cmake_minimum_required(VERSION 3.10)
project(joinMap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

add_executable(joinMap joinMap.cpp)
target_include_directories(joinMap PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

target_link_libraries(joinMap PRIVATE 
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    Sophus::Sophus
    Eigen3::Eigen
    OpenGL::GL
    Boost::headers
)
