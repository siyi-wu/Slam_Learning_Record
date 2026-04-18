cmake_minimum_required(VERSION 3.10)
project(joinMap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Boost REQUIRED CONFIG)

add_executable(joinMap joinMap.cpp)

target_link_libraries(joinMap PRIVATE 
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    Sophus::Sophus
    OpenGL::GL
    Boost::boost
)