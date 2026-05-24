cmake_minimum_required(VERSION 3.10)

project(transform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_executable(coordinateTransform coordinateTransform.cpp)
target_link_libraries(coordinateTransform PRIVATE Eigen3::Eigen)


find_package(Pangolin QUIET)

if(Pangolin_FOUND)
    add_executable(plotTrajectory plotTrajectory.cpp)
    target_include_directories(plotTrajectory PRIVATE ${Pangolin_INCLUDE_DIRS})
    target_link_libraries(plotTrajectory PRIVATE ${Pangolin_LIBRARIES})
else()
    message(STATUS "Skipping plotTrajectory: Pangolin was not found.")
endif()
