cmake_minimum_required(VERSION 3.10)
project(ch7 LANGUAGES CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(g2o CONFIG QUIET)

find_path(SOPHUS_INCLUDE_DIR NAMES sophus/se3.hpp
    HINTS
        ${Sophus_DIR}/..
        ${Sophus_DIR}/../..
        /usr/local/include
        /usr/include
)

if(NOT SOPHUS_INCLUDE_DIR)
    message(FATAL_ERROR "Sophus headers were not found. Install Sophus or set Sophus_DIR/CMAKE_PREFIX_PATH to a valid Sophus installation.")
endif()

if(TARGET g2o::core)
    set(G2O_LIBRARIES g2o::core g2o::stuff g2o::solver_dense)
else()
    find_path(G2O_INCLUDE_DIR NAMES g2o/core/base_vertex.h)
    find_library(G2O_CORE_LIBRARY NAMES g2o_core)
    find_library(G2O_STUFF_LIBRARY NAMES g2o_stuff)
    find_library(G2O_SOLVER_DENSE NAMES g2o_solver_dense)

    if(NOT G2O_INCLUDE_DIR OR NOT G2O_CORE_LIBRARY OR NOT G2O_STUFF_LIBRARY OR NOT G2O_SOLVER_DENSE)
        message(FATAL_ERROR "g2o was not found. Install g2o, then rerun CMake.")
    endif()

    set(G2O_LIBRARIES
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        ${G2O_SOLVER_DENSE}
    )
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIR} ${SOPHUS_INCLUDE_DIR})

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv PRIVATE ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d PRIVATE ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation PRIVATE ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d PRIVATE
    ${OpenCV_LIBS} 
    ${G2O_LIBRARIES}
    Eigen3::Eigen
    Sophus::Sophus
)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d PRIVATE
    ${OpenCV_LIBS} 
    ${G2O_LIBRARIES}
    Eigen3::Eigen
    Sophus::Sophus
)
