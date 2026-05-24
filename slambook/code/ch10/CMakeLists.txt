cmake_minimum_required(VERSION 3.10)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Eigen
find_package(Eigen3 REQUIRED)

# sophus 
find_package(Sophus REQUIRED)

# g2o 
find_package(g2o CONFIG QUIET)

if(TARGET g2o::core)
    set(G2O_SE3_LIBRARIES g2o::core g2o::stuff g2o::types_slam3d g2o::solver_eigen)
    set(G2O_LIE_LIBRARIES g2o::core g2o::stuff g2o::solver_eigen)
else()
    find_package(G2O REQUIRED)
    if(NOT G2O_INCLUDE_DIR OR NOT G2O_CORE_LIBRARY OR NOT G2O_STUFF_LIBRARY OR NOT G2O_SOLVER_EIGEN OR NOT G2O_TYPES_SLAM3D)
        message(FATAL_ERROR "g2o was not found. Install g2o, then rerun CMake.")
    endif()
    set(G2O_SE3_LIBRARIES
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        ${G2O_TYPES_SLAM3D}
        ${G2O_SOLVER_EIGEN}
    )
    set(G2O_LIE_LIBRARIES
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        ${G2O_SOLVER_EIGEN}
    )
endif()

include_directories(${G2O_INCLUDE_DIR} ${G2O_INCLUDE_DIRS})

add_executable(pose_graph_g2o_SE3 pose_graph_g2o_SE3.cpp)
target_compile_options(pose_graph_g2o_SE3 PRIVATE -O2)
target_link_libraries(pose_graph_g2o_SE3 PRIVATE
    Eigen3::Eigen
    ${G2O_SE3_LIBRARIES}
)

add_executable(pose_graph_g2o_lie pose_graph_g2o_lie_algebra.cpp)
target_compile_options(pose_graph_g2o_lie PRIVATE -O2)
target_link_libraries(pose_graph_g2o_lie PRIVATE
    Eigen3::Eigen
    Sophus::Sophus
    ${G2O_LIE_LIBRARIES}
)
