cmake_minimum_required(VERSION 3.10)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O2")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Homebrew installs headers and libraries under /opt/homebrew on Apple Silicon.
include_directories("/opt/homebrew/include" "/opt/homebrew/include/eigen3" "/usr/local/include")
link_directories("/opt/homebrew/lib" "/usr/local/lib")

# sophus 
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# g2o 
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

add_executable(pose_graph_g2o_SE3 pose_graph_g2o_SE3.cpp)
target_link_libraries(pose_graph_g2o_SE3
        g2o_core g2o_stuff g2o_types_slam3d ${CHOLMOD_LIBRARIES}
        )

add_executable(pose_graph_g2o_lie pose_graph_g2o_lie_algebra.cpp)
target_link_libraries(pose_graph_g2o_lie
        g2o_core g2o_stuff
        ${CHOLMOD_LIBRARIES}
        ${Sophus_LIBRARIES}
        )
