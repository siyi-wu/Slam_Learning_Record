cmake_minimum_required(VERSION 3.16)

project(bundle_adjustment LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(g2o CONFIG QUIET)

if(TARGET g2o::core)
    set(G2O_BA_LIBRARIES g2o::core g2o::stuff g2o::solver_eigen)
else()
    find_path(G2O_INCLUDE_DIR NAMES g2o/core/base_vertex.h)
    find_library(G2O_CORE_LIBRARY NAMES g2o_core)
    find_library(G2O_STUFF_LIBRARY NAMES g2o_stuff)
    find_library(G2O_SOLVER_EIGEN NAMES g2o_solver_eigen)

    if(NOT G2O_INCLUDE_DIR OR NOT G2O_CORE_LIBRARY OR NOT G2O_STUFF_LIBRARY OR NOT G2O_SOLVER_EIGEN)
        message(FATAL_ERROR "g2o was not found. Install g2o, then rerun CMake.")
    endif()

    set(G2O_BA_LIBRARIES
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        ${G2O_SOLVER_EIGEN}
    )
endif()

add_library(bal_common common.cpp)
target_link_libraries(bal_common PUBLIC Eigen3::Eigen)
target_include_directories(bal_common PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${G2O_INCLUDE_DIR}
)

add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)
target_link_libraries(bundle_adjustment_ceres PRIVATE
    bal_common
    Ceres::ceres
)

add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)
target_link_libraries(bundle_adjustment_g2o PRIVATE
    bal_common
    ${G2O_BA_LIBRARIES}
    Sophus::Sophus
)
