cmake_minimum_required(VERSION 3.16)

project(bundle_adjustment LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(g2o REQUIRED)


find_package(Sophus REQUIRED)

add_library(bal_common common.cpp)
target_link_libraries(bal_common PUBLIC Eigen3::Eigen)
target_include_directories(bal_common PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)
target_link_libraries(bundle_adjustment_ceres PRIVATE
    bal_common
    Ceres::ceres
)

add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)
target_link_libraries(bundle_adjustment_g2o PRIVATE
    bal_common
    g2o::core
    g2o::stuff
    g2o::solver_eigen
    Sophus::Sophus
)
