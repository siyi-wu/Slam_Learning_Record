cmake_minimum_required(VERSION 3.10)
project(dense_monocular LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

############### dependencies ######################
# Eigen
find_package(Eigen3 REQUIRED)
# OpenCV
find_package(OpenCV REQUIRED)
# Sophus
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED)

set(THIRD_PARTY_LIBS ${OpenCV_LIBS} Sophus::Sophus Eigen3::Eigen Boost::headers)

add_executable(dense_mapping dense_mapping.cpp)
target_include_directories(dense_mapping PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_options(dense_mapping PRIVATE -march=native -O3)
target_compile_definitions(dense_mapping PRIVATE
    BOOST_TIMER_ENABLE_DEPRECATED
    CV_GRAY2BGR=cv::COLOR_GRAY2BGR
)
target_link_libraries(dense_mapping PRIVATE ${THIRD_PARTY_LIBS})
