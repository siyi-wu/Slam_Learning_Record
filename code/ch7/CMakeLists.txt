cmake_minimum_required(VERSION 3.10)
project(ch7 LANGUAGES CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

    set(G2O_INCLUDE_DIRS "/opt/homebrew/include")
    set(G2O_LIBRARIES 
        "/opt/homebrew/lib/libg2o_core.dylib"
        "/opt/homebrew/lib/libg2o_stuff.dylib"
        "/opt/homebrew/lib/libg2o_types_sba.dylib"
        "/opt/homebrew/lib/libg2o_solver_dense.dylib"
    )
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

# 打印调试信息
message(STATUS "G2O Include: ${G2O_INCLUDE_DIRS}")
message(STATUS "G2O Libs: ${G2O_LIBRARIES}")

include_directories(${G2O_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/opt/homebrew/include/eigen3")
include_directories("/opt/homebrew/include")

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
)

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d PRIVATE
    ${OpenCV_LIBS} 
    ${G2O_LIBRARIES}
    Eigen3::Eigen
    Sophus::Sophus
)