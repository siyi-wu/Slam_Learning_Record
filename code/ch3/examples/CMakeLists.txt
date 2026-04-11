cmake_minimum_required(VERSION 3.10)

project(transform)

set(CMAKE_CXX_FLAGS "-O3")

find_package(Eigen3 REQUIRED)

add_executable(coordinateTransform coordinateTransform.cpp)
target_link_libraries(coordinateTransform Eigen3::Eigen)


find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})





# cmake_minimum_required(VERSION 3.10)

# project(transform)

# find_package(Eigen3 REQUIRED)

# add_executable(ctransform coodinateTransform.cpp)
# target_link_libraries(ctransform Eigen3::Eigen)

# set(CMAKE_CXX_FLAGS "-O3") # 开启release优化