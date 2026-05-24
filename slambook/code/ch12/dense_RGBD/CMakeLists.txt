cmake_minimum_required(VERSION 3.10)
project(dense_rgbd LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# opencv 
find_package(OpenCV REQUIRED)

# eigen 
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

# pcl 
find_package(PCL REQUIRED)
add_definitions(${PCL_DEFINITIONS})

# octomap 
find_package(octomap REQUIRED)

add_executable(pointcloud_mapping pointcloud_mapping.cpp)
target_include_directories(pointcloud_mapping PRIVATE ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_compile_options(pointcloud_mapping PRIVATE -O2)
target_link_libraries(pointcloud_mapping PRIVATE ${OpenCV_LIBS} ${PCL_LIBRARIES} Eigen3::Eigen Boost::headers)

add_executable(octomap_mapping octomap_mapping.cpp)
target_include_directories(octomap_mapping PRIVATE ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS})
target_compile_options(octomap_mapping PRIVATE -O2)
target_link_libraries(octomap_mapping PRIVATE ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES} Eigen3::Eigen Boost::headers)

add_executable(surfel_mapping surfel_mapping.cpp)
target_include_directories(surfel_mapping PRIVATE ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_compile_options(surfel_mapping PRIVATE -O2)
target_link_libraries(surfel_mapping PRIVATE ${OpenCV_LIBS} ${PCL_LIBRARIES} Eigen3::Eigen)
