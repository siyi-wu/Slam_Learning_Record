cmake_minimum_required(VERSION 3.10)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(imageBasics)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES
    imageBasics
    undistortImage
)

foreach(src ${SOURCE_FILES})
    add_executable(${src} ${src}.cpp)
    target_include_directories(${src} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${src} PRIVATE ${OpenCV_LIBS})
endforeach(src ${SOURCE_FILES})
