cmake_minimum_required(VERSION 3.10)
project(ch2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 添加可执行文件
# add_executable(libtest main.cpp)

# 要编译库
# add_library(hello libtest.cpp)
add_library(hello_shared SHARED libtest.cpp)
target_include_directories(hello_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 链接库

# 先生成一个可执行文件
add_executable(usehello usehello.cpp)
target_link_libraries(usehello PRIVATE hello_shared) # 指向上面的编译库
