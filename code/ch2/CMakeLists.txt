cmake_minimum_required(VERSION 3.5)

# 声明cmake工程
# project(libtest)

# 添加可执行文件
# add_executable(libtest main.cpp)

# 要编译库
# add_library(hello libtest.cpp)
add_library(hello_shared SHARED libtest.cpp)

# 链接库

# 先生成一个可执行文件
add_executable(usehello usehello.cpp)
target_link_libraries(usehello hello_shared) # 指向上面的编译库