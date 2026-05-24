cmake_minimum_required(VERSION 3.10)
project(loop_closure LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# opencv 
find_package(OpenCV REQUIRED)
find_package(DBoW3 REQUIRED)

set(CH11_THIRD_PARTY_LIBS ${OpenCV_LIBS} ${DBoW3_LIBRARIES})
if(TARGET DBoW3::DBoW3)
    set(CH11_THIRD_PARTY_LIBS ${OpenCV_LIBS} DBoW3::DBoW3)
endif()

foreach(target feature_training loop_closure)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS} ${DBoW3_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE -O3)
    target_link_libraries(${target} PRIVATE ${CH11_THIRD_PARTY_LIBS})
endforeach()

add_executable(gen_vocab gen_vocab_large.cpp)
target_include_directories(gen_vocab PRIVATE ${OpenCV_INCLUDE_DIRS} ${DBoW3_INCLUDE_DIRS})
target_compile_options(gen_vocab PRIVATE -O3)
target_link_libraries(gen_vocab PRIVATE ${CH11_THIRD_PARTY_LIBS})
