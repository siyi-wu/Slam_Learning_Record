cmake_minimum_required(VERSION 3.10)
project(myslam LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################
# Eigen
find_package(Eigen3 REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# pangolin
find_package(Pangolin REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

# Sophus
find_package(Sophus REQUIRED)

# G2O
find_package(g2o CONFIG QUIET)

if(TARGET g2o::core)
    set(G2O_LIBRARIES
        g2o::core
        g2o::stuff
        g2o::types_sba
        g2o::solver_csparse
        g2o::csparse_extension
    )
else()
    find_package(G2O REQUIRED)
    if(NOT G2O_INCLUDE_DIR OR NOT G2O_CORE_LIBRARY OR NOT G2O_STUFF_LIBRARY OR NOT G2O_TYPES_SBA OR NOT G2O_SOLVER_CSPARSE OR NOT G2O_SOLVER_CSPARSE_EXTENSION)
        message(FATAL_ERROR "g2o was not found. Install g2o with CSparse support, then rerun CMake.")
    endif()
    set(G2O_LIBRARIES
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        ${G2O_TYPES_SBA}
        ${G2O_SOLVER_CSPARSE}
        ${G2O_SOLVER_CSPARSE_EXTENSION}
    )
endif()

# glog
find_package(glog CONFIG QUIET)
if(TARGET glog::glog)
    set(GLOG_LIBRARIES glog::glog)
else()
    find_package(Glog REQUIRED)
endif()

# gtest
find_package(GTest REQUIRED)
if(TARGET GTest::gtest)
    set(GTEST_LIBRARIES GTest::gtest GTest::gtest_main)
else()
    set(GTEST_LIBRARIES ${GTEST_BOTH_LIBRARIES})
endif()

# gflags
find_package(gflags CONFIG REQUIRED)
if(TARGET gflags::gflags)
    set(GFLAGS_LIBRARIES gflags::gflags)
elseif(TARGET gflags)
    set(GFLAGS_LIBRARIES gflags)
endif()

# csparse
find_package(CSparse REQUIRED)

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        Sophus::Sophus
        Eigen3::Eigen
        ${Pangolin_LIBRARIES}
        GLEW::GLEW
        OpenGL::GL
        ${G2O_LIBRARIES}
        ${GTEST_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${CSPARSE_LIBRARY}
        pthread
        )

enable_testing()

############### source and test ######################
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIR}
        ${G2O_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${GFLAGS_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        )
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(app)
