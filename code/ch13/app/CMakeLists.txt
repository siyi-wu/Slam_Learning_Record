add_executable(run_kitti_stereo run_kitti_stereo.cpp)
target_compile_options(run_kitti_stereo PRIVATE -Wall -O3)
target_link_libraries(run_kitti_stereo PRIVATE myslam ${THIRD_PARTY_LIBS})
