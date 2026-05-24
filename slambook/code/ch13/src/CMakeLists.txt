add_library(myslam SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
        config.cpp
        feature.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        visual_odometry.cpp
        dataset.cpp)

target_compile_options(myslam PRIVATE -Wall -O3)
target_link_libraries(myslam PUBLIC
        ${THIRD_PARTY_LIBS})
