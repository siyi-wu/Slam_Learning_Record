SET(TEST_SOURCES test_triangulation)

FOREACH (test_src ${TEST_SOURCES})
    ADD_EXECUTABLE(${test_src} ${test_src}.cpp)
    TARGET_COMPILE_OPTIONS(${test_src} PRIVATE -Wall -O3)
    TARGET_LINK_LIBRARIES(${test_src} PRIVATE ${THIRD_PARTY_LIBS} myslam)
    ADD_TEST(${test_src} ${test_src})
ENDFOREACH (test_src)
