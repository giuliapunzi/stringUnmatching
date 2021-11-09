enable_testing()

add_library(catch2 INTERFACE)
target_include_directories(catch2 INTERFACE ../libs/catch2)
set(TEST_NAMES "IO")
set(ALL_TEST_SRC "common.cpp")

if(CUDAToolkit_FOUND)
    set(TEST_NAMES ${TEST_NAMES} Matcher Stress Length)
else()
    message(STATUS "CUDA not found, skipping tests")
endif()

foreach(TEST_NAME ${TEST_NAMES})
    string(TOLOWER ${TEST_NAME} TEST_EXE)
    string(APPEND TEST_EXE "_test")
    set(TEST_SRC "${TEST_EXE}.cpp")

    add_executable(${TEST_EXE} common.cpp ${TEST_SRC})
    target_link_libraries(${TEST_EXE} strum_match strum_io catch2)
    add_test(NAME ${TEST_NAME}Test COMMAND ${TEST_EXE})

    set(ALL_TEST_SRC ${ALL_TEST_SRC} ${TEST_SRC})
endforeach()

add_executable(all_tests ${ALL_TEST_SRC})
target_link_libraries(all_tests strum_match strum_io catch2)
add_test(NAME AllTests COMMAND all_tests)
