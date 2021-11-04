enable_testing()

add_library(catch2 INTERFACE)
target_include_directories(catch2 INTERFACE ../libs/catch2)

add_executable(io_test common.cpp io_test.cpp)
target_link_libraries(io_test strum_io catch2)
add_test(NAME IO-Test COMMAND io_test)

if (CUDA_FOUND)
    add_executable(matcher_test common.cpp matcher_test.cpp)
    target_link_libraries(matcher_test strum_match strum_io catch2)
    add_test(NAME Matcher-Test COMMAND matcher_test)
endif()