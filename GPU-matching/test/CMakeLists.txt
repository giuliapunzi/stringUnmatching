add_executable(io_tests common.cpp io_test.cpp)
target_link_libraries(io_tests strum_io catch2)
add_test(NAME IO-Test COMMAND io_tests)
