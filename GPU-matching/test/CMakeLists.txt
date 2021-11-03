add_executable(io_test common.cpp io_test.cpp)
target_link_libraries(io_test strum_io catch2)
add_test(NAME IO-Test COMMAND io_test)

add_executable(matcher_test common.cpp matcher_test.cpp)
target_link_libraries(matcher_test strum_match strum_io catch2)
add_test(NAME Matcher-Test COMMAND matcher_test)
