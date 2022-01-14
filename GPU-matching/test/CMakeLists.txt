enable_testing()

add_library(catch2 INTERFACE)
target_include_directories(catch2 INTERFACE ../libs/catch2)

add_executable(io_test common.cpp io_test.cpp)
target_link_libraries(io_test strum_io catch2)
add_test(NAME IO-Test COMMAND io_test)

add_executable(hamming_test common.cpp hamming_test.cpp)
target_link_libraries(hamming_test strum_hamming catch2)
add_test(NAME Hamming-Test COMMAND hamming_test)

add_executable(edit_test common.cpp edit_test.cpp)
target_link_libraries(edit_test strum_edit catch2)
add_test(NAME Edit-Test COMMAND edit_test)

add_executable(stress_test common.cpp stress_test.cpp)
target_link_libraries(stress_test strum_hamming strum_edit catch2)
add_test(NAME Stress-Test COMMAND stress_test)

add_executable(reduction_test common.cpp reduction_test.cpp)
target_link_libraries(reduction_test strum_hamming strum_edit catch2)
add_test(NAME Reduction-Test COMMAND reduction_test)
