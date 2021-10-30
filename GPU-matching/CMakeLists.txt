cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(StringUnMatching LANGUAGES CXX)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "-mbmi2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -fdiagnostics-color -Wall -Wextra -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_FLAGS_DEBUG "-G")

include_directories(include)

add_library(strum_io include/io.hpp src/io.cpp)

add_executable(convert src/convert.cpp)
target_link_libraries(convert strum_io)

enable_testing()
add_subdirectory(test)