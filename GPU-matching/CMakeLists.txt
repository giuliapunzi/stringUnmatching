cmake_minimum_required(VERSION 3.16)
project(StringUnMatching LANGUAGES CUDA CXX)


include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "-fdiagnostics-color -Wall -Wextra -O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DDEBUG")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -DDEBUG")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)


include_directories(libs)

add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE libs/cxxopts)

include_directories(include)

add_library(strum_io include/io.hpp src/io.cpp)

add_executable(convert src/convert.cpp)
target_link_libraries(convert strum_io cxxopts)

add_library(strum_matcher
        include/matcher.hpp
        src/matcher.cpp)

add_library(strum_edit
        include/edit.hpp
        src/edit.cpp
        src/edit.cu)
target_link_libraries(strum_edit strum_matcher strum_io)
set_target_properties(strum_edit
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(strum_hamming
        include/hamming.hpp
        include/utils.cuh
        src/hamming.cpp
        src/hamming.cu)
target_link_libraries(strum_hamming strum_matcher strum_io)
set_target_properties(strum_hamming
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(mindist src/mindist.cpp)
target_link_libraries(mindist strum_hamming strum_edit cxxopts)

add_subdirectory(test)