cmake_minimum_required(VERSION 3.16)
project(StringUnMatching LANGUAGES CXX)


include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -fdiagnostics-color -Wall -Wextra -O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -g -G -DDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75")
    endif()
endif()

include_directories(libs)

add_library(cxxopts INTERFACE)
target_include_directories(cxxopts INTERFACE libs/cxxopts)

include_directories(include)

add_library(strum_io include/io.hpp src/io.cpp)

add_executable(convert src/convert.cpp)
target_link_libraries(convert strum_io cxxopts)

if (CMAKE_CUDA_COMPILER)
    add_library(strum_match
            include/matcher.hpp
            include/cuda_helper.cuh
            src/matcher.cpp
            src/matcher.cu)
    set_target_properties(strum_match
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_executable(hamming src/hamming.cpp)
    target_link_libraries(hamming strum_match strum_io cxxopts)
else()
    message(STATUS "CUDA not found, skipping targets")
endif()

enable_testing()

add_library(catch2 INTERFACE)
target_include_directories(catch2 INTERFACE libs/catch2)

add_subdirectory(test)