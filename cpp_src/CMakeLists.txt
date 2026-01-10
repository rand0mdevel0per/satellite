cmake_minimum_required(VERSION 3.26)
project(satellite-gpu LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# vcpkg integration
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

# CUDA/HIP detection
include(CheckLanguage)
check_language(CUDA)
check_language(HIP)

option(USE_CUDA "Build with CUDA support" ON)
option(USE_HIP "Build with HIP support" OFF)
option(BUILD_TESTS "Build tests" ON)

if(CMAKE_CUDA_COMPILER AND USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-DSATELLITE_CUDA)
    message(STATUS "CUDA support enabled")
endif()

if(CMAKE_HIP_COMPILER AND USE_HIP)
    enable_language(HIP)
    add_definitions(-DSATELLITE_HIP)
    message(STATUS "HIP support enabled")
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add subdirectories
add_subdirectory(gpu_worker)

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)
endif()
