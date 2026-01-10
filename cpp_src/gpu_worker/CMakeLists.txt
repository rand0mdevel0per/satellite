# GPU Worker Library

set(GPU_WORKER_SOURCES
    src/gpu_worker.cpp
)

set(GPU_WORKER_HEADERS
    include/gpu_worker.h
    include/types.h
)

# Main library
add_library(gpu_worker SHARED
    ${GPU_WORKER_SOURCES}
    ${GPU_WORKER_HEADERS}
)

target_include_directories(gpu_worker PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# CUDA sources
if(CMAKE_CUDA_COMPILER AND USE_CUDA)
    target_sources(gpu_worker PRIVATE
        src/cuda/mpmc_gpu.cu
        src/cuda/bcp_kernel.cu
    )
    set_target_properties(gpu_worker PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# HIP sources
if(CMAKE_HIP_COMPILER AND USE_HIP)
    target_sources(gpu_worker PRIVATE
        src/hip/mpmc_gpu.hip
        src/hip/bcp_kernel.hip
    )
endif()

# Tests
if(BUILD_TESTS)
    add_executable(test_gpu_worker
        tests/test_gpu_worker.cpp
    )
    target_link_libraries(test_gpu_worker
        gpu_worker
        GTest::gtest_main
    )
    add_test(NAME GpuWorkerTests COMMAND test_gpu_worker)
endif()

# Install
install(TARGETS gpu_worker
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
