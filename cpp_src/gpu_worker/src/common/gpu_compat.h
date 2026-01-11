/**
 * @file gpu_compat.h
 * @brief Portable GPU abstraction layer for CUDA and HIP.
 * 
 * This header provides macros and type aliases to write GPU code that compiles
 * for both NVIDIA CUDA and AMD HIP/ROCm platforms.
 */

#pragma once

// =============================================================================
// Platform Detection
// =============================================================================

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
    #define SATELLITE_GPU_HIP 1
    #define SATELLITE_GPU_CUDA 0
#elif defined(__CUDACC__) || defined(__NVCC__)
    #define SATELLITE_GPU_CUDA 1
    #define SATELLITE_GPU_HIP 0
#else
    #error "No supported GPU platform detected (CUDA or HIP)"
#endif

// =============================================================================
// Runtime API Abstraction
// =============================================================================

#if SATELLITE_GPU_HIP
    #include <hip/hip_runtime.h>
    #include <hip/hip_cooperative_groups.h>
    
    // Memory management
    #define gpuMalloc hipMalloc
    #define gpuFree hipFree
    #define gpuMemcpy hipMemcpy
    #define gpuMemcpyAsync hipMemcpyAsync
    #define gpuMemset hipMemset
    #define gpuMemGetInfo hipMemGetInfo
    
    // Memory copy directions
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
    
    // Device management
    #define gpuSetDevice hipSetDevice
    #define gpuGetDeviceCount hipGetDeviceCount
    #define gpuDeviceSynchronize hipDeviceSynchronize
    
    // Streams
    #define gpuStream_t hipStream_t
    #define gpuStreamCreate hipStreamCreate
    #define gpuStreamDestroy hipStreamDestroy
    #define gpuStreamSynchronize hipStreamSynchronize
    
    // Error handling
    #define gpuError_t hipError_t
    #define gpuSuccess hipSuccess
    #define gpuGetLastError hipGetLastError
    #define gpuGetErrorString hipGetErrorString
    
    // Atomics (device)
    #define gpuAtomicAdd atomicAdd
    #define gpuAtomicSub atomicSub
    #define gpuAtomicCAS atomicCAS
    
    // Cooperative groups
    namespace gpu_cg = cooperative_groups;
    
#else // CUDA
    #include <cuda_runtime.h>
    #include <cooperative_groups.h>
    
    // Memory management
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemcpy cudaMemcpy
    #define gpuMemcpyAsync cudaMemcpyAsync
    #define gpuMemset cudaMemset
    #define gpuMemGetInfo cudaMemGetInfo
    
    // Memory copy directions
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
    
    // Device management
    #define gpuSetDevice cudaSetDevice
    #define gpuGetDeviceCount cudaGetDeviceCount
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    
    // Streams
    #define gpuStream_t cudaStream_t
    #define gpuStreamCreate cudaStreamCreate
    #define gpuStreamDestroy cudaStreamDestroy
    #define gpuStreamSynchronize cudaStreamSynchronize
    
    // Error handling
    #define gpuError_t cudaError_t
    #define gpuSuccess cudaSuccess
    #define gpuGetLastError cudaGetLastError
    #define gpuGetErrorString cudaGetErrorString
    
    // Atomics (already available in CUDA)
    #define gpuAtomicAdd atomicAdd
    #define gpuAtomicSub atomicSub
    #define gpuAtomicCAS atomicCAS
    
    // Cooperative groups
    namespace gpu_cg = cooperative_groups;
    
#endif

// =============================================================================
// Common Types
// =============================================================================

#include <cstdint>

namespace satellite {
namespace gpu {

// Portable clock function
__device__ __forceinline__ uint64_t gpu_clock64() {
#if SATELLITE_GPU_HIP
    return clock64();
#else
    return clock64();
#endif
}

// Portable nanosleep (CUDA 11.6+, HIP has __builtin_amdgcn_s_sleep)
__device__ __forceinline__ void gpu_nanosleep(unsigned ns) {
#if SATELLITE_GPU_CUDA
    __nanosleep(ns);
#else
    // HIP: Use s_sleep with approximate conversion (1 cycle ≈ 1ns at 1GHz)
    for (unsigned i = 0; i < ns / 100; ++i) {
        __builtin_amdgcn_s_sleep(1);
    }
#endif
}

} // namespace gpu
} // namespace satellite
