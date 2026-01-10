/**
 * @file bcp_kernel.cu
 * @brief CUDA BCP kernel implementation.
 */

#include "types.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace satellite {
namespace gpu {

/**
 * @brief BCP kernel - each warp checks 32 clauses in parallel.
 *
 * @param clause_data Flattened clause data.
 * @param clause_offsets Start offset for each clause.
 * @param num_clauses Number of clauses.
 * @param assignments Variable assignments.
 * @param results Output: 1 if clause is satisfied/unit, 0 if conflict possible.
 */
__global__ void bcp_kernel(
    const int64_t* __restrict__ clause_data,
    const size_t* __restrict__ clause_offsets,
    size_t num_clauses,
    const int8_t* __restrict__ assignments,
    int* __restrict__ results
) {
    // Get warp and lane
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    size_t clause_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (clause_id >= num_clauses) {
        return;
    }
    
    // Get clause bounds
    size_t start = clause_offsets[clause_id];
    size_t end = clause_offsets[clause_id + 1];
    
    bool satisfied = false;
    int unassigned_count = 0;
    
    // Check each literal in the clause
    for (size_t i = start; i < end && !satisfied; ++i) {
        int64_t lit = clause_data[i];
        if (lit == 0) break;
        
        uint64_t var = (lit > 0 ? lit : -lit) - 1;
        int8_t val = assignments[var];
        
        // Check if literal is satisfied
        if ((lit > 0 && val == 1) || (lit < 0 && val == -1)) {
            satisfied = true;
        } else if (val == 0) {
            unassigned_count++;
        }
    }
    
    // Result: 0 = conflict, 1 = satisfied/unit, 2 = unresolved
    int result;
    if (satisfied) {
        result = 1; // Satisfied
    } else if (unassigned_count == 0) {
        result = 0; // Conflict!
    } else if (unassigned_count == 1) {
        result = 1; // Unit clause
    } else {
        result = 2; // Unresolved
    }
    
    results[clause_id] = result;
    
    // Warp-level reduction to detect any conflict
    int has_conflict = (result == 0) ? 1 : 0;
    has_conflict = warp.any(has_conflict);
    
    // Lane 0 could signal conflict here if needed
}

/**
 * @brief Aggregation kernel - reduces results to find conflicts.
 */
__global__ void aggregate_kernel(
    const int* __restrict__ results,
    size_t num_clauses,
    int* __restrict__ conflict_found,
    size_t* __restrict__ conflict_clause
) {
    __shared__ int shared_conflict;
    __shared__ size_t shared_clause;
    
    if (threadIdx.x == 0) {
        shared_conflict = 0;
        shared_clause = 0;
    }
    __syncthreads();
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_clauses && results[idx] == 0) {
        atomicExch(&shared_conflict, 1);
        atomicMin((unsigned long long*)&shared_clause, (unsigned long long)idx);
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0 && shared_conflict) {
        atomicExch(conflict_found, 1);
        atomicMin((unsigned long long*)conflict_clause, (unsigned long long)shared_clause);
    }
}

} // namespace gpu
} // namespace satellite

// C interface for the kernel launcher
extern "C" {

static cudaStream_t g_stream = nullptr;
static int* d_results = nullptr;
static size_t d_results_size = 0;

int init_gpu_device() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return -1;
    }
    
    err = cudaStreamCreate(&g_stream);
    if (err != cudaSuccess) {
        return -1;
    }
    
    return 0;
}

void shutdown_gpu_device() {
    if (d_results) {
        cudaFree(d_results);
        d_results = nullptr;
        d_results_size = 0;
    }
    
    if (g_stream) {
        cudaStreamDestroy(g_stream);
        g_stream = nullptr;
    }
}

int get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

void sync_gpu() {
    cudaDeviceSynchronize();
}

int get_memory_info(size_t* used, size_t* total) {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        return -1;
    }
    *used = total_mem - free_mem;
    *total = total_mem;
    return 0;
}

} // extern "C"
