/**
 * @file bcp_kernel.cu
 * @brief CUDA BCP kernel implementation with persistent kernel loop and work stealing.
 * 
 * This file implements:
 * - Persistent BCP kernel that runs continuously processing jobs from GPU queue
 * - Intra-warp work stealing for load balancing
 * - Warp-cooperative clause checking
 * - Proper aggregation of results with warp voting
 */

#include "../common/types.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

namespace satellite {
namespace gpu {

// PRNG Xorshift for random queue selection
__device__ __forceinline__ uint32_t xorshift32(uint32_t state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// Result codes
constexpr int CLAUSE_SATISFIED = 1;
constexpr int CLAUSE_CONFLICT = 0;
constexpr int CLAUSE_UNRESOLVED = 2;
constexpr int CLAUSE_UNIT = 3;

/**
 * @brief Check if a clause is satisfied, conflicting, unit, or unresolved.
 * 
 * @param clause_id Index of clause to check
 * @param data Flattened literal data
 * @param offsets Clause boundary offsets
 * @param assigns Variable assignments (-1=false, 0=unassigned, 1=true)
 * @return Result code
 */
__device__ int check_clause(
    size_t clause_id, 
    const int64_t* __restrict__ data, 
    const size_t* __restrict__ offsets, 
    const int8_t* __restrict__ assigns
) {
    size_t start = offsets[clause_id];
    size_t end = offsets[clause_id + 1];
    int unassigned_count = 0;
    int64_t unit_lit = 0;
    
    for (size_t i = start; i < end; ++i) {
        int64_t lit = data[i];
        if (lit == 0) break;
        
        uint64_t var = (lit > 0 ? lit : -lit) - 1;
        int8_t val = assigns[var];
        
        // Check if literal is satisfied
        if ((lit > 0 && val == 1) || (lit < 0 && val == -1)) {
            return CLAUSE_SATISFIED;
        }
        
        // Count unassigned literals
        if (val == 0) {
            unassigned_count++;
            unit_lit = lit;
        }
    }
    
    if (unassigned_count == 0) return CLAUSE_CONFLICT;
    if (unassigned_count == 1) return CLAUSE_UNIT;
    return CLAUSE_UNRESOLVED;
}

/**
 * @brief Aggregation result structure for warp-level reduction.
 */
struct WarpAggResult {
    int conflict_count;
    int unit_count;
    int satisfied_count;
    
    __device__ WarpAggResult() : conflict_count(0), unit_count(0), satisfied_count(0) {}
    
    __device__ WarpAggResult operator+(const WarpAggResult& other) const {
        WarpAggResult r;
        r.conflict_count = conflict_count + other.conflict_count;
        r.unit_count = unit_count + other.unit_count;
        r.satisfied_count = satisfied_count + other.satisfied_count;
        return r;
    }
};

/**
 * @brief Aggregate results within a warp using warp voting.
 * 
 * @param warp Warp cooperative group
 * @param result Individual thread's clause check result
 * @return Aggregated result (only valid on lane 0)
 */
__device__ WarpAggResult aggregate_warp_results(
    cg::thread_block_tile<32>& warp,
    int result
) {
    // Count each result type using warp voting
    uint32_t conflict_mask = warp.ballot(result == CLAUSE_CONFLICT);
    uint32_t unit_mask = warp.ballot(result == CLAUSE_UNIT);
    uint32_t satisfied_mask = warp.ballot(result == CLAUSE_SATISFIED);
    
    WarpAggResult agg;
    agg.conflict_count = __popc(conflict_mask);
    agg.unit_count = __popc(unit_mask);
    agg.satisfied_count = __popc(satisfied_mask);
    
    return agg;
}

/**
 * @brief Shared memory structure for block-level aggregation.
 */
struct BlockAggState {
    int total_conflicts;
    int total_units;
    int total_satisfied;
    int total_processed;
};

/**
 * @brief Persistent BCP kernel with work stealing and proper aggregation.
 *
 * @param queue MPMC priority queue
 * @param clause_data Flattened clause literal data
 * @param clause_offsets Start offset for each clause
 * @param num_clauses Total number of clauses
 * @param assignments Variable assignments
 * @param results Per-clause result buffer
 * @param agg_results Block-level aggregated results
 * @param stop_signal Flag to stop the kernel
 */
__global__ void bcp_persistent_kernel(
    GpuQueue* queue,
    const int64_t* __restrict__ clause_data,
    const size_t* __restrict__ clause_offsets,
    size_t num_clauses,
    const int8_t* __restrict__ assignments,
    int* __restrict__ results,
    BlockAggState* __restrict__ agg_results,
    volatile int* stop_signal
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Block-level aggregation in shared memory
    __shared__ BlockAggState block_agg;
    if (threadIdx.x == 0) {
        block_agg.total_conflicts = 0;
        block_agg.total_units = 0;
        block_agg.total_satisfied = 0;
        block_agg.total_processed = 0;
    }
    block.sync();
    
    // Initial RNG state based on thread ID and clock
    uint32_t rng_state = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x + clock64());
    
    while (*stop_signal == 0) {
        // 1. Select Queue Priority (Lane 0 only)
        rng_state = xorshift32(rng_state);
        uint32_t priority = 0;
        
        // Weighted sampling: 0=50%, 1=30%, 2=15%, 3=5%
        if (warp.thread_rank() == 0) {
            uint32_t r = rng_state % 100;
            if (r < 50) priority = 0;
            else if (r < 80) priority = 1;
            else if (r < 95) priority = 2;
            else priority = 3;
        }
        priority = warp.shfl(priority, 0);
        
        // 2. Pop Job (Lane 0 only)
        QueueNode job;
        bool success = false;
        
        if (warp.thread_rank() == 0) {
            success = queue->pop(priority, &job);
        }
        
        // Broadcast success to entire warp
        uint32_t success_mask = warp.ballot(success);
        if (success_mask == 0) {
            // No job available, spin/backoff
            __nanosleep(100);
            continue;
        }
        
        // Broadcast job details using 32-bit shuffles
        uint32_t* job_ptr = (uint32_t*)&job;
        #pragma unroll
        for (int i = 0; i < sizeof(QueueNode)/4; ++i) {
            job_ptr[i] = warp.shfl(job_ptr[i], 0);
        }
        
        // 3. Intra-Warp Work Stealing
        uint32_t work_size = job.clause_end - job.clause_start;
        bool is_idle = warp.thread_rank() >= work_size;
        uint32_t idle_mask = warp.ballot(is_idle);
        
        QueueNode stolen_job;
        bool has_stolen = false;
        int stealer_rank = (idle_mask != 0) ? (__ffs(idle_mask) - 1) : -1;
        
        if (is_idle && warp.thread_rank() == stealer_rank) {
            // Attempt to steal work from another queue
            uint32_t r = xorshift32(rng_state) % 100;
            uint32_t steal_priority = (r < 70) ? 0 : 1;
            
            QueueNode temp_job;
            if (queue->pop(steal_priority, &temp_job)) {
                if (temp_job.clause_end > temp_job.clause_start) {
                    // Take first unit of work
                    stolen_job = temp_job;
                    stolen_job.clause_end = temp_job.clause_start + 1;
                    
                    // Push remainder back
                    if (temp_job.clause_end > temp_job.clause_start + 1) {
                        queue->push(steal_priority, temp_job.job_id, temp_job.branch_id,
                                   temp_job.clause_start + 1, temp_job.clause_end);
                    }
                    has_stolen = true;
                }
            }
        }
        
        // 4. Execute clause checking
        int result = CLAUSE_UNRESOLVED;
        size_t clause_idx = 0;
        bool valid_work = false;
        
        if (!is_idle && warp.thread_rank() < work_size) {
            // Process from main job
            clause_idx = job.clause_start + warp.thread_rank();
            if (clause_idx < num_clauses) {
                result = check_clause(clause_idx, clause_data, clause_offsets, assignments);
                results[clause_idx] = result;
                valid_work = true;
            }
        } else if (has_stolen && warp.thread_rank() == stealer_rank) {
            // Process stolen work
            clause_idx = stolen_job.clause_start;
            if (clause_idx < num_clauses) {
                result = check_clause(clause_idx, clause_data, clause_offsets, assignments);
                results[clause_idx] = result;
                valid_work = true;
            }
        }
        
        // 5. Warp-level aggregation using warp voting
        WarpAggResult warp_agg = aggregate_warp_results(warp, valid_work ? result : -1);
        
        // 6. Block-level aggregation (lane 0 of each warp atomically updates)
        if (warp.thread_rank() == 0) {
            atomicAdd(&block_agg.total_conflicts, warp_agg.conflict_count);
            atomicAdd(&block_agg.total_units, warp_agg.unit_count);
            atomicAdd(&block_agg.total_satisfied, warp_agg.satisfied_count);
            atomicAdd(&block_agg.total_processed, __popc(warp.ballot(valid_work)));
        }
        
        // 7. Periodic flush to global memory (every N iterations or on conflict)
        if (warp_agg.conflict_count > 0) {
            // Early exit on conflict - flush immediately
            block.sync();
            if (threadIdx.x == 0 && agg_results != nullptr) {
                atomicAdd(&agg_results[blockIdx.x].total_conflicts, block_agg.total_conflicts);
                atomicAdd(&agg_results[blockIdx.x].total_units, block_agg.total_units);
                atomicAdd(&agg_results[blockIdx.x].total_satisfied, block_agg.total_satisfied);
                atomicAdd(&agg_results[blockIdx.x].total_processed, block_agg.total_processed);
                
                // Reset local counters
                block_agg.total_conflicts = 0;
                block_agg.total_units = 0;
                block_agg.total_satisfied = 0;
                block_agg.total_processed = 0;
            }
            block.sync();
        }
    }
    
    // Final flush on kernel exit
    block.sync();
    if (threadIdx.x == 0 && agg_results != nullptr) {
        atomicAdd(&agg_results[blockIdx.x].total_conflicts, block_agg.total_conflicts);
        atomicAdd(&agg_results[blockIdx.x].total_units, block_agg.total_units);
        atomicAdd(&agg_results[blockIdx.x].total_satisfied, block_agg.total_satisfied);
        atomicAdd(&agg_results[blockIdx.x].total_processed, block_agg.total_processed);
    }
}

// Proxy kernel to push job from host
__global__ void push_proxy_kernel(
    GpuQueue* queue, 
    uint32_t priority, 
    uint64_t job_id, 
    uint64_t branch_id, 
    uint32_t start, 
    uint32_t end
) {
    if (threadIdx.x == 0) {
        queue->push(priority, job_id, branch_id, start, end);
    }
}

} // namespace gpu
} // namespace satellite

// =============================================================================
// C Interface
// =============================================================================

extern "C" {

static cudaStream_t g_stream = nullptr;
static int* d_results = nullptr;
static int* d_stop_signal = nullptr;
static satellite::gpu::GpuQueue* d_queue = nullptr;
static satellite::gpu::BlockAggState* d_agg_results = nullptr;
static constexpr int NUM_BLOCKS = 128;

int init_gpu_device() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) return -1;
    
    err = cudaStreamCreate(&g_stream);
    if (err != cudaSuccess) return -1;
    
    d_queue = satellite::gpu::GpuQueue::create();
    if (!d_queue) return -1;
    
    err = cudaMalloc(&d_stop_signal, sizeof(int));
    if (err != cudaSuccess) return -1;
    cudaMemset(d_stop_signal, 0, sizeof(int));
    
    err = cudaMalloc(&d_agg_results, sizeof(satellite::gpu::BlockAggState) * NUM_BLOCKS);
    if (err != cudaSuccess) return -1;
    cudaMemset(d_agg_results, 0, sizeof(satellite::gpu::BlockAggState) * NUM_BLOCKS);
    
    return 0;
}

void launch_persistent_kernel(
    const int64_t* clause_data,
    const size_t* clause_offsets,
    size_t num_clauses,
    const int8_t* assignments,
    int* results
) {
    d_results = results;

    satellite::gpu::bcp_persistent_kernel<<<NUM_BLOCKS, 256, 0, g_stream>>>(
        d_queue, clause_data, clause_offsets, num_clauses, 
        assignments, results, d_agg_results, d_stop_signal
    );
}

void submit_job(uint32_t priority, uint64_t job_id, uint64_t branch_id, uint32_t start, uint32_t end) {
    satellite::gpu::push_proxy_kernel<<<1, 1, 0, g_stream>>>(
        d_queue, priority, job_id, branch_id, start, end
    );
}

int read_results(int* host_buffer, size_t start_idx, size_t count) {
    if (!d_results) return -1;
    cudaError_t err = cudaMemcpyAsync(
        host_buffer, d_results + start_idx, 
        count * sizeof(int), cudaMemcpyDeviceToHost, g_stream
    );
    return (err == cudaSuccess) ? 0 : -1;
}

int get_aggregated_stats(int* conflicts, int* units, int* satisfied, int* processed) {
    if (!d_agg_results) return -1;
    
    // Copy aggregated results from all blocks
    satellite::gpu::BlockAggState* h_agg = new satellite::gpu::BlockAggState[NUM_BLOCKS];
    cudaError_t err = cudaMemcpy(
        h_agg, d_agg_results, 
        sizeof(satellite::gpu::BlockAggState) * NUM_BLOCKS, 
        cudaMemcpyDeviceToHost
    );
    
    if (err != cudaSuccess) {
        delete[] h_agg;
        return -1;
    }
    
    // Sum across all blocks
    *conflicts = 0; *units = 0; *satisfied = 0; *processed = 0;
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        *conflicts += h_agg[i].total_conflicts;
        *units += h_agg[i].total_units;
        *satisfied += h_agg[i].total_satisfied;
        *processed += h_agg[i].total_processed;
    }
    
    delete[] h_agg;
    return 0;
}

void stop_persistent_kernel() {
    int stop = 1;
    cudaMemcpy(d_stop_signal, &stop, sizeof(int), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(g_stream);
}

void shutdown_gpu_device() {
    if (d_queue) satellite::gpu::GpuQueue::destroy(d_queue);
    if (d_stop_signal) cudaFree(d_stop_signal);
    if (d_agg_results) cudaFree(d_agg_results);
    if (g_stream) cudaStreamDestroy(g_stream);
    d_queue = nullptr;
    d_results = nullptr;
    d_stop_signal = nullptr;
    d_agg_results = nullptr;
    g_stream = nullptr;
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
    if (err != cudaSuccess) return -1;
    *used = total_mem - free_mem;
    *total = total_mem;
    return 0;
}

} // extern "C"
