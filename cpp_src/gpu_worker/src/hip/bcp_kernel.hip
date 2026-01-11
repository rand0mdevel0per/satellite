/**
 * @file bcp_kernel.hip
 * @brief HIP BCP kernel implementation (AMD GPU version).
 * 
 * This is the HIP-portable version of the BCP kernel for AMD GPUs.
 * It mirrors the CUDA implementation but uses HIP APIs.
 */

#include "../common/types.h"
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

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

/**
 * @brief Check if a clause is satisfied, conflicting, unit, or unresolved.
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
    
    for (size_t i = start; i < end; ++i) {
        int64_t lit = data[i];
        if (lit == 0) break;
        
        uint64_t var = (lit > 0 ? lit : -lit) - 1;
        int8_t val = assigns[var];
        
        if ((lit > 0 && val == 1) || (lit < 0 && val == -1)) {
            return CLAUSE_SATISFIED;
        }
        
        if (val == 0) unassigned_count++;
    }
    
    if (unassigned_count == 0) return CLAUSE_CONFLICT;
    if (unassigned_count == 1) return CLAUSE_UNIT;
    return CLAUSE_UNRESOLVED;
}

/**
 * @brief Aggregation result structure for wavefront-level reduction.
 */
struct WavefrontAggResult {
    int conflict_count;
    int unit_count;
    int satisfied_count;
};

/**
 * @brief Aggregate results within a wavefront (64 threads on AMD).
 */
__device__ WavefrontAggResult aggregate_wavefront_results(
    cg::thread_block_tile<64>& wave,
    int result,
    bool valid
) {
    uint64_t conflict_mask = wave.ballot(valid && result == CLAUSE_CONFLICT);
    uint64_t unit_mask = wave.ballot(valid && result == CLAUSE_UNIT);
    uint64_t satisfied_mask = wave.ballot(valid && result == CLAUSE_SATISFIED);
    
    WavefrontAggResult agg;
    agg.conflict_count = __popcll(conflict_mask);
    agg.unit_count = __popcll(unit_mask);
    agg.satisfied_count = __popcll(satisfied_mask);
    
    return agg;
}

/**
 * @brief Persistent BCP kernel for HIP/AMD GPUs.
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
    cg::thread_block_tile<64> wave = cg::tiled_partition<64>(block);
    
    __shared__ BlockAggState block_agg;
    if (threadIdx.x == 0) {
        block_agg.total_conflicts = 0;
        block_agg.total_units = 0;
        block_agg.total_satisfied = 0;
        block_agg.total_processed = 0;
    }
    __syncthreads();
    
    uint32_t rng_state = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x + clock64());
    
    while (*stop_signal == 0) {
        rng_state = xorshift32(rng_state);
        uint32_t priority = 0;
        
        if (wave.thread_rank() == 0) {
            uint32_t r = rng_state % 100;
            if (r < 50) priority = 0;
            else if (r < 80) priority = 1;
            else if (r < 95) priority = 2;
            else priority = 3;
        }
        priority = wave.shfl(priority, 0);
        
        QueueNode job;
        bool success = false;
        
        if (wave.thread_rank() == 0) {
            success = queue->pop(priority, &job);
        }
        
        uint64_t success_mask = wave.ballot(success);
        if (success_mask == 0) {
            __builtin_amdgcn_s_sleep(10);
            continue;
        }
        
        // Broadcast job
        uint32_t* job_ptr = (uint32_t*)&job;
        for (int i = 0; i < sizeof(QueueNode)/4; ++i) {
            job_ptr[i] = wave.shfl(job_ptr[i], 0);
        }
        
        uint32_t work_size = job.clause_end - job.clause_start;
        bool is_idle = wave.thread_rank() >= work_size;
        
        int result = CLAUSE_UNRESOLVED;
        bool valid_work = false;
        
        if (!is_idle && wave.thread_rank() < work_size) {
            size_t clause_idx = job.clause_start + wave.thread_rank();
            if (clause_idx < num_clauses) {
                result = check_clause(clause_idx, clause_data, clause_offsets, assignments);
                results[clause_idx] = result;
                valid_work = true;
            }
        }
        
        WavefrontAggResult wave_agg = aggregate_wavefront_results(wave, result, valid_work);
        
        if (wave.thread_rank() == 0) {
            atomicAdd(&block_agg.total_conflicts, wave_agg.conflict_count);
            atomicAdd(&block_agg.total_units, wave_agg.unit_count);
            atomicAdd(&block_agg.total_satisfied, wave_agg.satisfied_count);
            atomicAdd(&block_agg.total_processed, __popcll(wave.ballot(valid_work)));
        }
        
        if (wave_agg.conflict_count > 0) {
            __syncthreads();
            if (threadIdx.x == 0 && agg_results != nullptr) {
                atomicAdd(&agg_results[blockIdx.x].total_conflicts, block_agg.total_conflicts);
                atomicAdd(&agg_results[blockIdx.x].total_units, block_agg.total_units);
                atomicAdd(&agg_results[blockIdx.x].total_satisfied, block_agg.total_satisfied);
                atomicAdd(&agg_results[blockIdx.x].total_processed, block_agg.total_processed);
                block_agg = {0, 0, 0, 0};
            }
            __syncthreads();
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && agg_results != nullptr) {
        atomicAdd(&agg_results[blockIdx.x].total_conflicts, block_agg.total_conflicts);
        atomicAdd(&agg_results[blockIdx.x].total_units, block_agg.total_units);
        atomicAdd(&agg_results[blockIdx.x].total_satisfied, block_agg.total_satisfied);
        atomicAdd(&agg_results[blockIdx.x].total_processed, block_agg.total_processed);
    }
}

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
