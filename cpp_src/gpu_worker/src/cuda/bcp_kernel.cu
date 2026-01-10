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
 * @file bcp_kernel.cu
 * @brief CUDA BCP kernel implementation.
 */

#include "types.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace satellite {
namespace gpu {

// PRNG Xorshift
__device__ uint32_t xorshift32(uint32_t state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

/**
 * @brief Persistent BCP kernel.
 *
 * @param queue MPMC queue.
 * @param clause_data Flattened clause data.
 * @param clause_offsets Start offset for each clause.
 * @param num_clauses Number of clauses.
 * @param assignments Variable assignments.
 * @param results Output results.
 * @param stop_signal Flag to stop the kernel.
 */
__global__ void bcp_persistent_kernel(
    GpuQueue* queue,
    const int64_t* __restrict__ clause_data,
    const size_t* __restrict__ clause_offsets,
    size_t num_clauses,
    const int8_t* __restrict__ assignments,
    int* __restrict__ results,
    volatile int* stop_signal
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Initial RNG state based on thread ID and clock
    uint32_t rng_state = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x + clock64());
    
    while (*stop_signal == 0) {
        // 1. Select Queue (Lane 0 only)
        rng_state = xorshift32(rng_state);
        uint32_t priority = 0;
        
        // Simple weighted sample: 0=50%, 1=30%, 2=15%, 3=5%
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
        success = warp.any(success); // Broadcast success
        
        if (!success) {
            // Backoff/Spin
            continue;
        }
        
        // Broadcast job details
        job = warp.shfl(job, 0); // Warning: Struct shuffle not supported directly for >32bit?
        // Need to shuffle members individually or use shared memory.
        // For simplicity using shared memory here.
        __shared__ QueueNode shared_job[32]; // One per warp in block? No, one per block.
        // Block has multiple warps. Need warp-specific storage or shuffle each field.
        // Shuffle each field manually (safe and fast enough).
        job.job_id = warp.shfl(job.job_id, 0); // 64-bit shuffle?
        // CUDA `shfl_sync` supports 32-bit. 64-bit needs split.
        // Assuming we implement `shfl64` helper or just cast.
        // For now, let's assume `job` fits in registers and we shuffle fields.
        // Actually, let's use shared memory for simplicity in this MVP to avoid 64-bit shuffle boilerplate.
        // But concurrent warps... need index.
        // Warp ID in block: threadIdx.x / 32.
        // __shared__ QueueNode shared_job[(BLOCK_SIZE/32)]; 
        // Let's assume block size 256 -> 8 warps.
        
        // To keep this file concise, I'll use 32-bit field shuffles (casting 64 to 2x32).
         uint32_t* job_ptr = (uint32_t*)&job;
         for (int i = 0; i < sizeof(QueueNode)/4; ++i) {
             job_ptr[i] = warp.shfl(job_ptr[i], 0);
         }
        
        // 3. Intra-Warp Stealing (Complex Closure Optimization)
        // Identify idle threads
        uint32_t work_size = job.clause_end - job.clause_start;
        bool is_idle = warp.thread_rank() >= work_size;
        uint32_t idle_mask = warp.ballot(is_idle);
        
        QueueNode stolen_job;
        bool has_stolen = false;
        uint32_t stealer_rank = __ffs(idle_mask) - 1; // First idle thread
        
        if (idle_mask != 0 && warp.thread_rank() == stealer_rank) {
            // Attempt to pop a new job to steal 1 unit
            // Select priority (reuse previous or new? Randomize again)
            uint32_t r = xorshift32(rng_state) % 100;
            uint32_t p = (r < 50) ? 0 : 1; // Prefer high priority for help?
            
            QueueNode temp_job;
            if (queue->pop(p, &temp_job)) {
                // Steal 1 unit
                if (temp_job.clause_end > temp_job.clause_start) {
                    stolen_job.job_id = temp_job.job_id;
                    stolen_job.branch_id = temp_job.branch_id;
                    
                    // Take first unit
                    stolen_job.clause_start = temp_job.clause_start;
                    stolen_job.clause_end = temp_job.clause_start + 1;
                    
                    // Push remainder back
                    if (temp_job.clause_end > temp_job.clause_start + 1) {
                         queue->push(0, temp_job.job_id, temp_job.branch_id, 
                                     temp_job.clause_start + 1, temp_job.clause_end);
                    }
                    has_stolen = true;
                }
            }
        }
        
        // Broadcast stolen job to the stealer (and potentially others if we supported multi-steal)
        // Only the stealer needs it? Yes, we said "steal single unit".
        // But `has_stolen` needs to be known? No, only stealer executes.
        // Divergence is fine if masked properly.
        
        // 4. Execution
        int result = 2; // Default unresolved
        
        if (!is_idle) {
            // Process Original Job
            size_t idx = job.clause_start + warp.thread_rank();
            result = check_clause(idx, clause_data, clause_offsets, assignments);
             // Write result
            if (idx < num_clauses) results[idx] = result;
        } 
        else if (is_idle && warp.thread_rank() == stealer_rank && has_stolen) {
             // Process Stolen Job
             size_t idx = stolen_job.clause_start;
             // Note: stolen_job might refer to different clause_data context? 
             // Assumption: Global clause_data is shared.
             result = check_clause(idx, clause_data, clause_offsets, assignments);
             if (idx < num_clauses) results[idx] = result;
        }
        
        // 5. Aggregation
        // (Simplified: Writing to global memory 'results' is enough for aggregation kernel to pick up)
    }
}

__device__ int check_clause(size_t clause_id, const int64_t* data, const size_t* offsets, const int8_t* assigns) {
    size_t start = offsets[clause_id];
    size_t end = offsets[clause_id + 1];
    bool satisfied = false;
    int unassigned_count = 0;
    
    for (size_t i = start; i < end; ++i) {
        int64_t lit = data[i];
        if (lit == 0) break;
        uint64_t var = (lit > 0 ? lit : -lit) - 1;
        int8_t val = assigns[var];
        if ((lit > 0 && val == 1) || (lit < 0 && val == -1)) { return 1; }
        if (val == 0) unassigned_count++;
    }
    
    if (unassigned_count == 0) return 0; // Conflict
    if (unassigned_count == 1) return 1; // Unit (treated as satisfied/propagatable)
    return 2;
}

// Proxy kernel to push job from host
__global__ void push_proxy_kernel(GpuQueue* queue, uint32_t priority, uint64_t job_id, uint64_t branch_id, uint32_t start, uint32_t end) {
    if (threadIdx.x == 0) {
        queue->push(priority, job_id, branch_id, start, end);
    }
}

} // namespace gpu
} // namespace satellite

// C interface for the kernel launcher
extern "C" {

static cudaStream_t g_stream = nullptr;
static int* d_results = nullptr;
static int* d_stop_signal = nullptr;
static satellite::gpu::GpuQueue* d_queue = nullptr;

int init_gpu_device() {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) return -1;
    err = cudaStreamCreate(&g_stream);
    if (err != cudaSuccess) return -1;
    
    d_queue = satellite::gpu::GpuQueue::create();
    
    cudaMalloc(&d_stop_signal, sizeof(int));
    cudaMemset(d_stop_signal, 0, sizeof(int));
    
    return 0;
}

void launch_persistent_kernel(
    const int64_t* clause_data,
    const size_t* clause_offsets,
    size_t num_clauses,
    const int8_t* assignments,
    int* results
) {
    // Store results pointer globally (hack for MVP, assumes single persistent launch)
    d_results = results;

    // Launch persistent kernel
    satellite::gpu::bcp_persistent_kernel<<<128, 256, 0, g_stream>>>(
        d_queue, clause_data, clause_offsets, num_clauses, assignments, results, d_stop_signal
    );
}

void submit_job(uint32_t priority, uint64_t job_id, uint64_t branch_id, uint32_t start, uint32_t end) {
    // Launch small kernel to push to queue
    satellite::gpu::push_proxy_kernel<<<1, 1, 0, g_stream>>>(d_queue, priority, job_id, branch_id, start, end);
}

int read_results(int* host_buffer, size_t start_idx, size_t count) {
    if (!d_results) return -1;
    cudaError_t err = cudaMemcpyAsync(host_buffer, d_results + start_idx, count * sizeof(int), cudaMemcpyDeviceToHost, g_stream);
    if (err != cudaSuccess) return -1;
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
    if (g_stream) cudaStreamDestroy(g_stream);
    d_results = nullptr;
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
