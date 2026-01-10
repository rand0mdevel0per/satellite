/**
 * @file mpmc_gpu.cu
 * @brief GPU-side MPMC queue implementation.
 */

#include "types.h"
#include <cuda_runtime.h>
#include <atomic>

namespace satellite {
namespace gpu {

/**
 * @brief Lock-free node for GPU queue.
 */
struct QueueNode {
    uint64_t job_id;
    uint64_t branch_id;
    uint32_t priority;
    uint32_t clause_start;
    uint32_t clause_end;
    int next; // Index of next node (-1 = null)
};

/**
 * @brief GPU-side MPMC queue.
 *
 * Uses atomic operations for lock-free push/pop.
 * Stored in device memory.
 */
class GpuQueue {
public:
    static constexpr int MAX_NODES = 1024 * 1024; // 1M nodes max
    static constexpr int NUM_PRIORITY_LEVELS = 4;

private:
    QueueNode* nodes;
    int* heads; // One head per priority level
    int* free_head;
    int* node_count;

public:
    /**
     * @brief Allocates queue in device memory.
     */
    __host__ static GpuQueue* create() {
        GpuQueue* queue;
        cudaMalloc(&queue, sizeof(GpuQueue));
        
        QueueNode* nodes;
        cudaMalloc(&nodes, sizeof(QueueNode) * MAX_NODES);
        
        int* heads;
        cudaMalloc(&heads, sizeof(int) * NUM_PRIORITY_LEVELS);
        cudaMemset(heads, -1, sizeof(int) * NUM_PRIORITY_LEVELS);
        
        int* free_head;
        cudaMalloc(&free_head, sizeof(int));
        int zero = 0;
        cudaMemcpy(free_head, &zero, sizeof(int), cudaMemcpyHostToDevice);
        
        int* node_count;
        cudaMalloc(&node_count, sizeof(int));
        cudaMemset(node_count, 0, sizeof(int));
        
        // Initialize on device
        // ... (would need a kernel to do this properly)
        
        return queue;
    }

    /**
     * @brief Frees queue from device memory.
     */
    __host__ static void destroy(GpuQueue* queue) {
        // Free all allocations
        cudaFree(queue);
    }

    /**
     * @brief Pushes a job onto the queue (device function).
     */
    __device__ void push(uint32_t priority, uint64_t job_id, uint64_t branch_id,
                         uint32_t clause_start, uint32_t clause_end) {
        // Allocate node from free list
        int node_idx = atomicAdd(node_count, 1);
        if (node_idx >= MAX_NODES) {
            atomicSub(node_count, 1);
            return; // Queue full
        }
        
        // Initialize node
        nodes[node_idx].job_id = job_id;
        nodes[node_idx].branch_id = branch_id;
        nodes[node_idx].priority = priority;
        nodes[node_idx].clause_start = clause_start;
        nodes[node_idx].clause_end = clause_end;
        
        // Push to head of priority queue
        int old_head;
        do {
            old_head = heads[priority];
            nodes[node_idx].next = old_head;
        } while (atomicCAS(&heads[priority], old_head, node_idx) != old_head);
    }

    /**
     * @brief Pops a job from the queue (device function).
     */
    __device__ bool pop(uint32_t priority, QueueNode* out_node) {
        int old_head;
        int new_head;
        
        do {
            old_head = heads[priority];
            if (old_head == -1) {
                return false; // Queue empty
            }
            new_head = nodes[old_head].next;
        } while (atomicCAS(&heads[priority], old_head, new_head) != old_head);
        
        *out_node = nodes[old_head];
        return true;
    }
};

} // namespace gpu
} // namespace satellite
