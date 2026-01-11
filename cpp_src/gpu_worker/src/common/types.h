/**
 * @file types.h
 * @brief Common GPU types for BCP kernel (portable CUDA/HIP).
 */

#pragma once

#include "../common/gpu_compat.h"
#include <cstdint>

namespace satellite {
namespace gpu {

/**
 * @brief Node in the GPU work queue.
 */
struct QueueNode {
    uint64_t job_id;       ///< Unique job identifier
    uint64_t branch_id;    ///< Branch/exploration path ID
    uint32_t priority;     ///< Priority level (0-3, 0 = highest)
    uint32_t clause_start; ///< Start index in clause array
    uint32_t clause_end;   ///< End index in clause array (exclusive)
    int next;              ///< Next node index in linked list (-1 = end)
};

/**
 * @brief Lock-free multi-priority GPU work queue.
 * 
 * Implements a lock-free MPMC queue with multiple priority levels.
 * Uses atomic CAS for push/pop operations.
 */
class GpuQueue {
public:
    static constexpr int MAX_NODES = 1024 * 1024;  ///< Maximum queue capacity
    static constexpr int NUM_PRIORITY_LEVELS = 4;  ///< Priority levels (0-3)

    QueueNode* nodes;      ///< Node storage array
    int* heads;            ///< Head pointers for each priority level
    int* free_head;        ///< Free list head pointer
    int* node_count;       ///< Current node count

    /// Create queue on device
    __host__ static GpuQueue* create();
    
    /// Destroy queue and free device memory
    __host__ static void destroy(GpuQueue* queue);

    /**
     * @brief Push a job to the queue (device-side).
     * 
     * @param priority Priority level (0-3)
     * @param job_id Job identifier
     * @param branch_id Branch identifier
     * @param clause_start Start clause index
     * @param clause_end End clause index
     */
    __device__ void push(uint32_t priority, uint64_t job_id, uint64_t branch_id,
                         uint32_t clause_start, uint32_t clause_end) {
        // Allocate node
        int node_idx = gpuAtomicAdd(node_count, 1);
        if (node_idx >= MAX_NODES) { 
            gpuAtomicSub(node_count, 1); 
            return; // Queue full
        }

        // Initialize node
        nodes[node_idx].job_id = job_id;
        nodes[node_idx].branch_id = branch_id;
        nodes[node_idx].priority = priority;
        nodes[node_idx].clause_start = clause_start;
        nodes[node_idx].clause_end = clause_end;

        // Lock-free push to priority list
        int old_head = heads[priority];
        int assumed;
        do {
            assumed = old_head;
            nodes[node_idx].next = assumed;
            old_head = gpuAtomicCAS(&heads[priority], assumed, node_idx);
        } while (assumed != old_head);
    }

    /**
     * @brief Pop a job from the queue (device-side).
     * 
     * @param priority Priority level to pop from
     * @param out_node Output node (if successful)
     * @return true if job was popped, false if queue empty
     */
    __device__ bool pop(uint32_t priority, QueueNode* out_node) {
        int old_head = heads[priority];
        int assumed;
        
        do {
            assumed = old_head;
            if (assumed == -1) return false; // Empty
            int next_node = nodes[assumed].next;
            old_head = gpuAtomicCAS(&heads[priority], assumed, next_node);
        } while (assumed != old_head);
        
        *out_node = nodes[assumed];
        return true;
    }
};

/**
 * @brief Block-level aggregation state for BCP results.
 */
struct BlockAggState {
    int total_conflicts;   ///< Total conflict count
    int total_units;       ///< Total unit clause count
    int total_satisfied;   ///< Total satisfied clause count
    int total_processed;   ///< Total clauses processed
};

// Result codes for clause checking
constexpr int CLAUSE_SATISFIED = 1;
constexpr int CLAUSE_CONFLICT = 0;
constexpr int CLAUSE_UNRESOLVED = 2;
constexpr int CLAUSE_UNIT = 3;

} // namespace gpu
} // namespace satellite
