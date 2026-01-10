#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace satellite {
namespace gpu {

struct QueueNode {
    uint64_t job_id;
    uint64_t branch_id;
    uint32_t priority;
    uint32_t clause_start;
    uint32_t clause_end;
    // For work stealing: currently processing thread index or mask? 
    // Simplified: Just standard bounds.
    int next; 
};

class GpuQueue {
public:
    static constexpr int MAX_NODES = 1024 * 1024;
    static constexpr int NUM_PRIORITY_LEVELS = 4;

    QueueNode* nodes;
    int* heads;
    int* free_head;
    int* node_count;

    __host__ static GpuQueue* create();
    __host__ static void destroy(GpuQueue* queue);

    __device__ void push(uint32_t priority, uint64_t job_id, uint64_t branch_id,
                         uint32_t clause_start, uint32_t clause_end) {
        int node_idx = atomicAdd(node_count, 1);
        if (node_idx >= MAX_NODES) { atomicSub(node_count, 1); return; }

        nodes[node_idx].job_id = job_id;
        nodes[node_idx].branch_id = branch_id;
        nodes[node_idx].priority = priority;
        nodes[node_idx].clause_start = clause_start;
        nodes[node_idx].clause_end = clause_end;

        int old_head = heads[priority];
        int assumed;
        do {
            assumed = old_head;
            nodes[node_idx].next = assumed;
            old_head = atomicCAS(&heads[priority], assumed, node_idx);
        } while (assumed != old_head);
    }

    __device__ bool pop(uint32_t priority, QueueNode* out_node) {
        int old_head = heads[priority];
        int assumed;
        
        do {
            assumed = old_head;
            if (assumed == -1) return false;
            int next_node = nodes[assumed].next;
            old_head = atomicCAS(&heads[priority], assumed, next_node);
        } while (assumed != old_head);
        
        *out_node = nodes[assumed];
        return true;
    }
};

}
}
