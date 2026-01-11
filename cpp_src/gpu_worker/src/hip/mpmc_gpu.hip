/**
 * @file mpmc_gpu.hip
 * @brief HIP MPMC queue implementation for AMD GPUs.
 */

#include "../common/types.h"
#include <hip/hip_runtime.h>

namespace satellite {
namespace gpu {

__global__ void init_queue_kernel(GpuQueue* queue) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < GpuQueue::NUM_PRIORITY_LEVELS; ++i) {
            queue->heads[i] = -1;
        }
        *queue->free_head = 0;
        *queue->node_count = 0;
    }
}

__host__ GpuQueue* GpuQueue::create() {
    GpuQueue host_queue;
    
    hipMalloc(&host_queue.nodes, sizeof(QueueNode) * MAX_NODES);
    hipMalloc(&host_queue.heads, sizeof(int) * NUM_PRIORITY_LEVELS);
    hipMalloc(&host_queue.free_head, sizeof(int));
    hipMalloc(&host_queue.node_count, sizeof(int));
    
    GpuQueue* device_queue;
    hipMalloc(&device_queue, sizeof(GpuQueue));
    hipMemcpy(device_queue, &host_queue, sizeof(GpuQueue), hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(init_queue_kernel, dim3(1), dim3(1), 0, 0, device_queue);
    hipDeviceSynchronize();
    
    return device_queue;
}

__host__ void GpuQueue::destroy(GpuQueue* device_queue) {
    GpuQueue host_queue;
    hipMemcpy(&host_queue, device_queue, sizeof(GpuQueue), hipMemcpyDeviceToHost);
    
    hipFree(host_queue.nodes);
    hipFree(host_queue.heads);
    hipFree(host_queue.free_head);
    hipFree(host_queue.node_count);
    hipFree(device_queue);
}

} // namespace gpu
} // namespace satellite
