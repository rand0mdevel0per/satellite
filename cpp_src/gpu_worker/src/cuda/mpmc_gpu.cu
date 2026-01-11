/**
 * @file mpmc_gpu.cu
 * @brief CUDA MPMC queue host functions implementation.
 */

#include "../common/types.h"
#include <cuda_runtime.h>

namespace satellite {
namespace gpu {

/**
 * @brief Kernel to initialize queue on device.
 */
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
    
    // Allocate device memory for queue components
    cudaMalloc(&host_queue.nodes, sizeof(QueueNode) * MAX_NODES);
    cudaMalloc(&host_queue.heads, sizeof(int) * NUM_PRIORITY_LEVELS);
    cudaMalloc(&host_queue.free_head, sizeof(int));
    cudaMalloc(&host_queue.node_count, sizeof(int));
    
    // Allocate queue struct on device
    GpuQueue* device_queue;
    cudaMalloc(&device_queue, sizeof(GpuQueue));
    cudaMemcpy(device_queue, &host_queue, sizeof(GpuQueue), cudaMemcpyHostToDevice);
    
    // Initialize queue
    init_queue_kernel<<<1, 1>>>(device_queue);
    cudaDeviceSynchronize();
    
    return device_queue;
}

__host__ void GpuQueue::destroy(GpuQueue* device_queue) {
    GpuQueue host_queue;
    cudaMemcpy(&host_queue, device_queue, sizeof(GpuQueue), cudaMemcpyDeviceToHost);
    
    cudaFree(host_queue.nodes);
    cudaFree(host_queue.heads);
    cudaFree(host_queue.free_head);
    cudaFree(host_queue.node_count);
    cudaFree(device_queue);
}

} // namespace gpu
} // namespace satellite
