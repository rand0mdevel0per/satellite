#pragma once

/**
 * @file gpu_worker.h
 * @brief C API for Satellite GPU worker pool.
 *
 * This library provides CUDA/HIP GPU acceleration for Satellite's
 * parallel job execution system. It exposes a C API for FFI with Rust.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Result code for GPU operations.
 */
typedef enum {
    GPU_OK = 0,           /**< Operation succeeded */
    GPU_ERROR = -1,       /**< General error */
    GPU_NOT_AVAILABLE = -2, /**< GPU not available */
    GPU_OUT_OF_MEMORY = -3, /**< Out of GPU memory */
} GpuStatus;

/**
 * @brief Result from a GPU BCP operation.
 */
typedef struct {
    int32_t status;        /**< 0 = ok, 1 = conflict, -1 = error */
    size_t conflict_clause; /**< Conflict clause index (if status == 1) */
} GpuBcpResult;

/**
 * @brief Initializes the GPU worker pool.
 *
 * Must be called before any other GPU functions.
 *
 * @return GPU_OK on success, error code otherwise.
 */
int32_t gpu_worker_init(void);

/**
 * @brief Shuts down the GPU worker pool.
 *
 * Releases all GPU resources. No GPU functions should be called after this.
 */
void gpu_worker_shutdown(void);

/**
 * @brief Checks if GPU is available.
 *
 * @return 1 if GPU is available, 0 otherwise.
 */
int32_t gpu_worker_is_available(void);

/**
 * @brief Gets the number of available GPU devices.
 *
 * @return Number of GPU devices, or 0 if none available.
 */
int32_t gpu_worker_device_count(void);

/**
 * @brief Submits a BCP (Boolean Constraint Propagation) job to the GPU.
 *
 * The job checks multiple clauses in parallel using GPU warps.
 * Each warp handles 32 clauses (NVIDIA) or 64 clauses (AMD).
 *
 * @param clause_data Flattened clause literals (ends with 0 for each clause).
 * @param num_clauses Number of clauses to check.
 * @param assignments Current variable assignments (-1 = false, 0 = unassigned, 1 = true).
 * @param num_vars Number of variables.
 * @return GPU_OK if job was submitted, error code otherwise.
 */
int32_t gpu_worker_submit_bcp(
    const int64_t* clause_data,
    size_t num_clauses,
    const int8_t* assignments,
    size_t num_vars
);

/**
 * @brief Polls for a completed GPU result.
 *
 * Non-blocking. Check return code to determine if result is available.
 *
 * @param result Output parameter for the result.
 * @return 0 if result is available, 1 if no result ready, negative on error.
 */
int32_t gpu_worker_poll_result(GpuBcpResult* result);

/**
 * @brief Synchronously waits for all GPU jobs to complete.
 *
 * Blocks until the GPU queue is empty.
 */
void gpu_worker_sync(void);

/**
 * @brief Gets GPU memory usage in bytes.
 *
 * @param used Output parameter for used memory.
 * @param total Output parameter for total memory.
 * @return GPU_OK on success.
 */
int32_t gpu_worker_memory_info(size_t* used, size_t* total);

#ifdef __cplusplus
}
#endif
