/**
 * @file gpu_worker.cpp
 * @brief C API implementation for GPU worker.
 */

#include "gpu_worker.h"
#include "types.h"

#include <atomic>
#include <mutex>
#include <queue>
#include <memory>

namespace {

// Global state
std::atomic<bool> g_initialized{false};
std::mutex g_mutex;
std::queue<GpuBcpResult> g_results;

#if defined(SATELLITE_CUDA) || defined(SATELLITE_HIP)
// Forward declarations for GPU functions
extern "C" int init_gpu_device();
extern "C" void shutdown_gpu_device();
extern "C" int get_device_count();
extern "C" int submit_bcp_kernel(const int64_t*, size_t, const int8_t*, size_t);
extern "C" void sync_gpu();
extern "C" int get_memory_info(size_t*, size_t*);
#endif

} // anonymous namespace

extern "C" {

int32_t gpu_worker_init(void) {
    if (g_initialized.exchange(true)) {
        return GPU_OK; // Already initialized
    }

#if defined(SATELLITE_CUDA) || defined(SATELLITE_HIP)
    return init_gpu_device();
#else
    // No GPU support compiled in
    return GPU_NOT_AVAILABLE;
#endif
}

void gpu_worker_shutdown(void) {
    if (!g_initialized.exchange(false)) {
        return; // Not initialized
    }

#if defined(SATELLITE_CUDA) || defined(SATELLITE_HIP)
    shutdown_gpu_device();
#endif

    std::lock_guard<std::mutex> lock(g_mutex);
    while (!g_results.empty()) {
        g_results.pop();
    }
}

int32_t gpu_worker_is_available(void) {
#if defined(SATELLITE_CUDA) || defined(SATELLITE_HIP)
    return g_initialized.load() ? 1 : 0;
#else
    return 0;
#endif
}

int32_t gpu_worker_device_count(void) {
#if defined(SATELLITE_CUDA) || defined(SATELLITE_HIP)
    return get_device_count();
#else
    return 0;
#endif
}

int32_t gpu_worker_submit_bcp(
    const int64_t* clause_data,
    size_t num_clauses,
    const int8_t* assignments,
    size_t num_vars
) {
    if (!g_initialized.load()) {
        return GPU_ERROR;
    }

#if defined(SATELLITE_CUDA) || defined(SATELLITE_HIP)
    return submit_bcp_kernel(clause_data, num_clauses, assignments, num_vars);
#else
    // Fallback: CPU implementation
    // Check clauses on CPU
    for (size_t c = 0; c < num_clauses; ++c) {
        bool satisfied = false;
        bool unit = false;
        int64_t unit_lit = 0;
        size_t unassigned_count = 0;

        // Find clause start (assuming 0-terminated clauses)
        size_t start = 0;
        for (size_t i = 0; i < c; ++i) {
            while (clause_data[start] != 0) ++start;
            ++start;
        }

        // Check clause
        for (size_t i = start; clause_data[i] != 0; ++i) {
            int64_t lit = clause_data[i];
            uint64_t var = (lit > 0 ? lit : -lit) - 1;
            
            if (var >= num_vars) continue;
            
            int8_t val = assignments[var];
            bool lit_true = (lit > 0) ? (val == 1) : (val == -1);
            
            if (lit_true) {
                satisfied = true;
                break;
            }
            
            if (val == 0) {
                unassigned_count++;
                unit_lit = lit;
            }
        }

        if (!satisfied && unassigned_count == 0) {
            // Conflict!
            std::lock_guard<std::mutex> lock(g_mutex);
            g_results.push({1, c});
        }
    }

    // All clauses checked, no conflict
    std::lock_guard<std::mutex> lock(g_mutex);
    g_results.push({0, 0});
    return GPU_OK;
#endif
}

int32_t gpu_worker_poll_result(GpuBcpResult* result) {
    if (!result) {
        return GPU_ERROR;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_results.empty()) {
        return 1; // No result ready
    }

    *result = g_results.front();
    g_results.pop();
    return 0;
}

void gpu_worker_sync(void) {
#if defined(SATELLITE_CUDA) || defined(SATELLITE_HIP)
    sync_gpu();
#endif
}

int32_t gpu_worker_memory_info(size_t* used, size_t* total) {
    if (!used || !total) {
        return GPU_ERROR;
    }

#if defined(SATELLITE_CUDA) || defined(SATELLITE_HIP)
    return get_memory_info(used, total);
#else
    *used = 0;
    *total = 0;
    return GPU_NOT_AVAILABLE;
#endif
}

} // extern "C"
