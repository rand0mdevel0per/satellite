# FFI API Reference

This document describes the C Foreign Function Interface (FFI) for Satellite, enabling integration with C, C++, and other languages.

## Overview

The FFI API provides:
- **Handle-based context management**: Integer handles for solver instances
- **Zero-copy clause addition**: Direct buffer access for performance
- **Async job submission**: Non-blocking solve operations
- **Thread-safe operations**: Safe concurrent access from multiple threads

## Header Files

### C Header

```c
// satellite_ffi.h
#ifndef SATELLITE_FFI_H
#define SATELLITE_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Context management
uint32_t satellite_create_context(void);
bool satellite_destroy_context(uint32_t ctx_id);
uint32_t satellite_clone_context(uint32_t ctx_id);

// Variable creation
int64_t satellite_new_bool_var(uint32_t ctx_id);
int64_t satellite_new_batch_var(uint32_t ctx_id, uint32_t dim);
int64_t satellite_new_int_var(uint32_t ctx_id, uint32_t bits);

// Clause management
void satellite_add_clause(uint32_t ctx_id, const int64_t* literals, uint32_t len);
uint32_t satellite_add_clauses_buffer(uint32_t ctx_id, const uint8_t* buffer, uint32_t buffer_len);

// Synchronous solving
typedef enum {
    SAT_RESULT_SAT = 0,
    SAT_RESULT_UNSAT = 1,
    SAT_RESULT_UNKNOWN = 2,
} SatResultType;

typedef struct {
    SatResultType type;
    bool* model;
    uint32_t model_len;
    char* error_msg;
} SatResult;

SatResult satellite_solve(uint32_t ctx_id);
SatResult satellite_solve_with_assumptions(uint32_t ctx_id, const int64_t* assumptions, uint32_t len);
void satellite_free_result(SatResult result);

// Asynchronous solving
typedef enum {
    JOB_STATUS_PENDING = 0,
    JOB_STATUS_RUNNING = 1,
    JOB_STATUS_COMPLETED = 2,
    JOB_STATUS_NOT_FOUND = 3,
} JobStatus;

uint64_t satellite_submit_solve(uint32_t ctx_id);
JobStatus satellite_poll_job(uint64_t job_id);
uint32_t satellite_fetch_finished_jobs(uint64_t* job_ids, SatResult* results, uint32_t max_count);

// Worker pool management
void satellite_init_worker_pool(uint32_t num_workers);
void satellite_shutdown_worker_pool(void);
uint32_t satellite_get_worker_count(void);

// Statistics
uint32_t satellite_context_count(void);
uint32_t satellite_job_count(void);

// Parallel exploration
uint32_t satellite_fork_context(uint32_t src_ctx_id, uint32_t* dst_ctx_ids, uint32_t num_clones);

#ifdef __cplusplus
}
#endif

#endif // SATELLITE_FFI_H
```

## Usage Examples

### Basic Solving (C)

```c
#include "satellite_ffi.h"
#include <stdio.h>

int main() {
    // Create solver context
    uint32_t ctx = satellite_create_context();

    // Create variables
    int64_t x = satellite_new_bool_var(ctx);
    int64_t y = satellite_new_bool_var(ctx);

    // Add clauses: (x OR y) AND (NOT x OR NOT y)
    int64_t clause1[] = {x, y};
    satellite_add_clause(ctx, clause1, 2);

    int64_t clause2[] = {-x, -y};
    satellite_add_clause(ctx, clause2, 2);

    // Solve
    SatResult result = satellite_solve(ctx);

    if (result.type == SAT_RESULT_SAT) {
        printf("SAT: x=%d, y=%d\n", result.model[0], result.model[1]);
        satellite_free_result(result);
    } else if (result.type == SAT_RESULT_UNSAT) {
        printf("UNSAT\n");
    } else {
        printf("Unknown: %s\n", result.error_msg);
        satellite_free_result(result);
    }

    // Cleanup
    satellite_destroy_context(ctx);
    return 0;
}
```

### Async Solving (C)

```c
#include "satellite_ffi.h"
#include <stdio.h>
#include <unistd.h>

int main() {
    // Initialize worker pool
    satellite_init_worker_pool(4);

    // Create context and add clauses
    uint32_t ctx = satellite_create_context();
    int64_t x = satellite_new_bool_var(ctx);
    int64_t y = satellite_new_bool_var(ctx);

    int64_t clause[] = {x, y};
    satellite_add_clause(ctx, clause, 2);

    // Submit async job
    uint64_t job_id = satellite_submit_solve(ctx);
    printf("Submitted job %llu\n", job_id);

    // Poll until complete
    JobStatus status;
    do {
        status = satellite_poll_job(job_id);
        if (status == JOB_STATUS_RUNNING) {
            printf("Job still running...\n");
            usleep(100000);  // 100ms
        }
    } while (status == JOB_STATUS_PENDING || status == JOB_STATUS_RUNNING);

    // Fetch result
    uint64_t job_ids[1];
    SatResult results[1];
    uint32_t count = satellite_fetch_finished_jobs(job_ids, results, 1);

    if (count > 0 && results[0].type == SAT_RESULT_SAT) {
        printf("SAT: x=%d, y=%d\n", results[0].model[0], results[0].model[1]);
        satellite_free_result(results[0]);
    }

    // Cleanup
    satellite_destroy_context(ctx);
    satellite_shutdown_worker_pool();
    return 0;
}
```

### C++ Wrapper Example

```cpp
#include "satellite_ffi.h"
#include <memory>
#include <vector>
#include <stdexcept>

class SatelliteSolver {
    uint32_t ctx_id;

public:
    SatelliteSolver() : ctx_id(satellite_create_context()) {
        if (ctx_id == 0) throw std::runtime_error("Failed to create context");
    }

    ~SatelliteSolver() {
        satellite_destroy_context(ctx_id);
    }

    int64_t new_bool_var() {
        return satellite_new_bool_var(ctx_id);
    }

    void add_clause(const std::vector<int64_t>& literals) {
        satellite_add_clause(ctx_id, literals.data(), literals.size());
    }

    SatResult solve() {
        return satellite_solve(ctx_id);
    }
};
```

## Best Practices

### Memory Management

1. **Always free results**: Call `satellite_free_result()` for SAT results with models or error messages
2. **Destroy contexts**: Call `satellite_destroy_context()` when done to prevent leaks
3. **Shutdown worker pool**: Call `satellite_shutdown_worker_pool()` before program exit

### Thread Safety

- Context creation/destruction is thread-safe
- Multiple threads can solve different contexts concurrently
- Do not modify the same context from multiple threads simultaneously
- Worker pool is shared across all threads

### Performance Tips

1. **Use buffer API**: `satellite_add_clauses_buffer()` for bulk clause addition (zero-copy)
2. **Reuse contexts**: Clone contexts instead of recreating for similar problems
3. **Async solving**: Use worker pool for parallel solving of multiple problems
4. **Batch operations**: Group variable creation and clause addition

### Error Handling

Always check return values:
- Context creation returns 0 on failure
- Variable creation returns -1 on failure
- Job submission returns 0 on failure
- Check `SatResult.type` before accessing model
