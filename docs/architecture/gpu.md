# GPU Acceleration

Satellite supports GPU acceleration for Boolean Constraint Propagation (BCP) using CUDA and HIP.

## Overview

The GPU worker provides:
- **Persistent BCP Kernel**: Runs continuously processing jobs from a GPU queue
- **Warp-Cooperative Processing**: 32 threads check 32 clauses in parallel
- **Intra-Warp Work Stealing**: Idle threads steal work from busy queues
- **Lock-Free MPMC Queue**: Priority-based job distribution

## Supported Features on GPU

| Feature | GPU Support | Notes |
|---------|-------------|-------|
| BCP (Clause Checking) | ✅ Full | Warp-parallel |
| Unit Propagation | ✅ Full | Detected and reported |
| Conflict Detection | ✅ Full | Early exit |
| UNSAT Core Tracking | ⚠️ Partial | CPU aggregates |
| Assumptions | ⚠️ CPU-side | Applied before GPU offload |
| Fork Context | ⚠️ CPU-side | GPU memory not cloned |
| Clause Learning | ❌ CPU only | Complex resolution |
| Decision Heuristics | ❌ CPU only | Sequential VSIDS |

## Architecture

### Memory Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                       GPU Memory                                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Clause Data                            │   │
│  │  [lit₀, lit₁, lit₂, 0, lit₀, lit₁, 0, ...]              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Clause Offsets                          │   │
│  │  [0, 3, 5, 8, ...]  (start index for each clause)        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Assignments                             │   │
│  │  [-1, 0, 1, 0, -1, ...]  (-1=false, 0=unassigned, 1=true)│   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Results Buffer                         │   │
│  │  [SAT, UNRES, CONFLICT, UNIT, ...]  (per-clause status)  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 MPMC Priority Queue                       │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                         │   │
│  │  │ P=0 │ │ P=1 │ │ P=2 │ │ P=3 │  (4 priority levels)   │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Kernel Execution Model

```
┌─────────────────────────────────────────────────────────────────┐
│                   Persistent BCP Kernel                          │
│                                                                  │
│  Block 0          Block 1          Block 2         Block N-1    │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐  ┌───────────┐ │
│  │ Warp 0    │    │ Warp 0    │    │ Warp 0    │  │ Warp 0    │ │
│  │ Warp 1    │    │ Warp 1    │    │ Warp 1    │  │ Warp 1    │ │
│  │ Warp 2    │    │ Warp 2    │    │ Warp 2    │  │ Warp 2    │ │
│  │ ...       │    │ ...       │    │ ...       │  │ ...       │ │
│  │ Warp 7    │    │ Warp 7    │    │ Warp 7    │  │ Warp 7    │ │
│  └───────────┘    └───────────┘    └───────────┘  └───────────┘ │
│       │                │                │              │        │
│       └────────────────┴────────────────┴──────────────┘        │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  MPMC Queue     │                          │
│                    │  (Pop Jobs)     │                          │
│                    └─────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
```

## Warp-Cooperative Clause Checking

Each warp (32 threads) processes 32 clauses in parallel:

```cuda
__device__ int check_clause(
    size_t clause_id, 
    const int64_t* data, 
    const size_t* offsets, 
    const int8_t* assigns
) {
    size_t start = offsets[clause_id];
    size_t end = offsets[clause_id + 1];
    int unassigned = 0;
    
    for (size_t i = start; i < end; ++i) {
        int64_t lit = data[i];
        uint64_t var = abs(lit) - 1;
        int8_t val = assigns[var];
        
        // Check satisfaction
        if ((lit > 0 && val == 1) || (lit < 0 && val == -1))
            return CLAUSE_SATISFIED;
        
        if (val == 0) unassigned++;
    }
    
    if (unassigned == 0) return CLAUSE_CONFLICT;
    if (unassigned == 1) return CLAUSE_UNIT;
    return CLAUSE_UNRESOLVED;
}
```

## Work Stealing

Idle threads can steal work from other queues:

```cuda
// Idle thread attempts to steal
if (is_idle && warp.thread_rank() == stealer_rank) {
    QueueNode temp_job;
    if (queue->pop(random_priority, &temp_job)) {
        // Take one unit of work
        stolen_job.clause_start = temp_job.clause_start;
        stolen_job.clause_end = temp_job.clause_start + 1;
        
        // Push remainder back
        if (temp_job.clause_end > temp_job.clause_start + 1) {
            queue->push(priority, job_id, branch_id,
                       temp_job.clause_start + 1, temp_job.clause_end);
        }
    }
}
```

## C Interface

```c
// Initialize GPU
int init_gpu_device();

// Launch persistent kernel (non-blocking)
void launch_persistent_kernel(
    const int64_t* clause_data,
    const size_t* clause_offsets,
    size_t num_clauses,
    const int8_t* assignments,
    int* results
);

// Submit BCP job
void submit_job(uint32_t priority, uint64_t job_id, 
                uint64_t branch_id, uint32_t start, uint32_t end);

// Read results back
int read_results(int* host_buffer, size_t start_idx, size_t count);

// Get aggregated statistics
int get_aggregated_stats(int* conflicts, int* units, 
                         int* satisfied, int* processed);

// Stop and cleanup
void stop_persistent_kernel();
void shutdown_gpu_device();
```

## Requirements

### CUDA
- CUDA Toolkit 11.0+
- SM 7.0+ (Volta or newer)
- CUB library (bundled with CUDA 11+)

### HIP (AMD)
- ROCm 5.0+
- CDNA or RDNA2+ GPU

## Performance Considerations

1. **Clause Size**: Shorter clauses benefit more from GPU parallelism
2. **Problem Size**: GPU overhead amortizes for >100K clauses
3. **Memory Bandwidth**: Clause data should fit in GPU L2 cache
4. **Load Balancing**: Work stealing helps with non-uniform clause sizes

## Limitations

1. **No Clause Learning on GPU**: Complex resolution is CPU-bound
2. **Memory-Bound**: Large problems may exceed GPU memory
3. **Synchronization Overhead**: CPU-GPU communication has latency
4. **No UNSAT Core on GPU**: Tracked on CPU during aggregation
