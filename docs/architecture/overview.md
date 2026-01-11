# Architecture Overview

Satellite is a next-generation constraint satisfaction solver that extends traditional CDCL-based SAT solving with a rich type system, heterogeneous computing (CPU/GPU), and programmable constraints.

## Component Structure

```
satellite/
├── crates/
│   ├── satellite-base      # Core types and error handling
│   ├── satellite-format    # DIMACS/JSON parsers
│   ├── satellite-cdcl      # CDCL solver core
│   ├── satellite-branch    # Branch management
│   ├── satellite-lockfree  # Lock-free data structures
│   ├── satellite-worker    # Thread pool
│   ├── satellite-jit       # LLVM JIT compilation
│   ├── satellite-kit       # High-level API
│   ├── satellite-cli       # Command-line interface
│   ├── satellite-daemon    # Server mode
│   └── satellite-ide       # Tauri+React IDE
├── cpp_src/
│   └── gpu_worker/         # CUDA/HIP kernels
└── py_src/
    └── satellite_lab/      # Python bindings
```

## Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                     User Input Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │   Python    │  │    CLI      │  │    IDE      │               │
│  │ satellite_lab  │  │  satellite  │  │ satellite-ide │               │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘               │
└─────────┼────────────────┼────────────────┼──────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      satellite-kit                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Solver API   │  Context Manager  │  Frontend Manager       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬───────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ satellite-cdcl│  │satellite-jit │  │satellite-worker│
│  CDCL Core   │  │  LLVM JIT    │  │  Thread Pool │
└──────┬───────┘  └──────────────┘  └──────┬───────┘
       │                                    │
       ▼                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                         GPU Worker                               │
│  ┌────────────────────┐  ┌────────────────────┐                 │
│  │    CUDA Kernel     │  │     HIP Kernel     │                 │
│  │  Persistent BCP    │  │   (AMD version)    │                 │
│  └────────────────────┘  └────────────────────┘                 │
└──────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Handle-Based Session Management

Solver contexts are managed via integer handles:

```python
ctx = create_context()      # Returns handle (u32)
add_clause(ctx, [1, 2, -3])
result = solve(ctx)
destroy_context(ctx)
```

### Async Job Submission

Jobs can be submitted asynchronously for parallel processing:

```python
init_worker_pool(4)         # 4 worker threads
job_id = submit_solve(ctx)  # Non-blocking
status = poll_job(job_id)   # Check status
results = fetch_finished_jobs(10)  # Collect completed
```

### GPU Acceleration

The GPU worker provides:
- **Persistent BCP Kernel**: Runs continuously, processing jobs from GPU queue
- **Warp-Cooperative Checking**: 32 threads check 32 clauses in parallel
- **Work Stealing**: Idle threads steal work from busy queues
- **Lock-Free MPMC Queue**: Efficient job distribution across warps

### Type System

| Type | Description | Example |
|------|-------------|---------|
| `bool` | Single boolean | `x = solver.bool_var()` |
| `batch[dim]` | Fixed-width bitvector | `reg = solver.batch_var(32)` |
| `int` | Arbitrary-width integer | `n = solver.int_var(16)` |
| `vec` | Vector of batches | `mem = solver.vec_var(32, 256)` |
| `BitVec` | Circuit-level bitvector | `a = BitVec(solver, 8)` |

## CPU/GPU Coordination

```
┌─────────────────────────────────────────────────────────────────┐
│                         CPU Side                                 │
│                                                                  │
│  ┌───────────┐     ┌───────────┐     ┌───────────────────────┐  │
│  │  Decision │ ──► │ Propagate │ ──► │ Conflict Analysis     │  │
│  │  (VSIDS)  │     │ (CPU BCP) │     │ (1-UIP, UNSAT Core)   │  │
│  └───────────┘     └─────┬─────┘     └───────────────────────┘  │
│                          │                                       │
│                    Large Problem?                                │
│                          │                                       │
│                     ┌────▼────┐                                  │
│                     │ Offload │                                  │
│                     │ to GPU  │                                  │
│                     └────┬────┘                                  │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                         GPU Side                                 │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Job Queue   │ ◄──│ Persistent   │ ──►│ Result Buffer    │   │
│  │  (MPMC)      │    │ BCP Kernel   │    │ (Clause Status)  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                  │
│  Work Stealing between warps for load balancing                 │
└──────────────────────────────────────────────────────────────────┘
```

## UNSAT Core Tracking

UNSAT core extraction is handled on the CPU side:

1. Enable tracking: `solver.enable_unsat_core()`
2. During conflict resolution, record used clause IDs
3. After UNSAT result: `core = solver.get_unsat_core()`

The GPU contributes to BCP but clause tracking aggregates on CPU for simplicity.
