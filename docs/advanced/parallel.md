# Parallel Solving

Satellite provides multiple strategies for parallel constraint solving, from multi-threaded worker pools to distributed solving across machines.

## Overview

Parallel solving strategies:
- **Worker Pool**: Thread pool for async job execution
- **Context Cloning**: Fork solver states for parallel exploration
- **Branch Management**: Git-style branching with reference counting
- **Distributed Solving**: WebSocket-based daemon for remote workers
- **GPU Acceleration**: CUDA/HIP kernels for BCP

## Worker Pool

### Initialization

```python
from satellite_lab import init_worker_pool, shutdown_worker_pool

# Initialize with 4 worker threads
init_worker_pool(4)

# ... submit jobs ...

# Cleanup
shutdown_worker_pool()
```

### Async Job Submission

```python
from satellite_lab import BatchingSolver, submit_solve, poll_job, fetch_finished_jobs

solver = BatchingSolver()
# Add constraints...

# Submit non-blocking solve
job_id = submit_solve(solver.ctx_id)
print(f"Job {job_id} submitted")

# Poll for completion
status = poll_job(job_id)
while status == JobStatus.PENDING or status == JobStatus.RUNNING:
    time.sleep(0.1)
    status = poll_job(job_id)

# Fetch results
finished = fetch_finished_jobs(max_count=10)
for job_id, result in finished:
    print(f"Job {job_id}: {result.satisfiable}")
```

### Batch Job Submission

Submit multiple problems in parallel:

```python
solvers = []
job_ids = []

# Create multiple solver contexts
for problem_file in problem_files:
    solver = BatchingSolver()
    solver.load_from_file(problem_file)
    solvers.append(solver)

    # Submit async
    job_id = submit_solve(solver.ctx_id)
    job_ids.append(job_id)

# Wait for all to complete
while job_ids:
    finished = fetch_finished_jobs(max_count=len(job_ids))
    for job_id, result in finished:
        print(f"Problem {job_id}: {result.satisfiable}")
        job_ids.remove(job_id)
    time.sleep(0.1)
```

## Context Cloning

Clone solver contexts for parallel exploration of different branches.

### Basic Cloning

```python
from satellite_lab import BatchingSolver

# Create base solver
base_solver = BatchingSolver()
x = base_solver.bool_var()
y = base_solver.bool_var()
base_solver.add_clause([x.id, y.id])

# Clone for parallel exploration
solver1 = base_solver.clone()
solver2 = base_solver.clone()

# Explore different branches
solver1.add_clause([x.id])   # Assume x = true
solver2.add_clause([-x.id])  # Assume x = false

result1 = solver1.solve()
result2 = solver2.solve()
```

### Fork Multiple Contexts

```python
from satellite_lab import fork_context

# Fork into 4 clones
base_ctx = base_solver.ctx_id
clone_ids = fork_context(base_ctx, num_clones=4)

# Each clone can be modified independently
for i, ctx_id in enumerate(clone_ids):
    # Add branch-specific constraints
    # Submit async solve
    job_id = submit_solve(ctx_id)
```

## Branch Management

Satellite uses git-style branching with reference-counted garbage collection.

### Reference Counting

```rust
use satellite_branch::BranchManager;

let manager = BranchManager::new();

// Create branch
let branch_id = manager.create_branch(parent_id);

// Clone increments reference count
let branch_clone = manager.clone_branch(branch_id);

// Drop decrements reference count
// Branch freed when count reaches 0
```

### Lock-Free Status Tracking

```rust
// Check branch status without locking
let status = manager.get_status(branch_id);

match status {
    BranchStatus::Active => { /* solving */ }
    BranchStatus::Sat => { /* found solution */ }
    BranchStatus::Unsat => { /* no solution */ }
    BranchStatus::Pruned => { /* branch cut */ }
}
```

## Distributed Solving

Use the daemon for distributed solving across multiple machines.

### Starting the Daemon

```bash
# Start daemon on port 8080
satellite daemon --port 8080 --workers 8

# With GPU support
satellite daemon --port 8080 --workers 8 --gpu
```

### Client Connection

```python
import websocket
import json

# Connect to daemon
ws = websocket.create_connection("ws://localhost:8080")

# Submit problem
problem = {
    "clauses": [[1, 2], [-1, 3], [-2, -3]],
    "num_vars": 3
}
ws.send(json.dumps({"type": "solve", "problem": problem}))

# Receive result
result = json.loads(ws.recv())
print(f"Result: {result['satisfiable']}")

ws.close()
```

## GPU Acceleration

Satellite supports GPU-accelerated BCP using CUDA or HIP.

### Enabling GPU

```python
from satellite_lab import BatchingSolver

solver = BatchingSolver(use_gpu=True)
# GPU will be used for large problems automatically
```

### GPU Architecture

- **Persistent BCP Kernel**: Runs continuously on GPU
- **Warp-Cooperative Checking**: 32 threads check 32 clauses in parallel
- **Work Stealing**: Idle threads steal work from busy queues
- **Lock-Free MPMC Queue**: Efficient job distribution

### When GPU Helps

GPU acceleration is beneficial for:
- Large problems (>100K clauses)
- Long-running solves (>1 second)
- Batch processing multiple problems

GPU overhead makes it slower for small problems (<10K clauses).

### GPU Memory Management

```rust
use satellite_gpu::GpuWorker;

let mut gpu = GpuWorker::new()?;

// Allocate GPU memory
gpu.allocate_clauses(num_clauses)?;
gpu.allocate_variables(num_vars)?;

// Transfer data
gpu.upload_clauses(&clauses)?;

// Run BCP
let conflicts = gpu.run_bcp(&assignments)?;
```

## Best Practices

### 1. Choose the Right Strategy

- **Worker Pool**: Multiple independent problems
- **Context Cloning**: Parallel exploration of search space
- **Distributed**: Very large problems or limited local resources
- **GPU**: Large problems with many clauses

### 2. Worker Pool Sizing

```python
import os

# Use CPU count for I/O-bound tasks
num_workers = os.cpu_count()

# Use fewer workers for CPU-bound tasks to avoid contention
num_workers = max(1, os.cpu_count() - 1)

init_worker_pool(num_workers)
```

### 3. Avoid Over-Cloning

Cloning has memory cost:
- Each clone duplicates clause database
- Use assumptions instead of cloning when possible
- Limit clone depth in recursive algorithms

### 4. Load Balancing

Distribute work evenly:
```python
# Good: Similar-sized problems
problems = sorted(problems, key=lambda p: p.num_clauses)
for i, problem in enumerate(problems):
    worker_id = i % num_workers
    submit_to_worker(worker_id, problem)
```

## Performance Comparison

| Strategy | Overhead | Scalability | Use Case |
|----------|----------|-------------|----------|
| Worker Pool | Low | Linear | Independent problems |
| Cloning | Medium | Sub-linear | Search space exploration |
| Distributed | High | Super-linear | Very large problems |
| GPU | High | Problem-dependent | Large clause databases |
