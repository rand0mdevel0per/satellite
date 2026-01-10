# Satellite: High-Performance Parallel SAT Solver with Advanced Type System

## Project Overview

**Satellite** is a next-generation constraint satisfaction solver that extends traditional CDCL-based SAT solving with a rich type system, heterogeneous computing (CPU/GPU), and programmable constraints. The name reflects its intended deployment model: large-scale compute servers (satellites) serving multiple client endpoints in research institutions and universities.

### Core Innovation Points

1. **Hierarchical Type System**: From boolean primitives to batches, integers, vectors, and arbitrary-precision floats
2. **Branch-Based Parallel Model**: Git-style constraint branching with reference-counted garbage collection
3. **Heterogeneous Computing**: Unified CPU/GPU task execution through lock-free MPMC queues
4. **Programmable Constraints (ABI-OP)**: User-defined constraints compiled via LLVM JIT with automatic optimization
5. **Research-Friendly Ecosystem**: Jupyter integration, Python bindings, modern IDE with LSP support

### Target Applications

- **Cryptography**: Key recovery, hash collision, protocol verification, side-channel analysis
- **EDA (Electronic Design Automation)**: Boolean equivalence checking, timing analysis
- **Financial Optimization**: Integer programming for portfolio optimization
- **Algorithm Research**: General constraint satisfaction problems with complex domains

---

## Architecture Components

### Component Structure

```
[rust] satellite-kit       → Core solver library and API
[rust] satellite-cli       → Full-featured command-line interface
[rust] satellite-daemon    → Server process for distributed solving
[tauri+react] satellite-ide → Modern IDE with LSP, syntax highlighting, debugger
[python] satellite-lab     → Python wrapper for researchers
[jupyter] satellite-jupyter → Jupyter kernel integration
```

### Data Flow

```
User Input (Python/IDE/CLI)
    ↓
satellite-kit (constraint compilation)
    ↓
Advanced-CNF (.json format)
    ↓
CDCL Solver (branch-based parallelization)
    ↓
CPU/GPU Worker Pool (MPMC job queue)
    ↓
Result + Profiling Data
```

---

## Type System

### Type Hierarchy

```
bool                    → Single boolean variable
batch[dim]              → Bus-like batch of booleans (similar to Verilog)
int                     → Integer stored as big-endian in batch
vec[batch[dim1], dim2]  → Vector of batches
float                   → Arbitrary-precision float (each precision segment in separate batch, aggregated as vec)
```

### Type Operations

#### Batch Operations
- **Bitwise Operations**: Execute on entire batch (AND, OR, XOR, NOT, shifts)
- **Unrolling**: Execute operation on each boolean in batch individually
- **Double Unrolling**: Execute on each batch within a vector

#### Integer Operations
- Implemented via batch/vec manipulation
- Big-endian storage for efficient arithmetic
- Supports standard arithmetic: +, -, *, /, %, comparisons

#### Float Operations
- **Arbitrary Precision via Decimal**: Uses Rust `decimal` crate for unlimited precision
- **Pre-running Analysis**: Automatically determines required precision based on user's actual usage
- **Segmented Storage**: Each precision segment stored in separate batch, aggregated as vector
- **Special Algorithms**: Custom algorithms enable unlimited precision on big-endian float representation

### Advanced Constraints

The solver provides high-level constraint primitives optimized for specific domains:
- XOR constraints (cryptographic applications)
- Cardinality constraints (at-most-k, exactly-k)
- Pseudo-boolean constraints
- Statistical random number generation (range-constrained via statistical methods)

---

## Constraint Compilation

### ABI-OP System

User-defined constraints can be written in compilable languages (Rust, C++, etc.) and integrated through the ABI-OP trait system.

#### Compilation Pipeline

```
User Code (Rust/C++/etc.)
    ↓
Language Compiler → LLVM IR
    ↓
LLVM Optimization (O3) via inkwell
    ↓
JIT Compilation
    ↓
    ├─ CPU: LLVM toolchain
    └─ GPU: NVIDIA/AMD toolchain from inkwell output
    ↓
Control Flow Analysis & Segmentation
    ↓
Multi-threaded Async Modules (duration-based separation)
    ↓
Props Generation → Namespace Loading
```

#### Control Flow Segmentation

- **Lifetime Analysis**: Analyze variable lifetimes (duration) in the compiled LLVM IR
- **Non-overlapping Segments**: Identify segments with non-overlapping lifetimes for parallel execution
- **Automatic Parallelization**: Split into independent async modules that can run concurrently

#### Truth Table Optimization

When pre-running analysis detects that an ABI-OP's maximum input dimension is acceptably small:
- **Exhaustive Enumeration**: Generate complete truth table by brute-force evaluation
- **Lookup Performance**: Runtime execution becomes simple table lookup (orders of magnitude faster than JIT call)
- **Cache Integration**: Truth tables stored in code hash cache for reuse

#### IDE Integration

- **Language Server**: Frontend's LSP verifies function existence before execution
- **Type Checking**: Validates ABI-OP signatures against constraint type system
- **Real-time Feedback**: Syntax errors and type mismatches shown in IDE

#### Sandboxing

User code runs in a custom sandbox (to be integrated from existing implementation) to prevent:
- Filesystem access outside allowed directories
- Network operations
- System calls that could compromise host security
- Resource exhaustion attacks

---

## CDCL Implementation

### Parallel Job Decomposition

Traditional CDCL stages are decomposed into fine-grained jobs:

#### 1. Decision Phase
- **Heuristic Computation**: Multiple heuristics (VSIDS, EVSIDS, LRB) run in parallel
- **Weighted Gating**: Results combined via weighted sum, filtered by confidence threshold
- **Branch Generation**: If gating passed, generate constraint branches (not decision tree branches)

#### 2. Boolean Constraint Propagation (BCP)
- **Granularity**: Each job checks 32 clauses (exact warp size for GPU)
- **Lock-free Vector**: All clauses stored in lock-free vector structure
- **Parallel Checking**: Multiple jobs check different clause batches concurrently
- **Job-count Trigger**: Atomic counter triggers aggregation job when all checks complete

**Job-count Trigger Implementation**:
```
atomic_inc(counter)
if CAS(counter, total_jobs, 0) succeeds:
    trigger aggregation job
```
Only the last completing job triggers aggregation via CAS.

#### 3. Conflict Analysis
- **Implication Graph**: Constructed incrementally as conflicts discovered
- **UIP Resolution**: First UIP (Unique Implication Point) found via graph traversal
- **Learned Clause Generation**: Resolution steps produce new clause
- **Clause Database Update**: Global skiplist for learned clauses (lock-free)

#### 4. Backtracking
- Handled by branch failure mechanism (see Branch Model below)

#### 5. Database Maintenance
- **Filter Rate Tracking**: Each BCP records clause trigger frequency
- **Weighted Averaging**: Filter rates weighted-averaged and stored via CAS
- **Deletion Policy**: Clauses with low filter rates (infrequently triggered) are deleted
- **Memory Overflow Handling**:
  - RAM full → spill to disk
  - Disk access patterns optimized for sequential I/O
  - Warning issued to user (recommends SSD)

---

## Branch Model

### Git-Style Constraint Branching

Unlike traditional CDCL decision tree branching, Satellite's branches represent **semantic constraint splits** (e.g., XOR expansions, if-else in ABI-OP).

```
Parent Branch (refcount=3)
    ├─ Child A: constraint variant 1
    ├─ Child B: constraint variant 2
    └─ Child C: constraint variant 3
```

### Branch Lifecycle

#### Creation
- **Fork on Constraint Split**: When encountering multi-branch constraints (XOR, conditional, etc.)
- **Reference Counting**: Parent's refcount = number of children
- **Inheritance**: Child inherits parent's state (assignments, learned clauses)

#### Failure Propagation

**Child → Parent**:
```rust
on_child_fail() {
    let new_count = atomic_dec(parent.refcount);
    if new_count == 0 {
        mark_parent_failed();
        spawn_job(check_grandparent);
    }
}
```
Last failing child marks parent as failed and spawns job to check grandparent.

**Parent → Child**:
- When parent fails, all descendants immediately marked as failed
- Implemented via skiplist-based branch status tracking
- Jobs check branch status before execution; skip if branch failed

#### State Management

- **Skiplist Storage**: All branch states stored in lock-free skiplist
- **Job Validation**: Each job checks `branch_state` before execution
- **Graceful Termination**: Workers skip jobs belonging to failed branches

---

## Heterogeneous Computing

### MPMC Queue Architecture

#### Lock-Free Implementation
- **Atomic Stack Pointer**: CAS-based stack head pointer
- **Pointer-Linked Stack**: Nodes linked via pointers for O(1) push/pop
- **No Mutex Contention**: Pure atomic operations, no locks

#### Priority Queues

**4-Level Priority System**:
1. High priority (fresh branches)
2. Medium-high priority
3. Medium-low priority
4. Low priority (stale branches)

**Priority Assignment**:
- Based on branch confidence: `confidence = f(jobs_completed, success_rate, time_elapsed)`
- Exponential mapping to priority queues
- Dynamic demotion: older branches with fewer successes get lower priority

**Queue Selection (Worker Side)**:
```rust
let rand = xorshift(job_data ^ txid ^ warp_id);  // Fast PRNG
let queue_id = weighted_sample(rand, [p1, p2, p3, p4]);
let job = queues[queue_id].pop();
```
- **Xorshift PRNG**: Fast, lightweight random number generation using job data as seed
- **Weighted Sampling**: Higher priority queues have higher selection probability
- **Starvation Prevention**: Even low-priority queues occasionally selected via randomization

### CPU Worker Pool

#### Implementation (Rust)
- **Thread Pool**: Fixed-size pool of worker threads (typically = num_cores)
- **Job Execution Loop**:
  ```rust
  loop {
      let job = mpmc.pop_with_priority();
      if job.branch.is_failed() { continue; }
      execute(job);
  }
  ```

#### Optimizations
- **CPU Cache Locality**: Jobs grouped by branch to improve cache hit rate
- **NUMA Awareness**: Workers pinned to NUMA nodes when available

### GPU Worker Pool

#### Implementation (HIP + CUDA)
- **Warp as Unit**: Each warp (32 threads on NVIDIA, 64 on AMD) executes one job
- **Lane 0 Fetches Job**: First thread in warp pops job from MPMC
- **Shared Memory Distribution**: Job data distributed to all lanes via shared memory
- **Warp-Level Primitives**: Use `__shfl`, `__ballot`, `__all_sync` for intra-warp communication

#### BCP Job on GPU (32 clauses)
```cuda
__shared__ bool results[32];
results[threadIdx.x] = check_clause(clause_id);
__syncthreads();

bool all_satisfied = __all_sync(0xffffffff, results[threadIdx.x]);
// Shuffle-based reduction for aggregation
```

#### Memory Constraints
- **Shared Memory**: 48KB per block (NVIDIA) / 64KB (AMD)
- **Job Splitting**: If job exceeds shared memory, split into 2 jobs and re-enqueue
- **CPU Fallback**: Single-clause jobs that still exceed memory sent to CPU queue
  - GPU → CPU: Enqueue to CPU-specific queue
  - CPU processes and enqueues callback job to GPU

#### Work Stealing (Warp-Level)
- **Single-Threaded Jobs**: Some jobs only need 1 thread (e.g., simple checks)
- **Idle Lanes**: Remaining 31 threads can steal lightweight jobs from MPMC
- **Warp Synchronization**: Careful to avoid divergence; stealing only for homogeneous jobs

---

## Memory Management

### MVCC (Multi-Version Concurrency Control)

#### Hot Path Optimization
Frequently updated data structures (VSIDS scores, clause activity) use MVCC to avoid lock contention.

#### Transaction ID Allocation
- **Hardware-Based**: Each hardware thread (CPU core / GPU SM) has unique hardware ID
- **TXN-ID**: Combination of `hwid` + `logical_timestamp`
- **Version Chain**: Each write creates new version, linked to previous

```
v1 (txid=100) → v2 (txid=105) → v3 (txid=112) → ...
```

#### Read Operations
- Reader uses its own `txid` to determine visible version
- Snapshot isolation: reads consistent view despite concurrent writes

#### Write Operations (Copy-on-Write)
- Writer creates new version, atomically updates version pointer
- Old versions remain until garbage collected

#### Garbage Collection
- **Refcount-Based**: Based on minimum active transaction ID
- **No Per-Version Atomics**: Instead of per-version refcount, track global min(txid)
- **Epoch Reclamation**: Versions older than min(txid) are safe to reclaim

```rust
if version.txid < global_min_active_txid {
    reclaim(version);
}
```

### Critical Path (Atomic Operations)

Some operations cannot use MVCC due to correctness requirements:

#### Reference Counting (Branch Lifecycle)
- `atomic_dec` on parent refcount when child fails
- CAS to detect refcount==0 and trigger parent failure

#### Job-Count Triggers
- `atomic_inc` as jobs complete
- CAS to detect completion and trigger aggregation

#### Clause Database Insertions
- Lock-free skiplist for learned clauses
- Multiple workers insert concurrently without coordination

---

## Data Structures

### Lock-Free Vector (Clause Storage)

Original clauses and learned clauses stored in growable lock-free vector:
- **Chunked Allocation**: Vector grows in chunks to avoid reallocation
- **Atomic Append**: New clauses added via atomic pointer updates
- **Parallel Read**: Multiple workers read different ranges concurrently

**Lookup Performance**:
- O(1) random access
- Better cache locality than skiplist
- Trade-off: No efficient deletion (but clauses marked inactive via filter rate)

### Lock-Free Skiplist (Branch State)

Branch status tracking uses skiplist for efficient updates:
- **Insert**: O(log n) expected, fully concurrent
- **Lookup**: O(log n), read-only operations are wait-free
- **No Global Lock**: Uses atomic CAS for node insertion

---

## Client Caching System

### Code Hash Cache

Compiled ABI-OP results cached on client side:

#### Hash Generation
```
code_content → SHA256 → code_hash
code_hash → Bloom filter (fast negative check)
code_hash → Cache lookup (actual compiled binary)
```

#### Bloom Filter
- **False Positive Handling**: Bloom says "exists" → check cache → miss → compile and cache
- **No False Negatives**: If Bloom says "not exists", guaranteed cache miss
- **Fixed Size**: No dynamic resizing (simplicity over optimality)

#### Cache Structure
```
cache/
  ├─ <code_hash1>.so      (compiled shared object)
  ├─ <code_hash2>.wasm    (WebAssembly for browser IDE)
  └─ bloom_filter.bin     (serialized filter)
```

### Truth Table Cache

Small ABI-OPs with exhaustively enumerated truth tables:
```
<code_hash>_truth_table.bin → serialized truth table
```
Loaded directly for instant lookup during solving.

---

## User Interfaces

### satellite-kit (Rust Library)

Core API for programmatic access:

```rust
use satellite_kit::*;

let mut solver = Solver::new();
let x = solver.bool_var();
let y = solver.batch_var(32); // 32-bit batch

solver.add_constraint(x.and(y[0]));
solver.add_abi_constraint("my_func", &[x, y]);

match solver.solve() {
    SatResult::Sat(model) => println!("Solution: {:?}", model),
    SatResult::Unsat => println!("No solution"),
    SatResult::Unknown => println!("Timeout/Resource limit"),
}
```

### satellite-cli (Command Line)

Full-featured CLI with:
- File input: `satellite solve input.sat --format advanced-cnf`
- Profiling: `--profile output.json`
- Incremental solving: `--incremental --add-constraint "x > 0"`
- Distributed solving: `--daemon ws://server:8080`

### satellite-ide (Tauri + React)

Modern IDE features:
- **LSP Integration**: Real-time syntax checking, autocomplete, go-to-definition
- **Debugger**: Set breakpoints on constraints, step through branch exploration
- **Visualizer**: Tree view of branch hierarchy, clause database inspector
- **Performance**: Modeled after JetBrains RustRover/DataGrip
  - Incremental analysis (only reanalyze changed regions)
  - Cursor position + sight range determines analysis scope
  - Handles massive codebases efficiently

### satellite-lab (Python)

Research-friendly Python wrapper:

```python
from satellite_lab import Solver

solver = Solver()
x = solver.int_var(32, name="x")
y = solver.float_var(precision=128, name="y")

solver.add_constraint(x + y == 10.5)

@solver.abi_constraint
def custom(a, b):
    return a[0] ^ b[1]  # XOR constraint

solver.add_constraint(custom(x.to_batch(), y.to_batch()))
```

#### FFI Optimization
- **Batch API**: `solver.add_constraints([c1, c2, ...])` for bulk additions
- **Python-Side Buffering**: Constraints buffered in Python, flushed on `finalize()`
- **NumPy Integration**: `np.array` ↔ `Batch` conversion via zero-copy views
- **SQL-Style Joins**: Results returned as structured data with join operations

### satellite-jupyter (Jupyter Kernel)

Jupyter-specific features:
- **Progress Bars**: `tqdm` integration showing branch exploration progress
- **Real-time Logging**: Live updates of best-known solution
- **Inline Visualization**: Branch tree rendered as interactive widget
- **Interrupt Handling**: Ctrl+C gracefully stops solving, preserves state

#### Example Notebook
```python
%%satellite
constraint sudoku_rules {
    forall i, j: grid[i][j] in 1..9
    forall i: all_different(grid[i])
    forall j: all_different(grid[:, j])
    # ... more rules
}

result = solve(timeout=60)
visualize(result)
```

### satellite-daemon (Server)

Long-running server process for:
- **Distributed Solving**: Multiple clients submit jobs to central daemon
- **Resource Management**: Allocates GPU/CPU resources across jobs
- **Persistent State**: Checkpoint/resume for long-running solves
- **WebSocket API**: Real-time bidirectional communication with clients

---

## Edge Cases and Error Handling

### Incremental Constraint Addition

User adds constraints after initial solve:

```python
solver.add_constraint(x > 0)
solver.solve()  # First solve

# User adds new constraint
solver.add_constraint(x < 10)
solver.solve()  # Incremental solve
```

**Strategy**: Incremental solving
- Reuse learned clauses from first solve (those still valid)
- Only process new constraints
- Restart branch exploration from modified points

### Memory Exhaustion

When RAM insufficient:
- **Warning**: "Estimated 100GB required, 16GB available. Enabling disk cache (80% performance penalty)."
- **Automatic Spillover**: Learned clauses → RAM → Disk
- **Sequential I/O**: Optimized for SSD performance
- **No Prompt**: Automatic fallback (no user interaction during solve)

### GPU Errors

GPU driver crash / OOM / CUDA version mismatch:
- **No Fallback**: Report error immediately, let user resolve
- **Error Message**: "GPU error: CUDA out of memory. Solution: Reduce problem size or use CPU-only mode with --no-gpu"
- **Rationale**: GPU issues are configuration problems, not transient failures

### Result Reproducibility

Multiple solves of identical problem produce identical results:
- **Deterministic PRNG**: Seed xorshift with problem hash, not wallclock time
- **Branch Ordering**: Consistent branch ID generation
- **Cache Snapshot**: Include cache state in reproducibility snapshot

**Snapshot Contents**:
```json
{
    "problem_hash": "abc123...",
    "branch_tree": {...},
    "learned_clauses": [...],
    "prng_state": {...},
    "solver_config": {...}
}
```

Restoring snapshot produces bitwise-identical solving trace.

---

## Profiling and Debugging

### Snapshot System

Real-time profiling via snapshot transmission to client:

#### Snapshot Contents
```json
{
    "timestamp": "2025-01-10T12:34:56Z",
    "branches": [
        {
            "id": "branch_001",
            "parent_id": "branch_000",
            "depth": 5,
            "status": "active",
            "jobs_completed": 1234,
            "jobs_pending": 56,
            "priority_queue": 2,
            "time_elapsed_ms": 3200,
            "memory_usage_mb": 128,
            "clauses_learned": 89,
            "filter_rate": 0.73,
            "confidence": 0.85
        }
    ],
    "global_stats": {
        "total_branches": 45,
        "active_branches": 12,
        "failed_branches": 33,
        "cpu_workers": 16,
        "gpu_workers": 8,
        "gpu_utilization": 0.87,
        "learned_clauses_total": 45672,
        "memory_usage_mb": 2048,
        "disk_usage_mb": 15360
    },
    "top_slow_constraints": [
        {"name": "custom_xor", "avg_time_us": 1234, "calls": 56789}
    ]
}
```

#### Update Frequency
- **Default**: Every 1 second
- **On-Demand**: Client can request immediate snapshot
- **Throttling**: Rate-limited to prevent network saturation

#### Visualization (IDE)
- **Tree View**: Interactive branch hierarchy
- **Flame Graph**: Time spent per constraint type
- **Timeline**: Branch creation/failure events over time
- **Resource Usage**: CPU/GPU/Memory utilization graphs

---

## Performance Optimizations

### Summary of Key Techniques

1. **MVCC on Hot Paths**: Reduces contention on VSIDS scores, clause activity
2. **Truth Table Precomputation**: O(1) lookup for small ABI-OPs
3. **LLVM O3 + JIT**: Maximal optimization of user constraints
4. **Lock-Free Data Structures**: MPMC queue, skiplist, vector
5. **GPU Warp-Level Parallelism**: 32 clauses per job, shuffle-based reduction
6. **Weighted Priority Queues**: Focuses computation on promising branches
7. **Incremental Solving**: Reuses learned clauses across solve() calls
8. **Cache Aggressive**: Code hash cache, truth table cache
9. **Parallel Heuristics**: Multiple decision heuristics computed concurrently
10. **Bloom Filter**: O(1) cache existence check before expensive hash lookup

---

## File Formats

### Advanced-CNF (.json)

Unified internal representation:

```json
{
    "variables": [
        {"id": 0, "type": "bool", "name": "x"},
        {"id": 1, "type": "batch", "dim": 32, "name": "y"},
        {"id": 2, "type": "int", "bits": 32, "name": "z"},
        {"id": 3, "type": "float", "precision": 128, "name": "w"}
    ],
    "clauses": [
        {"literals": [1, -2, 3], "type": "original"},
        {"literals": [-1, 4], "type": "learned", "lbd": 2}
    ],
    "abi_constraints": [
        {
            "name": "my_custom",
            "inputs": [0, 1],
            "code_hash": "abc123...",
            "cached": true
        }
    ]
}
```

All input formats (DIMACS CNF, SMT-LIB, custom DSL) compiled to Advanced-CNF.

### DIMACS Import

Standard DIMACS CNF files directly supported:
```
satellite solve problem.cnf --format dimacs
```
Automatically converted to Advanced-CNF internally.

---

## Development Timeline

### Approach
**Solo development** with assistance from:
- **Antigravity**: (assumed to be AI coding assistant)
- **Kilo-code**: (assumed to be code generation tool)

### Roadmap
**Random development order** (no strict timeline):
1. **satellite-kit MVP**: Basic CDCL + branch model + CPU-only
2. **GPU Integration**: HIP/CUDA worker pool + job system
3. **Type System**: Batch/int/vec/float implementation
4. **ABI-OP**: LLVM pipeline + JIT + sandboxing
5. **CLI**: Full-featured command-line interface
6. **Python Bindings**: PyO3 wrapper for satellite-lab
7. **IDE**: Tauri + React + LSP
8. **Jupyter Kernel**: IPython integration
9. **Daemon**: Server mode for distributed solving

### Testing & Release
- **CI/CD**: TeamCity for builds, YouTrack for issue tracking
- **Benchmarks**: SAT Competition standard benchmarks + cryptographic problems
- **Documentation**: Examples gallery (AES, sudoku, N-queens, etc.)
- **Publication**: Paper submission (optional), MIT license open-source
- **Maintenance**: "Fix bugs when I feel like it" approach

---

## Comparison with Existing Solvers

### Strengths
- **Flexibility**: Rich type system allows natural problem encoding
- **Programmability**: ABI-OP enables domain-specific optimizations
- **Scalability**: Heterogeneous computing leverages modern hardware
- **User Experience**: IDE + Jupyter integration lowers barrier to entry

### Weaknesses
- **Type System Overhead**: Additional abstraction layers vs. raw CNF
- **JIT Warm-up**: First ABI-OP call incurs compilation latency
- **GPU Memory Limits**: Large problems may exceed VRAM capacity
- **Maturity**: New solver vs. decades of CDCL optimizations in established solvers

### Target Niches
Not competing directly with MiniSat/CryptoMiniSat on pure speed, but targeting:
- Problems naturally expressed with high-level types
- Domains requiring custom constraints (cryptography, EDA)
- Research workflows needing rapid prototyping
- Users preferring Python/Jupyter over raw CNF files

---

## Open Questions for Future Work

1. **SMT Theory Integration**: Should floats use SMT theory solvers instead of bit-blasting?
2. **Portfolio Solving**: Run multiple solving strategies in parallel branches?
3. **Machine Learning Heuristics**: Train neural networks to predict good branch choices?
4. **Proof Generation**: Emit UNSAT proofs for verification (DRAT/DRUP)?
5. **Distributed Solving**: Multi-node cluster support beyond single daemon?

---

## License and Attribution

- **License**: MIT
- **Repository**: (To be published on GitHub)
- **Contact**: (Maintainer information TBD)

---

## Glossary

- **CDCL**: Conflict-Driven Clause Learning
- **MPMC**: Multi-Producer Multi-Consumer (queue)
- **MVCC**: Multi-Version Concurrency Control
- **ABI-OP**: Application Binary Interface Operation (user-defined constraint)
- **LSP**: Language Server Protocol
- **BCP**: Boolean Constraint Propagation
- **UIP**: Unique Implication Point
- **JIT**: Just-In-Time (compilation)
- **LLVM**: Low-Level Virtual Machine (compiler infrastructure)
- **Warp**: Group of 32 threads on NVIDIA GPUs (64 on AMD)

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-10  
**Status**: Pre-implementation Specification