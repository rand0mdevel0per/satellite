# Satellite Architecture & Design Philosophy

## The Philosophy of Freedom

Satellite is not just a SAT solver; it is a **Constraint Programming Platform** designed for maximum flexibility. Unlike traditional solvers that are rigid black boxes, Satellite is built on the philosophy of **programmable constraints**.

### Key Differentiators

1.  **Programmable Logic (ABI-OP)**:
    - Users can define custom constraints in Rust/C++/Python that are JIT-compiled to native machine code.
    - These constraints operate directly on the solver's internal state via a high-performance ABI.
    - **Freedom**: You are not limited to CNF. If you can write code to check it, Satellite can solve it.

2.  **Rich Type System**:
    - Satellite understands that the world isn't just Booleans.
    - Native support for:
        - `Integers` (finite domain)
        - `Vectors` (fixed-size arrays)
        - `Floats` (arbitrary precision)
    - **Freedom**: Model your problem naturally, not in standard CNF encoding.

3.  **Heterogeneous Compute**:
    - Seamlessly offload massive, parallelizable constraint checks to GPUs.
    - The `satellite-gpuk` kernel handles millions of clauses/constraints in parallel.
    - **Freedom**: Scale from a laptop to a GPU cluster without rewriting your model.

## System Architecture

### 1. The Core (satellite-kit & satellite-cdcl)
The heart of Satellite is a modern CDCL solver with:
- **State**: Managed by `ImplicationGraph` and `ClauseDatabase`.
- **Propagation**: A hybrid engine combining watched literals (Boolean) and ABI calls (Custom).
- **Learning**: 1-UIP conflict analysis that can "learn" from custom constraint failures.

### 2. The JIT Compiler (satellite-jit)
- **Input**: LLVM IR or supported high-level languages.
- **Process**:
    1.  Validates code safety (no illegal syscalls).
    2.  Optimizes using LLVM passes.
    3.  Compiles to native code in memory.
    4.  Links against the ABI.
- **Sanbox**: Windows JobObjects / Linux Seccomp ensure user constraints cannot crash the solver or steal data.

### 3. The GPU Worker (cpp_src/gpu_worker)
- **Role**: Massively parallel BCP (Boolean Constraint Propagation).
- **Design**:
    - **Persistent Kernel**: Stays resident on GPU to avoid launch overhead.
    - **Work Stealing**: Intra-warp and inter-block load balancing.
    - **Lock-Free Queue**: MPMC queue in unified memory for host-device comms.

### 4. The Daemon (satellite-daemon)
- **Role**: Distributed scheduler.
- **Protocol**: `rkyv` zero-copy serialization for microsecond-latency task submission.
- **Feature**: Supports "incremental solving" - add constraints to a running job.
