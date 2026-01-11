# Python API Reference

This document describes the Python API provided by `satellite-lab`.

## Installation

```bash
cd py_src/satellite_lab
pip install -e .
```

## Quick Start

```python
from satellite_lab import BatchingSolver

# Create solver
solver = BatchingSolver()

# Create variables
x = solver.bool_var(name="x")
y = solver.bool_var(name="y")

# Add clauses: (x OR y) AND (NOT x OR y)
solver.add_clause([x.id, y.id])
solver.add_clause([-x.id, y.id])

# Solve
result = solver.solve()
print(f"SAT: {result.satisfiable}")
print(f"Model: {result.model}")
```

## Core Classes

### Solver

The base solver class (direct FFI to Rust).

```python
from satellite_lab import Solver

solver = Solver()
x = solver.bool_var(name="x")
solver.add_clause([x.id])
result = solver.solve()
```

### BatchingSolver

Optimized solver that batches clause additions.

```python
from satellite_lab import BatchingSolver

solver = BatchingSolver()

# Variables
x = solver.bool_var(name="x")
y = solver.bool_var(name="y")
reg = solver.batch_var(32, name="register")

# Clauses (cached locally, sent on solve)
solver.add_clause([x.id, y.id])
solver.add_clauses([[x.id], [-y.id, x.id]])

# Solve
result = solver.solve()

# Clone for parallel exploration
solver2 = solver.clone()
```

**Methods:**

| Method | Description |
|--------|-------------|
| `bool_var(name=None)` | Create boolean variable |
| `batch_var(dim, name=None)` | Create batch (bitvector) variable |
| `int_var(bits, name=None)` | Create integer variable |
| `add_clause(literals)` | Add single clause |
| `add_clauses(clauses)` | Add multiple clauses |
| `set_timeout(ms)` | Set timeout in milliseconds |
| `clone()` | Deep copy solver |
| `finalize()` | Send cached clauses to Rust |
| `solve()` | Solve the problem |
| `solve_with_assumptions(assumptions)` | Solve with temporary assumptions |
| `reset()` | Reset solver state |

**Properties:**

| Property | Description |
|----------|-------------|
| `num_vars` | Number of variables |
| `num_clauses` | Number of clauses (excluding pending) |
| `pending_clauses_count` | Pending clauses waiting to be sent |

### BitVec

Fixed-width bitvector for circuit-level operations.

```python
from satellite_lab import BatchingSolver, BitVec

solver = BatchingSolver()

# Create 8-bit bitvectors
a = BitVec(solver, 8)
b = BitVec(solver, 8)
result = BitVec(solver, 8)

# Add XOR constraint: result = a XOR b
a.xor(b, result)

# Solve
solver.solve()
```

**Methods:**

| Method | Description |
|--------|-------------|
| `bit(index)` | Get literal at bit index |
| `xor(other, result)` | Add XOR constraints |
| `and_(other, result)` | Add AND constraints |
| `or_(other, result)` | Add OR constraints |
| `not_(result)` | Add NOT constraints |

### CircuitBuilder

High-level circuit builder for generating SAT constraints.

```python
from satellite_lab import BatchingSolver, CircuitBuilder

solver = BatchingSolver()
cb = CircuitBuilder(solver)

# Create bitvectors
a = cb.new_bitvec(8)
b = cb.new_bitvec(8)

# Build circuit
xor_result = cb.add_xor(a, b)
and_result = cb.add_and(a, b)
mux_result = cb.add_ite(cond_lit, a, b)  # if-then-else

solver.solve()
```

**Methods:**

| Method | Description |
|--------|-------------|
| `new_bitvec(width)` | Allocate bitvector |
| `add_xor(a, b)` | XOR gate |
| `add_and(a, b)` | AND gate |
| `add_or(a, b)` | OR gate |
| `add_not(a)` | NOT gate |
| `add_ite(cond, then_val, else_val)` | If-then-else (mux) |

## Handle-Based Context API

For advanced use cases with multiple solver contexts.

```python
from satellite_lab import (
    create_context, destroy_context, add_clause,
    solve, fork_context, submit_solve, poll_job
)

# Create context
ctx = create_context()

# Add clauses
add_clause(ctx, [1, 2, -3])
add_clause(ctx, [-1, 2])

# Solve
result = solve(ctx)

# Fork into multiple clones for parallel exploration
clones = fork_context(ctx, 4)

# Async solving
job_id = submit_solve(ctx)
status = poll_job(job_id)  # 'pending', 'running', 'completed'

# Cleanup
destroy_context(ctx)
```

**Functions:**

| Function | Description |
|----------|-------------|
| `create_context()` | Create solver context, return handle |
| `destroy_context(ctx_id)` | Destroy context |
| `add_clause(ctx_id, literals)` | Add clause to context |
| `solve(ctx_id)` | Solve synchronously |
| `solve_with_assumptions(ctx_id, assumptions)` | Solve with assumptions |
| `submit_solve(ctx_id)` | Submit async job |
| `poll_job(job_id)` | Check job status |
| `fetch_finished_jobs(max_count)` | Get completed results |
| `fork_context(ctx_id, num_clones)` | Clone context |
| `add_clauses_buffer(ctx_id, buffer)` | Zero-copy clause injection |
| `init_worker_pool(num_workers)` | Initialize worker threads |

## Decorators

### @satellite_constraint

Mark a function as a Satellite constraint for ABI-OP compilation.

```python
from satellite_lab import satellite_constraint

@satellite_constraint
def my_xor(a: int, b: int) -> int:
    return a ^ b
```

## Examples

### Example 1: Simple SAT Problem

```python
from satellite_lab import BatchingSolver

solver = BatchingSolver()
x1, x2, x3 = [solver.bool_var(name=f"x{i}").id for i in range(1, 4)]

# (x1 OR x2) AND (NOT x2 OR x3) AND (NOT x1 OR NOT x3)
solver.add_clause([x1, x2])
solver.add_clause([-x2, x3])
solver.add_clause([-x1, -x3])

result = solver.solve()
if result.satisfiable:
    print("SAT:", result.model[:3])
else:
    print("UNSAT")
```

### Example 2: Incremental Solving with Assumptions

```python
solver = BatchingSolver()
x = solver.bool_var().id
y = solver.bool_var().id

solver.add_clause([x, y])
solver.add_clause([-x, y])

# Try with assumption x=true
result1 = solver.solve_with_assumptions([x])
print("With x=true:", result1.satisfiable)

# Try with assumption x=false
result2 = solver.solve_with_assumptions([-x])
print("With x=false:", result2.satisfiable)
```

### Example 3: Parallel BFS Exploration

```python
from satellite_lab import BatchingSolver

def parallel_solve(base_solver, decisions):
    results = []
    for assumption in decisions:
        solver = base_solver.clone()
        result = solver.solve_with_assumptions([assumption])
        results.append((assumption, result))
    return results

solver = BatchingSolver()
# ... add clauses ...

# Explore both branches in parallel
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(solver.clone().solve_with_assumptions, [1])
    future2 = executor.submit(solver.clone().solve_with_assumptions, [-1])
```

### Example 4: XOR Circuit

```python
from satellite_lab import BatchingSolver, BitVec

solver = BatchingSolver()

# 4-bit XOR
a = BitVec(solver, 4)
b = BitVec(solver, 4)
result = BitVec(solver, 4)

a.xor(b, result)

# Constrain result to 0b1010
for i in range(4):
    expected = (0b1010 >> i) & 1
    solver.add_clause([result[i]] if expected else [-result[i]])

sat_result = solver.solve()
print("SAT:", sat_result.satisfiable)
```
