# Assumptions and Incremental Solving

Assumptions enable temporary constraints without permanently modifying the problem, allowing efficient exploration of solution spaces.

## Overview

**Assumptions** are literals that are assumed to be true for a single solve call. They allow:
- Testing hypotheses without adding permanent clauses
- Incremental solving with different constraints
- Efficient backtracking in search algorithms
- UNSAT core extraction for debugging

## Basic Usage

### Python

```python
from satellite_lab import BatchingSolver

solver = BatchingSolver()

# Create variables
x = solver.bool_var(name="x")
y = solver.bool_var(name="y")
z = solver.bool_var(name="z")

# Add base constraints
solver.add_clause([x.id, y.id])      # x OR y
solver.add_clause([-x.id, z.id])     # NOT x OR z

# Solve with assumption: x = true
result1 = solver.solve_with_assumptions([x.id])
print(f"With x=true: {result1.satisfiable}")

# Solve with assumption: x = false
result2 = solver.solve_with_assumptions([-x.id])
print(f"With x=false: {result2.satisfiable}")
```

### Rust

```rust
use satellite_kit::{Solver, SatResult};

let mut solver = Solver::new();

let x = solver.bool_var();
let y = solver.bool_var();
let z = solver.bool_var();

// Add base constraints
solver.add_clause(vec![x.positive_lit(), y.positive_lit()]);
solver.add_clause(vec![x.negative_lit(), z.positive_lit()]);

// Solve with assumption: x = true
match solver.solve_with_assumptions(&[x.positive_lit()])? {
    SatResult::Sat(model) => println!("With x=true: SAT"),
    SatResult::Unsat => println!("With x=true: UNSAT"),
    _ => {}
}

// Solve with assumption: x = false
match solver.solve_with_assumptions(&[x.negative_lit()])? {
    SatResult::Sat(model) => println!("With x=false: SAT"),
    SatResult::Unsat => println!("With x=false: UNSAT"),
    _ => {}
}
```

## Use Cases

### 1. Binary Search

Find the minimum value that satisfies constraints:

```python
def binary_search_min(solver, var, low, high):
    """Find minimum value for var that satisfies constraints."""
    best = None

    while low <= high:
        mid = (low + high) // 2
        # Assume var <= mid
        assumptions = encode_less_equal(var, mid)

        result = solver.solve_with_assumptions(assumptions)
        if result.satisfiable:
            best = mid
            high = mid - 1  # Try smaller
        else:
            low = mid + 1   # Need larger

    return best
```

### 2. Configuration Testing

Test different system configurations:

```python
# Test all combinations of feature flags
features = [solver.bool_var(name=f"feature_{i}") for i in range(5)]

for config in itertools.product([True, False], repeat=5):
    assumptions = [f.id if enabled else -f.id
                   for f, enabled in zip(features, config)]

    result = solver.solve_with_assumptions(assumptions)
    if result.satisfiable:
        print(f"Config {config} is valid")
```

### 3. Constraint Debugging

Identify conflicting constraints:

```python
# Add constraints one by one with assumptions
constraints = [
    [x.id, y.id],
    [-x.id, z.id],
    [-y.id, -z.id],
    [x.id, -z.id]
]

for i, clause in enumerate(constraints):
    solver_copy = solver.clone()
    # Assume this clause is NOT satisfied
    negated = [-lit for lit in clause]

    result = solver_copy.solve_with_assumptions(negated)
    if not result.satisfiable:
        print(f"Constraint {i} is essential")
```

## Incremental Solving Patterns

### Pattern 1: Progressive Refinement

Add constraints incrementally and test:

```python
solver = BatchingSolver()

# Base problem
x = solver.bool_var()
y = solver.bool_var()
solver.add_clause([x.id, y.id])

# Test with additional constraint
result1 = solver.solve_with_assumptions([x.id])
if result1.satisfiable:
    # Make it permanent
    solver.add_clause([x.id])
```

### Pattern 2: Backtracking Search

Explore solution space with backtracking:

```python
def backtrack_search(solver, decisions, depth=0):
    """DFS with assumptions for backtracking."""
    if depth >= len(decisions):
        return solver.solve()

    var = decisions[depth]

    # Try positive
    result = solver.solve_with_assumptions([var.id])
    if result.satisfiable:
        return backtrack_search(solver, decisions, depth + 1)

    # Try negative
    result = solver.solve_with_assumptions([-var.id])
    if result.satisfiable:
        return backtrack_search(solver, decisions, depth + 1)

    return None  # Backtrack
```

### Pattern 3: Constraint Relaxation

Relax constraints to find near-solutions:

```python
# Soft constraints as assumptions
soft_constraints = [
    [x.id],      # Prefer x = true
    [y.id],      # Prefer y = true
    [-z.id]      # Prefer z = false
]

# Try with all soft constraints
result = solver.solve_with_assumptions([c[0] for c in soft_constraints])

if not result.satisfiable:
    # Relax one constraint at a time
    for i in range(len(soft_constraints)):
        relaxed = [c[0] for j, c in enumerate(soft_constraints) if j != i]
        result = solver.solve_with_assumptions(relaxed)
        if result.satisfiable:
            print(f"Solution found by relaxing constraint {i}")
            break
```

## UNSAT Core Extraction

When solving with assumptions results in UNSAT, you can extract the minimal set of assumptions that caused the conflict.

### Python

```python
solver = BatchingSolver()
solver.enable_unsat_core()

# Add base constraints
x = solver.bool_var()
y = solver.bool_var()
z = solver.bool_var()

solver.add_clause([x.id, y.id])
solver.add_clause([-x.id, z.id])
solver.add_clause([-y.id, -z.id])

# Solve with conflicting assumptions
assumptions = [x.id, -y.id, z.id]
result = solver.solve_with_assumptions(assumptions)

if not result.satisfiable:
    core = solver.get_unsat_core()
    print(f"UNSAT core: {core}")
    # Core contains subset of assumptions that caused conflict
```

### Rust

```rust
use satellite_cdcl::CdclSolver;

let mut solver = CdclSolver::new(&problem);
solver.enable_unsat_core();

let assumptions = vec![x.positive_lit(), y.negative_lit(), z.positive_lit()];
match solver.solve_with_assumptions(&assumptions)? {
    SatResult::Unsat => {
        if let Some(core) = solver.get_unsat_core() {
            println!("UNSAT core clause IDs: {:?}", core);
        }
        if let Some(clauses) = solver.get_unsat_core_clauses() {
            println!("UNSAT core clauses:");
            for clause in clauses {
                println!("  {:?}", clause);
            }
        }
    }
    _ => {}
}
```

## Best Practices

### 1. Minimize Assumptions

Use the smallest set of assumptions necessary:
- Fewer assumptions = faster solving
- Easier to understand UNSAT cores
- Better cache locality in CDCL solver

### 2. Reuse Solver State

Don't recreate solvers for each assumption set:
```python
# Good: Reuse solver
solver = BatchingSolver()
# Add base constraints once
for assumptions in test_cases:
    result = solver.solve_with_assumptions(assumptions)

# Bad: Recreate solver
for assumptions in test_cases:
    solver = BatchingSolver()  # Wasteful!
    result = solver.solve_with_assumptions(assumptions)
```

### 3. Clone for Parallel Exploration

Use cloning for independent branches:
```python
base_solver = BatchingSolver()
# Add base constraints

# Clone for parallel exploration
solvers = [base_solver.clone() for _ in range(4)]
# Each clone can be solved independently
```

### 4. Enable UNSAT Core Selectively

UNSAT core tracking has overhead:
- Enable only when needed for debugging
- Disable for production solving
- Use for constraint analysis and debugging

## Performance Considerations

- **Assumption overhead**: ~5-10% compared to permanent clauses
- **UNSAT core tracking**: ~10-20% overhead when enabled
- **Cloning cost**: O(clauses) memory, O(1) time (reference counted)
- **Best for**: Testing hypotheses, incremental solving, backtracking search

## See Also

- [Parallel Solving](parallel.md) - Parallel exploration with assumptions
- [UNSAT Core](../advanced/unsat-core.md) - Detailed UNSAT core extraction
- [Core API](../api/core-api.md) - Solver API reference

