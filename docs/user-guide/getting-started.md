# Getting Started

Welcome to Satellite! This guide will help you get up and running quickly.

## Installation

### Rust (Core)

```bash
git clone https://github.com/yourusername/satellite
cd satellite
cargo build --release
```

### Python (satellite-lab)

```bash
cd py_src/satellite_lab
pip install -e .
```

### CLI

```bash
cargo install --path crates/satellite-cli
```

## Quick Start

### Python Example

```python
from satellite_lab import BatchingSolver

# Create a solver
solver = BatchingSolver()

# Create some boolean variables
x = solver.bool_var(name="x")
y = solver.bool_var(name="y")
z = solver.bool_var(name="z")

# Add clauses (CNF format)
# (x OR y) AND (NOT x OR z) AND (NOT y OR NOT z)
solver.add_clause([x.id, y.id])      # x OR y
solver.add_clause([-x.id, z.id])     # NOT x OR z  
solver.add_clause([-y.id, -z.id])    # NOT y OR NOT z

# Solve!
result = solver.solve()

if result.satisfiable:
    print("Found a solution!")
    print(f"x = {result.model[0]}")
    print(f"y = {result.model[1]}")
    print(f"z = {result.model[2]}")
else:
    print("No solution exists")
```

### CLI Example

```bash
# Convert DIMACS to JSON
satellite convert input.cnf output.json

# Solve a single problem
satellite solve problem.json --output result.json

# Batch solve multiple problems
satellite batch --input-dir ./problems --output-dir ./results --workers 4
```

### Rust Example

```rust
use satellite_kit::{Solver, SatResult};

fn main() -> anyhow::Result<()> {
    let mut solver = Solver::new();
    
    let x = solver.bool_var();
    let y = solver.bool_var();
    
    // Add clause: x OR y
    solver.add_clause(vec![
        x.positive_lit(),
        y.positive_lit()
    ]);
    
    // Add clause: NOT x OR NOT y
    solver.add_clause(vec![
        x.negative_lit(),
        y.negative_lit()
    ]);
    
    match solver.solve()? {
        SatResult::Sat(model) => {
            println!("SAT: x={}, y={}", model[0], model[1]);
        }
        SatResult::Unsat => {
            println!("UNSAT");
        }
        SatResult::Unknown(reason) => {
            println!("Unknown: {}", reason);
        }
    }
    
    Ok(())
}
```

## Core Concepts

### Variables and Literals

- **Variable**: A boolean variable (can be true or false)
- **Literal**: A variable or its negation
  - Positive literal: `var.id` (variable is true)
  - Negative literal: `-var.id` (variable is false)

### Clauses

Clauses are disjunctions (OR) of literals:
- `[1, 2, -3]` means: x₁ OR x₂ OR NOT x₃
- The problem is satisfiable if all clauses can be made true simultaneously

### CNF (Conjunctive Normal Form)

SAT problems are expressed as a conjunction (AND) of clauses:
```
(x₁ OR x₂) AND (NOT x₁ OR x₃) AND (NOT x₂ OR NOT x₃)
```

### Variable Types

| Type | Description | Use Case |
|------|-------------|----------|
| `bool` | Single boolean | Simple constraints |
| `batch[N]` | N-bit bitvector | Registers, bitwise ops |
| `int` | Arbitrary-width integer | Arithmetic |
| `BitVec` | Circuit-level bitvector | XOR/AND/OR gates |

## Workflows

### Basic Workflow

1. Create solver
2. Create variables
3. Add clauses
4. Solve
5. Interpret result

### Incremental Solving

Solve with temporary assumptions without modifying the base problem:

```python
result1 = solver.solve_with_assumptions([x.id])   # Assume x=true
result2 = solver.solve_with_assumptions([-x.id])  # Assume x=false
```

### Parallel Exploration

Clone solvers for parallel BFS:

```python
solver_copy = solver.clone()
# Explore different branches in parallel
```

### Circuit Building

Use `CircuitBuilder` for gate-level constraints:

```python
from satellite_lab import BatchingSolver, CircuitBuilder, BitVec

solver = BatchingSolver()
cb = CircuitBuilder(solver)

a = cb.new_bitvec(8)
b = cb.new_bitvec(8)
sum_result = cb.add_xor(a, b)

solver.solve()
```

## Next Steps

- [Python Tutorial](python-tutorial.md) - Deep dive into Python API
- [CLI Reference](cli-reference.md) - Command-line options
- [Architecture Overview](../architecture/overview.md) - System design
- [API Reference](../api/python-api.md) - Complete API docs
