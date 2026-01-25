# Python Tutorial

Complete guide to using Satellite from Python via the `satellite-lab` package.

## Installation

```bash
pip install satellite-lab
```

Or install from source:

```bash
cd py_src/satellite_lab
pip install -e .
```

## Quick Start

### Your First SAT Problem

```python
from satellite_lab import BatchingSolver

# Create solver
solver = BatchingSolver()

# Create variables
x = solver.bool_var(name="x")
y = solver.bool_var(name="y")
z = solver.bool_var(name="z")

# Add clauses: (x OR y) AND (NOT x OR z) AND (NOT y OR NOT z)
solver.add_clause([x.id, y.id])
solver.add_clause([-x.id, z.id])
solver.add_clause([-y.id, -z.id])

# Solve
result = solver.solve()

if result.satisfiable:
    print("SAT - Found solution!")
    print(f"x = {result.model[x.id]}")
    print(f"y = {result.model[y.id]}")
    print(f"z = {result.model[z.id]}")
else:
    print("UNSAT - No solution exists")
```

## Variables and Types

### Boolean Variables

Single boolean variables:

```python
# Create with optional name
x = solver.bool_var(name="x")
y = solver.bool_var()  # Anonymous

# Get variable ID
print(f"Variable ID: {x.id}")

# Create literals
pos_lit = x.id   # x = true
neg_lit = -x.id  # x = false
```

### Batch Variables

Fixed-width bitvectors:

```python
# Create 32-bit register
reg = solver.batch_var(32, name="register")

# Access individual bits
bit_0 = reg.get(0)
bit_31 = reg.get(31)

# Slice ranges
low_byte = reg.slice(0, 8)   # bits [0..8)
high_byte = reg.slice(24, 32) # bits [24..32)

# Get literal for specific bit
lit = reg.lit(5)  # Literal for bit 5
```

### Integer Variables

Arbitrary-width integers:

```python
# Create 16-bit integer
counter = solver.int_var(16, name="counter")

# Use in constraints (via CircuitBuilder)
```

## Adding Clauses

### Basic Clauses

```python
# Clause: x OR y OR NOT z
solver.add_clause([x.id, y.id, -z.id])

# Unit clause: x must be true
solver.add_clause([x.id])

# Binary clause: x OR NOT y
solver.add_clause([x.id, -y.id])
```

### Bulk Clause Addition

```python
clauses = [
    [x.id, y.id],
    [-x.id, z.id],
    [-y.id, -z.id]
]

for clause in clauses:
    solver.add_clause(clause)
```

## Solving

### Basic Solving

```python
result = solver.solve()

if result.satisfiable:
    # Access model
    for var_id in range(solver.num_vars()):
        value = result.model[var_id]
        print(f"var_{var_id} = {value}")
else:
    print("UNSAT")
```

### Solving with Assumptions

Test hypotheses without modifying the problem:

```python
# Solve assuming x = true
result1 = solver.solve_with_assumptions([x.id])

# Solve assuming x = false
result2 = solver.solve_with_assumptions([-x.id])

# Multiple assumptions
result3 = solver.solve_with_assumptions([x.id, -y.id, z.id])
```

### Timeouts

Set solving timeout:

```python
# Set 5 second timeout
solver.set_timeout(5000)  # milliseconds

result = solver.solve()
if result.satisfiable:
    print("SAT")
elif result.satisfiable is False:
    print("UNSAT")
else:
    print("Timeout or unknown")
```

## Circuit Building

### Using CircuitBuilder

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
sum_result, carry = cb.add_add(a, b)

# Add clauses to solver
for clause in cb.get_clauses():
    solver.add_clause(clause)
```

## Advanced Features

### Cloning Solvers

Clone solvers for parallel exploration:

```python
# Create base solver
base_solver = BatchingSolver()
x = base_solver.bool_var()
y = base_solver.bool_var()
base_solver.add_clause([x.id, y.id])

# Clone for different branches
solver1 = base_solver.clone()
solver2 = base_solver.clone()

# Explore different branches
solver1.add_clause([x.id])   # Branch 1: x = true
solver2.add_clause([-x.id])  # Branch 2: x = false

result1 = solver1.solve()
result2 = solver2.solve()
```

### Parallel Solving

Use worker pool for async solving:

```python
from satellite_lab import init_worker_pool, submit_solve, fetch_finished_jobs

# Initialize worker pool
init_worker_pool(4)

# Submit multiple jobs
job_ids = []
for problem in problems:
    solver = BatchingSolver()
    # ... add clauses ...
    job_id = submit_solve(solver.ctx_id)
    job_ids.append(job_id)

# Collect results
while job_ids:
    finished = fetch_finished_jobs(max_count=10)
    for job_id, result in finished:
        print(f"Job {job_id}: {result.satisfiable}")
        job_ids.remove(job_id)
```

### File I/O

Load and save problems:

```python
# Load from DIMACS CNF
solver = BatchingSolver()
solver.load_from_file("problem.cnf")

# Load from JSON
solver.load_from_file("problem.json")

# Save to file
solver.save_to_file("output.json")
```

### UNSAT Core

Extract minimal conflicting constraints:

```python
solver = BatchingSolver()
solver.enable_unsat_core()

# Add constraints
solver.add_clause([x.id, y.id])
solver.add_clause([-x.id, z.id])
solver.add_clause([-y.id, -z.id])
solver.add_clause([x.id, -z.id])

result = solver.solve()
if not result.satisfiable:
    core = solver.get_unsat_core()
    print(f"UNSAT core clause IDs: {core}")
```

## Complete Examples

### Example 1: N-Queens Problem

```python
from satellite_lab import BatchingSolver

def solve_n_queens(n):
    solver = BatchingSolver()

    # Create variables: queen[i][j] = queen at row i, col j
    queens = [[solver.bool_var(name=f"q_{i}_{j}")
               for j in range(n)] for i in range(n)]

    # One queen per row
    for i in range(n):
        solver.add_clause([queens[i][j].id for j in range(n)])

    # At most one queen per row
    for i in range(n):
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                solver.add_clause([-queens[i][j1].id, -queens[i][j2].id])

    # At most one queen per column
    for j in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                solver.add_clause([-queens[i1][j].id, -queens[i2][j].id])

    # At most one queen per diagonal
    for i1 in range(n):
        for j1 in range(n):
            for i2 in range(i1 + 1, n):
                j2 = j1 + (i2 - i1)
                if j2 < n:
                    solver.add_clause([-queens[i1][j1].id, -queens[i2][j2].id])
                j2 = j1 - (i2 - i1)
                if j2 >= 0:
                    solver.add_clause([-queens[i1][j1].id, -queens[i2][j2].id])

    result = solver.solve()
    if result.satisfiable:
        print(f"{n}-Queens solution found:")
        for i in range(n):
            row = ""
            for j in range(n):
                if result.model[queens[i][j].id]:
                    row += "Q "
                else:
                    row += ". "
            print(row)
    else:
        print(f"No {n}-Queens solution")

solve_n_queens(8)
```

### Example 2: Sudoku Solver

```python
from satellite_lab import BatchingSolver

def solve_sudoku(grid):
    """Solve 9x9 Sudoku puzzle. 0 represents empty cell."""
    solver = BatchingSolver()

    # Variables: cell[i][j][k] = digit k+1 at position (i,j)
    cells = [[[solver.bool_var(name=f"c_{i}_{j}_{k}")
               for k in range(9)] for j in range(9)] for i in range(9)]

    # Each cell has exactly one digit
    for i in range(9):
        for j in range(9):
            # At least one digit
            solver.add_clause([cells[i][j][k].id for k in range(9)])
            # At most one digit
            for k1 in range(9):
                for k2 in range(k1 + 1, 9):
                    solver.add_clause([-cells[i][j][k1].id, -cells[i][j][k2].id])

    # Each row has each digit exactly once
    for i in range(9):
        for k in range(9):
            solver.add_clause([cells[i][j][k].id for j in range(9)])

    # Each column has each digit exactly once
    for j in range(9):
        for k in range(9):
            solver.add_clause([cells[i][j][k].id for i in range(9)])

    # Each 3x3 box has each digit exactly once
    for box_i in range(3):
        for box_j in range(3):
            for k in range(9):
                clause = []
                for i in range(3):
                    for j in range(3):
                        clause.append(cells[box_i*3+i][box_j*3+j][k].id)
                solver.add_clause(clause)

    # Add given digits
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                k = grid[i][j] - 1
                solver.add_clause([cells[i][j][k].id])

    result = solver.solve()
    if result.satisfiable:
        solution = [[0]*9 for _ in range(9)]
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    if result.model[cells[i][j][k].id]:
                        solution[i][j] = k + 1
        return solution
    return None

# Example puzzle
puzzle = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9]
]

solution = solve_sudoku(puzzle)
if solution:
    for row in solution:
        print(row)
```

## Best Practices

1. **Name variables**: Use descriptive names for debugging
2. **Minimize bit width**: Use smallest width that satisfies constraints
3. **Reuse solvers**: Clone instead of recreating for similar problems
4. **Use assumptions**: Test hypotheses without modifying base problem
5. **Enable UNSAT core selectively**: Only when debugging

## Next Steps

- [Assumptions and Incremental Solving](../advanced/assumptions.md)
- [Parallel Solving](../advanced/parallel.md)
- [Circuit Gadgets](../advanced/gadgets.md)
- [Python API Reference](../api/python-api.md)
