# Satellite Lab

Python wrapper for the Satellite SAT solver.

## Installation

```bash
pip install satellite-lab
```

Or build from source:

```bash
cd py_src/satellite_lab
maturin develop
```

## Usage

```python
from satellite_lab import Solver

solver = Solver()

# Create variables
x = solver.bool_var(name="x")
y = solver.bool_var(name="y")
z = solver.bool_var(name="z")

# Add clauses (x OR y), (NOT x OR z)
solver.add_clause([+x, +y])
solver.add_clause([-x, +z])

# Solve
result = solver.solve()

if result.satisfiable:
    print("SAT!")
    print("Model:", result.model)
else:
    print("UNSAT")
```

## Batch API

For better performance, use the batch API:

```python
clauses = [
    [1, 2, 3],
    [-1, 2],
    [1, -2, -3],
]
solver.add_clauses(clauses)
```

## Types

- `BoolVar`: Single boolean variable
- `Batch`: Vector of booleans (like Verilog vectors)
- `IntVar`: Integer stored as big-endian bits
