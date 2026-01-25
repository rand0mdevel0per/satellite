# Type System

Satellite extends traditional SAT solving with a rich hierarchical type system that enables constraint solving beyond boolean satisfiability.

## Type Hierarchy

```
Type
├── bool          # Single boolean variable
├── batch[N]      # Fixed-width bitvector (N bits)
├── int           # Arbitrary-width integer
├── float         # Arbitrary-precision floating-point
├── vec           # Vector of batches
└── BitVec        # Circuit-level bitvector with operations
```

## Boolean Variables

The most basic type in Satellite.

### Definition

```rust
pub struct BoolVar {
    id: VarId,  // Internal variable identifier
}
```

### Usage

**Rust:**
```rust
let x = solver.bool_var();
let y = solver.bool_var();

// Create literals
let pos_lit = x.positive_lit();  // x = true
let neg_lit = x.negative_lit();  // x = false

// Add clause: x OR NOT y
solver.add_clause(vec![pos_lit, neg_lit]);
```

**Python:**
```python
x = solver.bool_var(name="x")
y = solver.bool_var(name="y")

# Add clause: x OR NOT y
solver.add_clause([x.id, -y.id])
```

### Literal Encoding

- **Positive literal**: `var_id + 1` (variable is true)
- **Negative literal**: `-(var_id + 1)` (variable is false)
- **Variable ID**: 0-indexed internally

## Batch Variables

Fixed-width bitvectors representing registers, memory words, or packed boolean arrays.

### Definition

```rust
pub struct Batch {
    base_id: VarId,  // First variable ID
    dim: usize,      // Number of bits
}
```
