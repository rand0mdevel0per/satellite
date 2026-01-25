# Circuit Gadgets Reference

Circuit gadgets are pre-built constraint patterns for common operations like arithmetic, comparisons, and bitwise logic.

## Overview

Gadgets provide:
- **High-level operations**: XOR, AND, OR, ADD, MUX, etc.
- **Optimized encodings**: Minimal clause count
- **Type safety**: Compile-time width checking
- **Composability**: Chain gadgets to build complex circuits

## CircuitBuilder

The `CircuitBuilder` class manages variable allocation and clause generation.

### Python

```python
from satellite_lab import BatchingSolver, CircuitBuilder

solver = BatchingSolver()
cb = CircuitBuilder(solver)

# Allocate bitvectors
a = cb.new_bitvec(8)
b = cb.new_bitvec(8)

# Build circuit
result = cb.add_xor(a, b)

# Get generated clauses
clauses = cb.get_clauses()
for clause in clauses:
    solver.add_clause(clause)
```

### Rust

```rust
use satellite_kit::{CircuitBuilder, BitVec};

let mut cb = CircuitBuilder::new(0);

// Allocate bitvectors
let a = cb.new_bitvec(8);
let b = cb.new_bitvec(8);

// Build circuit
let result = cb.add_xor(&a, &b);

// Get clauses
let clauses = cb.into_clauses();
```

## Bitwise Operations

### XOR

Exclusive OR operation.

```python
# result = a XOR b
result = cb.add_xor(a, b)
```

**Encoding**: 4 clauses per bit
- `(¬a ∨ ¬b ∨ ¬r)`
- `(a ∨ b ∨ ¬r)`
- `(a ∨ ¬b ∨ r)`
- `(¬a ∨ b ∨ r)`

**Use cases**: Cryptography, checksums, parity

### AND

Bitwise AND operation.

```python
# result = a AND b
result = cb.add_and(a, b)
```

**Encoding**: 3 clauses per bit
- `(¬a ∨ ¬b ∨ r)`
- `(a ∨ ¬r)`
- `(b ∨ ¬r)`

**Use cases**: Bit masking, flag checking

### OR

Bitwise OR operation.

```python
# result = a OR b
result = cb.add_or(a, b)
```

**Encoding**: 3 clauses per bit
- `(a ∨ b ∨ ¬r)`
- `(¬a ∨ r)`
- `(¬b ∨ r)`

**Use cases**: Flag setting, bit merging

### NOT

Bitwise NOT (inversion).

```python
# result = NOT a
result = cb.add_not(a)
```

**Encoding**: 2 clauses per bit
- `(a ∨ r)`
- `(¬a ∨ ¬r)`

**Use cases**: Bit inversion, complement

### Shift Left (SHL)

Logical left shift.

```python
# result = a << 2
result = cb.add_shl(a, 2)
```

**Encoding**: Direct wire connections
- Lower bits: Connect to zero
- Upper bits: Connect to shifted input bits

**Use cases**: Multiplication by powers of 2, bit positioning

### Shift Right (SHR)

Logical right shift.

```python
# result = a >> 2
result = cb.add_shr(a, 2)
```

**Encoding**: Direct wire connections
- Upper bits: Connect to zero
- Lower bits: Connect to shifted input bits

**Use cases**: Division by powers of 2, bit extraction

## Arithmetic Operations

### Addition

Full adder with carry.

```python
# result, carry_out = a + b + carry_in
result, carry_out = cb.add_add(a, b, carry_in=None)
```

**Encoding**: Ripple-carry adder, ~10 clauses per bit
- Full adder per bit position
- Carry chain from LSB to MSB

**Use cases**: Integer arithmetic, counters

### Subtraction

Subtraction using two's complement.

```python
# result = a - b
result = cb.add_sub(a, b)
```

**Encoding**: Adder with inverted second operand
- Invert b: `b_inv = NOT b`
- Add with carry: `result = a + b_inv + 1`

**Use cases**: Difference calculation, comparisons

### Equality

Check if two bitvectors are equal.

```python
# eq = (a == b)
eq = cb.add_eq(a, b)
```

**Encoding**: XOR + NOR reduction
- XOR each bit pair
- NOR all XOR results

**Use cases**: Conditional logic, assertions

## Control Flow

### Multiplexer (MUX)

Select between two values based on condition.

```python
# result = cond ? then_val : else_val
result = cb.add_ite(cond, then_val, else_val)
```

**Encoding**: 2 clauses per bit
- `(¬cond ∨ ¬then ∨ result)`
- `(cond ∨ ¬else ∨ result)`

**Use cases**: Conditional assignment, state machines

## Complete Example

### 8-bit ALU

```python
from satellite_lab import BatchingSolver, CircuitBuilder

solver = BatchingSolver()
cb = CircuitBuilder(solver)

# Inputs
a = cb.new_bitvec(8)
b = cb.new_bitvec(8)
op = cb.new_bitvec(2)  # Operation selector

# Operations
add_result = cb.add_add(a, b)[0]
sub_result = cb.add_sub(a, b)
and_result = cb.add_and(a, b)
xor_result = cb.add_xor(a, b)

# Select operation based on op
# op=00: ADD, op=01: SUB, op=10: AND, op=11: XOR
temp1 = cb.add_ite(op.bit(0), sub_result, add_result)
temp2 = cb.add_ite(op.bit(0), xor_result, and_result)
result = cb.add_ite(op.bit(1), temp2, temp1)

# Add all clauses to solver
for clause in cb.get_clauses():
    solver.add_clause(clause)

# Constrain inputs and solve
solver.add_clause([a.bit(0)])  # a = 1
solver.add_clause([b.bit(0)])  # b = 1
solver.add_clause([op.bit(0)]) # op = 01 (SUB)

result_sat = solver.solve()
print(f"Result: {result_sat.satisfiable}")
```

## Best Practices

### 1. Minimize Bit Width

Use the smallest bit width that satisfies constraints:
```python
# Bad: Wasteful
counter = cb.new_bitvec(64)  # Only need 0-100

# Good: Efficient
counter = cb.new_bitvec(7)   # 2^7 = 128 > 100
```

### 2. Reuse Intermediate Results

Cache common subexpressions:
```python
# Bad: Duplicate computation
result1 = cb.add_add(a, b)[0]
result2 = cb.add_add(a, b)[0]  # Redundant

# Good: Reuse
sum_ab = cb.add_add(a, b)[0]
result1 = sum_ab
result2 = sum_ab
```

### 3. Choose Efficient Encodings

- **XOR**: 4 clauses/bit (expensive)
- **AND/OR**: 3 clauses/bit (moderate)
- **NOT**: 2 clauses/bit (cheap)
- **Shifts**: 0 clauses (free - just wiring)

### 4. Avoid Deep Carry Chains

Ripple-carry adders are slow for wide operands:
```python
# Bad: 64-bit ripple-carry
result = cb.add_add(a64, b64)[0]  # Long carry chain

# Better: Break into smaller chunks
low = cb.add_add(a_low, b_low)
high = cb.add_add(a_high, b_high, carry_in=low[1])
```

## Performance Comparison

| Operation | Clauses/bit | Depth | Use When |
|-----------|-------------|-------|----------|
| XOR | 4 | 1 | Crypto, parity |
| AND/OR | 3 | 1 | Masking, flags |
| NOT | 2 | 1 | Inversion |
| Shift | 0 | 0 | Multiply/divide by 2^n |
| Add | ~10 | O(n) | Arithmetic |
| Equality | ~6 | O(log n) | Comparisons |
| MUX | 2 | 1 | Conditional logic |