# Satellite

**High-Performance Parallel SAT Solver with Advanced Type System**

Satellite is a next-generation constraint satisfaction solver that extends traditional CDCL-based SAT solving with a rich type system, heterogeneous computing (CPU/GPU), and programmable constraints.

## Features

- 🧮 **Hierarchical Type System**: From boolean primitives to batches, integers, vectors, and arbitrary-precision floats
- 🌿 **Branch-Based Parallel Model**: Git-style constraint branching with reference-counted garbage collection
- 🚀 **Heterogeneous Computing**: Unified CPU/GPU task execution through lock-free MPMC queues
- 🔧 **Programmable Constraints (ABI-OP)**: User-defined constraints compiled via LLVM JIT with automatic optimization
- 🔬 **Research-Friendly Ecosystem**: Jupyter integration, Python bindings, modern IDE with LSP support

## Components

| Crate | Description |
|-------|-------------|
| `satellite-kit` | Core solver library and API integration |
| `satellite-cdcl` | High-performance CDCL solver core with 1-UIP learning |
| `satellite-jit` | LLVM-based JIT compiler for ABI-OP constraints |
| `satellite-daemon` | Distributed solving scheduler and server |
| `satellite-format` | Advanced-CNF and DIMACS format parsers |
| `satellite-base` | Common types and utilities |
| `satellite-cli` | Full-featured command-line interface |
| `satellite-ide` | Modern IDE with LSP, syntax highlighting, debugger |

### Python Packages (PyPI)

| Package | Description |
|---------|-------------|
| `satellite-lab` | Python wrapper for researchers |
| `satellite-jupyter` | Jupyter kernel integration |

## Building

### Prerequisites

- Rust 1.85+
- CUDA Toolkit 12.x (for GPU support)
- vcpkg (for C++ dependencies)
- Node.js (for IDE frontend)

### Build Commands

```bash
# Build all Rust crates
cargo build --workspace

# Build GPU worker
cd cpp_src && cmake -B build && cmake --build build

# Build Python package
cd py_src/satellite_lab && maturin develop

# Build IDE
cd crates/satellite-ide && cargo tauri build
```

## License

MIT
