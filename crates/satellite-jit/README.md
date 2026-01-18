# satellite-jit

JIT compilation for Satellite ABI-OPs using LLVM.

## Features

- **Runtime Compilation**: Compile Rust/C++ constraints to machine code at runtime
- **LLVM Integration**: Uses `inkwell` bindings to LLVM
- **Optimizations**: Applies O3 optimizations to generated code

## Prerequisites

- LLVM 15/16/17 installed and in PATH
- C++ compiler (MSVC/GCC/Clang)

## Usage

```rust
use satellite_jit::JitCompiler;

let compiler = JitCompiler::new();
let module = compiler.compile_source(source_code)?;
let func = module.get_function("constraint_check")?;
```

## License

MIT
