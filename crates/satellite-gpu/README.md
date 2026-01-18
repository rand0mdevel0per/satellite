# satellite-gpu

GPU acceleration bindings for Satellite SAT solver.

## Features

- **CUDA/HIP support** for NVIDIA and AMD GPUs
- **GPU BCP** - Boolean Constraint Propagation on GPU
- **ABI-OP execution** - User-defined constraints on GPU
- **Memory info** - GPU memory usage tracking

## Usage

```rust
use satellite_gpu::{GpuWorker, GpuStatus};

let worker = GpuWorker::new()?;
if worker.is_available() {
    worker.submit_bcp(&clauses, num_clauses, &assignments)?;
    worker.sync();
    if let Some(result) = worker.poll_result()? {
        println!("Conflict: {}", result.has_conflict);
    }
}
```

## Feature Flags

- `cuda` - Enable NVIDIA CUDA support
- `hip` - Enable AMD HIP support

## License

MIT
