//! GPU BCP Benchmarks
//! 
//! Run with: cargo bench --features cuda -p satellite-gpu

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

#[cfg(feature = "cuda")]
use satellite_gpu::GpuWorker;

/// Generate test clauses in flattened format (0-terminated)
fn generate_clauses(num_clauses: usize, clause_size: usize) -> Vec<i64> {
    let mut clauses = Vec::with_capacity(num_clauses * (clause_size + 1));
    let mut lit = 1;
    
    for _ in 0..num_clauses {
        for _ in 0..clause_size {
            // Alternate positive/negative literals
            clauses.push(if lit % 2 == 0 { -(lit as i64) } else { lit as i64 });
            lit += 1;
        }
        clauses.push(0); // Clause terminator
    }
    
    clauses
}

/// Generate random assignments
fn generate_assignments(num_vars: usize) -> Vec<i8> {
    (0..num_vars).map(|i| (i % 3) as i8 - 1).collect() // -1, 0, 1 pattern
}

#[cfg(feature = "cuda")]
fn bench_bcp_throughput(c: &mut Criterion) {
    let worker = match GpuWorker::new() {
        Ok(w) => w,
        Err(_) => {
            eprintln!("GPU not available, skipping benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_bcp");
    
    // Test with different clause counts
    for num_clauses in [1000, 10000, 100000, 1000000].iter() {
        let clause_size = 3; // 3-SAT
        let num_vars = num_clauses * clause_size;
        
        let clauses = generate_clauses(*num_clauses, clause_size);
        let assignments = generate_assignments(num_vars);
        
        group.throughput(Throughput::Elements(*num_clauses as u64));
        group.bench_with_input(
            BenchmarkId::new("clauses", num_clauses),
            &(&clauses, &assignments, *num_clauses),
            |b, (clauses, assignments, nc)| {
                b.iter(|| {
                    worker.submit_bcp(
                        black_box(clauses),
                        black_box(*nc),
                        black_box(assignments),
                    ).unwrap();
                    worker.sync();
                    worker.poll_result().unwrap()
                })
            },
        );
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_bcp_latency(c: &mut Criterion) {
    let worker = match GpuWorker::new() {
        Ok(w) => w,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("gpu_bcp_latency");
    
    // Small problem for latency measurement
    let clauses = generate_clauses(32, 3);
    let assignments = generate_assignments(96);
    
    group.bench_function("32_clauses", |b| {
        b.iter(|| {
            worker.bcp_sync(
                black_box(&clauses),
                black_box(32),
                black_box(&assignments),
            ).unwrap()
        })
    });
    
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_memory_bandwidth(c: &mut Criterion) {
    let worker = match GpuWorker::new() {
        Ok(w) => w,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("gpu_memory");
    
    // Large data transfer test
    for size_mb in [1, 10, 100].iter() {
        let num_elements = size_mb * 1024 * 1024 / 8; // i64 = 8 bytes
        let clauses: Vec<i64> = (0..num_elements as i64).collect();
        let num_clauses = num_elements / 4; // Approximate
        let assignments = generate_assignments(num_elements);
        
        group.throughput(Throughput::Bytes(*size_mb as u64 * 1024 * 1024));
        group.bench_with_input(
            BenchmarkId::new("transfer_mb", size_mb),
            &(&clauses, &assignments, num_clauses),
            |b, (clauses, assignments, nc)| {
                b.iter(|| {
                    worker.submit_bcp(
                        black_box(clauses),
                        black_box(*nc),
                        black_box(assignments),
                    ).unwrap();
                    worker.sync();
                })
            },
        );
    }
    
    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    bench_bcp_throughput,
    bench_bcp_latency,
    bench_memory_bandwidth
);

#[cfg(feature = "cuda")]
criterion_main!(benches);

// Stub when no GPU feature
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("GPU benchmarks require --features cuda");
}
