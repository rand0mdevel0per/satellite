//! Benchmarks for MPMC queue.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use satellite_lockfree::{MpmcQueue, Priority};
use std::sync::Arc;
use std::thread;

fn bench_push(c: &mut Criterion) {
    c.bench_function("mpmc_push", |b| {
        let queue = MpmcQueue::new();
        let mut i = 0u64;
        b.iter(|| {
            queue.push(Priority::High, black_box(i));
            i += 1;
        });
    });
}

fn bench_concurrent_push_pop(c: &mut Criterion) {
    c.bench_function("mpmc_concurrent", |b| {
        b.iter(|| {
            let queue = Arc::new(MpmcQueue::new());
            let mut handles = vec![];

            // Producers
            for _ in 0..4 {
                let q = queue.clone();
                handles.push(thread::spawn(move || {
                    for i in 0..1000 {
                        q.push(Priority::High, i);
                    }
                }));
            }

            // Consumers
            for _ in 0..4 {
                let q = queue.clone();
                handles.push(thread::spawn(move || {
                    for _ in 0..1000 {
                        let _ = q.pop(0.5);
                    }
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
        });
    });
}

criterion_group!(benches, bench_push, bench_concurrent_push_pop);
criterion_main!(benches);
