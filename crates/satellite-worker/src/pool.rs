//! CPU worker thread pool.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use satellite_lockfree::{MpmcQueue, Priority};
use satellite_branch::BranchId;
use crate::job::{Job, JobResult};
use crate::gpu_bridge::GpuBridge;

/// Configuration for the worker pool.
#[derive(Debug, Clone)]
pub struct WorkerPoolConfig {
    /// Number of CPU worker threads.
    pub cpu_workers: usize,
    /// Whether to enable GPU workers.
    pub enable_gpu: bool,
    /// Path to GPU worker library.
    pub gpu_lib_path: Option<String>,
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        Self {
            cpu_workers: num_cpus(),
            enable_gpu: false,
            gpu_lib_path: None,
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

/// Worker thread pool.
pub struct WorkerPool {
    /// Job queue.
    queue: Arc<MpmcQueue<Job>>,
    /// Worker threads.
    workers: Vec<JoinHandle<()>>,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// GPU bridge (if enabled).
    gpu_bridge: Option<GpuBridge>,
    /// Result sender.
    result_tx: std::sync::mpsc::Sender<JobResult>,
    /// Result receiver.
    result_rx: std::sync::mpsc::Receiver<JobResult>,
}

impl WorkerPool {
    /// Creates a new worker pool.
    pub fn new(config: WorkerPoolConfig) -> Self {
        let queue = Arc::new(MpmcQueue::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let (result_tx, result_rx) = std::sync::mpsc::channel();

        let mut workers = Vec::with_capacity(config.cpu_workers);

        for id in 0..config.cpu_workers {
            let q = queue.clone();
            let s = shutdown.clone();
            let tx = result_tx.clone();

            let handle = thread::Builder::new()
                .name(format!("satellite-worker-{}", id))
                .spawn(move || {
                    worker_loop(id, q, s, tx);
                })
                .expect("Failed to spawn worker thread");

            workers.push(handle);
        }

        let gpu_bridge = if config.enable_gpu {
            config
                .gpu_lib_path
                .as_ref()
                .and_then(|path| GpuBridge::new(path).ok())
        } else {
            None
        };

        Self {
            queue,
            workers,
            shutdown,
            gpu_bridge,
            result_tx,
            result_rx,
        }
    }

    /// Submits a job with the given priority.
    pub fn submit(&self, priority: Priority, job: Job) {
        self.queue.push(priority, job);
    }

    /// Receives a completed job result (blocking).
    pub fn recv_result(&self) -> Option<JobResult> {
        self.result_rx.recv().ok()
    }

    /// Tries to receive a completed job result (non-blocking).
    pub fn try_recv_result(&self) -> Option<JobResult> {
        self.result_rx.try_recv().ok()
    }

    /// Returns the number of pending jobs.
    pub fn pending_jobs(&self) -> usize {
        self.queue.len()
    }

    /// Returns whether GPU is available.
    pub fn has_gpu(&self) -> bool {
        self.gpu_bridge.is_some()
    }

    /// Shuts down the worker pool.
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::SeqCst);

        // Push dummy jobs to wake up workers
        for _ in 0..self.workers.len() {
            self.queue.push(Priority::Low, Job::shutdown());
        }

        for worker in self.workers {
            let _ = worker.join();
        }
    }
}

fn worker_loop(
    id: usize,
    queue: Arc<MpmcQueue<Job>>,
    shutdown: Arc<AtomicBool>,
    result_tx: std::sync::mpsc::Sender<JobResult>,
) {
    use satellite_base::utils::XorShift64;

    let mut rng = XorShift64::new((id as u64 + 1) * 12345);

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let rand = rng.next_f64();
        if let Some(job) = queue.pop(rand) {
            if job.is_shutdown() {
                break;
            }

            // Check if branch is still active
            if job.is_branch_failed() {
                continue;
            }

            // Execute job
            let result = job.execute();

            // Send result
            let _ = result_tx.send(result);
        } else {
            // No work, yield
            thread::yield_now();
        }
    }
}
