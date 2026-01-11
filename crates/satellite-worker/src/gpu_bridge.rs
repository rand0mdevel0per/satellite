//! GPU worker bridge via FFI.

use libloading::{Library, Symbol};
use satellite_base::{Error, Result};
use std::path::Path;

/// C API function signatures for GPU worker.
type GpuInitFn = unsafe extern "C" fn() -> i32;
type GpuShutdownFn = unsafe extern "C" fn();
type GpuLaunchKernelFn = unsafe extern "C" fn(
    clause_data: *const i64,
    clause_offsets: *const usize,
    num_clauses: usize,
    assignments: *const i8,
    results: *mut i32,
);
type GpuSubmitJobFn =
    unsafe extern "C" fn(priority: u32, job_id: u64, branch_id: u64, start: u32, end: u32);
type GpuReadResultsFn =
    unsafe extern "C" fn(host_buffer: *mut i32, start_idx: usize, count: usize) -> i32;
type GpuStopKernelFn = unsafe extern "C" fn();

/// Bridge to GPU worker library.
pub struct GpuBridge {
    library: Library,
    initialized: bool,
}

impl GpuBridge {
    /// Loads and initializes the GPU worker library.
    pub fn new<P: AsRef<Path>>(lib_path: P) -> Result<Self> {
        let library = unsafe {
            Library::new(lib_path.as_ref())
                .map_err(|e| Error::GpuError(format!("Failed to load GPU library: {}", e)))?
        };

        let mut bridge = Self {
            library,
            initialized: false,
        };

        bridge.init()?;

        Ok(bridge)
    }

    fn init(&mut self) -> Result<()> {
        unsafe {
            let init: Symbol<GpuInitFn> = self
                .library
                .get(b"init_gpu_device")
                .map_err(|e| Error::GpuError(format!("Missing init_gpu_device: {}", e)))?;

            let result = init();
            if result != 0 {
                return Err(Error::GpuError(format!(
                    "GPU initialization failed with code {}",
                    result
                )));
            }

            self.initialized = true;
            Ok(())
        }
    }

    /// Launches the persistent BCP kernel.
    /// Note: Buffers must remain valid as long as kernel is running!
    pub fn start_kernel(
        &self,
        clause_data: &[i64],
        clause_offsets: &[usize],
        assignments: &[i8],
        results: &mut [i32],
    ) -> Result<()> {
        unsafe {
            let launch: Symbol<GpuLaunchKernelFn> =
                self.library.get(b"launch_persistent_kernel").map_err(|e| {
                    Error::GpuError(format!("Missing launch_persistent_kernel: {}", e))
                })?;

            launch(
                clause_data.as_ptr(),
                clause_offsets.as_ptr(),
                clause_offsets.len() - 1,
                assignments.as_ptr(),
                results.as_mut_ptr(),
            );
            Ok(())
        }
    }

    /// Submits a job to the persistent kernel.
    pub fn enqueue_job(
        &self,
        priority: u32,
        job_id: u64,
        branch_id: u64,
        start: u32,
        end: u32,
    ) -> Result<()> {
        unsafe {
            let submit: Symbol<GpuSubmitJobFn> = self
                .library
                .get(b"submit_job")
                .map_err(|e| Error::GpuError(format!("Missing submit_job: {}", e)))?;

            submit(priority, job_id, branch_id, start, end);
            Ok(())
        }
    }

    /// Reads results from the GPU.
    pub fn read_results(&self, start_idx: usize, count: usize, buffer: &mut [i32]) -> Result<()> {
        unsafe {
            let read: Symbol<GpuReadResultsFn> = self
                .library
                .get(b"read_results")
                .map_err(|e| Error::GpuError(format!("Missing read_results: {}", e)))?;

            let code = read(buffer.as_mut_ptr(), start_idx, count);
            if code != 0 {
                return Err(Error::GpuError(format!(
                    "GPU read failed with code {}",
                    code
                )));
            }
            Ok(())
        }
    }

    /// Stops the persistent kernel.
    pub fn stop_kernel(&self) -> Result<()> {
        unsafe {
            let stop: Symbol<GpuStopKernelFn> = self
                .library
                .get(b"stop_persistent_kernel")
                .map_err(|e| Error::GpuError(format!("Missing stop_persistent_kernel: {}", e)))?;

            stop();
            Ok(())
        }
    }

    /// Returns whether GPU is available and initialized.
    pub fn is_available(&self) -> bool {
        self.initialized
    }
}

impl Drop for GpuBridge {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                if let Ok(shutdown) = self.library.get::<GpuShutdownFn>(b"shutdown_gpu_device") {
                    shutdown();
                }
            }
        }
    }
}
