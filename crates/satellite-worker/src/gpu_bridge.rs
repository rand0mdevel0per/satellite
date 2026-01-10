//! GPU worker bridge via FFI.

use libloading::{Library, Symbol};
use satellite_base::{Error, Result};
use std::path::Path;

/// C API function signatures for GPU worker.
type GpuInitFn = unsafe extern "C" fn() -> i32;
type GpuShutdownFn = unsafe extern "C" fn();
type GpuSubmitBcpFn = unsafe extern "C" fn(
    clause_data: *const i64,
    num_clauses: usize,
    assignments: *const i8,
    num_vars: usize,
) -> i32;
type GpuPollResultFn = unsafe extern "C" fn(result: *mut GpuResult) -> i32;

/// Result from GPU operation.
#[repr(C)]
pub struct GpuResult {
    /// Status code (0 = ok, 1 = conflict, -1 = error).
    pub status: i32,
    /// Conflict clause index (if status == 1).
    pub conflict_clause: usize,
}

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
                .get(b"gpu_worker_init")
                .map_err(|e| Error::GpuError(format!("Missing gpu_worker_init: {}", e)))?;

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

    /// Submits a BCP job to the GPU.
    pub fn submit_bcp(
        &self,
        clause_data: &[i64],
        num_clauses: usize,
        assignments: &[i8],
    ) -> Result<()> {
        if !self.initialized {
            return Err(Error::GpuError("GPU not initialized".to_string()));
        }

        unsafe {
            let submit: Symbol<GpuSubmitBcpFn> = self
                .library
                .get(b"gpu_worker_submit_bcp")
                .map_err(|e| Error::GpuError(format!("Missing gpu_worker_submit_bcp: {}", e)))?;

            let result = submit(
                clause_data.as_ptr(),
                num_clauses,
                assignments.as_ptr(),
                assignments.len(),
            );

            if result != 0 {
                return Err(Error::GpuError(format!(
                    "GPU BCP submit failed with code {}",
                    result
                )));
            }

            Ok(())
        }
    }

    /// Polls for a completed GPU result.
    pub fn poll_result(&self) -> Result<Option<GpuResult>> {
        if !self.initialized {
            return Err(Error::GpuError("GPU not initialized".to_string()));
        }

        unsafe {
            let poll: Symbol<GpuPollResultFn> = self
                .library
                .get(b"gpu_worker_poll_result")
                .map_err(|e| Error::GpuError(format!("Missing gpu_worker_poll_result: {}", e)))?;

            let mut result = GpuResult {
                status: -2,
                conflict_clause: 0,
            };

            let code = poll(&mut result);
            if code == 0 {
                Ok(Some(result))
            } else if code == 1 {
                Ok(None) // No result ready
            } else {
                Err(Error::GpuError(format!("GPU poll failed with code {}", code)))
            }
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
                if let Ok(shutdown) = self.library.get::<GpuShutdownFn>(b"gpu_worker_shutdown") {
                    shutdown();
                }
            }
        }
    }
}
