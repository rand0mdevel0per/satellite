//! GPU worker wrapper providing safe Rust API.

use crate::ffi;
use crate::status::{GpuStatus, GpuMemoryInfo, BcpResult};
use crate::{GpuError, Result};

/// GPU worker for accelerated SAT solving.
///
/// Provides safe Rust API for GPU-accelerated Boolean Constraint Propagation.
/// 
/// # Example
/// ```ignore
/// use satellite_gpu::GpuWorker;
/// 
/// let worker = GpuWorker::new()?;
/// if worker.is_available() {
///     println!("GPU device count: {}", worker.device_count());
/// }
/// ```
pub struct GpuWorker {
    initialized: bool,
}

impl GpuWorker {
    /// Create a new GPU worker and initialize the GPU.
    pub fn new() -> Result<Self> {
        let result = unsafe { ffi::gpu_worker_init() };
        
        match result {
            ffi::GPU_OK => Ok(Self { initialized: true }),
            ffi::GPU_NOT_AVAILABLE => Err(GpuError::NotAvailable),
            _ => Err(GpuError::InitFailed),
        }
    }

    /// Check if GPU is available.
    pub fn is_available(&self) -> bool {
        unsafe { ffi::gpu_worker_is_available() != 0 }
    }

    /// Get the number of GPU devices.
    pub fn device_count(&self) -> i32 {
        unsafe { ffi::gpu_worker_device_count() }
    }

    /// Get current GPU status.
    pub fn status(&self) -> GpuStatus {
        if !self.initialized {
            return GpuStatus::NotCompiled;
        }
        
        if self.is_available() {
            GpuStatus::Ready
        } else {
            GpuStatus::Unavailable
        }
    }

    /// Get GPU memory information.
    pub fn memory_info(&self) -> Result<GpuMemoryInfo> {
        let mut used: usize = 0;
        let mut total: usize = 0;
        
        let result = unsafe {
            ffi::gpu_worker_memory_info(&mut used, &mut total)
        };
        
        match result {
            ffi::GPU_OK => Ok(GpuMemoryInfo { used, total }),
            ffi::GPU_NOT_AVAILABLE => Err(GpuError::NotAvailable),
            _ => Err(GpuError::Error("Failed to get memory info".into())),
        }
    }

    /// Submit a BCP job to the GPU.
    /// 
    /// # Arguments
    /// * `clause_data` - Flattened clause literals (0-terminated per clause)
    /// * `num_clauses` - Number of clauses
    /// * `assignments` - Variable assignments (-1=false, 0=unassigned, 1=true)
    pub fn submit_bcp(
        &self,
        clause_data: &[i64],
        num_clauses: usize,
        assignments: &[i8],
    ) -> Result<()> {
        if !self.initialized {
            return Err(GpuError::NotAvailable);
        }

        let result = unsafe {
            ffi::gpu_worker_submit_bcp(
                clause_data.as_ptr(),
                num_clauses,
                assignments.as_ptr(),
                assignments.len(),
            )
        };

        match result {
            ffi::GPU_OK => Ok(()),
            ffi::GPU_NOT_AVAILABLE => Err(GpuError::NotAvailable),
            _ => Err(GpuError::Error("BCP submission failed".into())),
        }
    }

    /// Poll for BCP result (non-blocking).
    /// 
    /// Returns `Ok(None)` if no result is ready yet.
    pub fn poll_result(&self) -> Result<Option<BcpResult>> {
        let mut ffi_result = ffi::GpuBcpResult {
            has_conflict: 0,
            conflict_clause_id: 0,
        };

        let status = unsafe { ffi::gpu_worker_poll_result(&mut ffi_result) };

        match status {
            0 => {
                // Result available
                let result = if ffi_result.has_conflict != 0 {
                    BcpResult::conflict(ffi_result.conflict_clause_id)
                } else {
                    BcpResult::no_conflict()
                };
                Ok(Some(result))
            }
            1 => Ok(None), // No result ready
            _ => Err(GpuError::Error("Poll failed".into())),
        }
    }

    /// Synchronize with GPU (wait for all pending operations).
    pub fn sync(&self) {
        unsafe { ffi::gpu_worker_sync() }
    }

    /// Submit BCP and wait for result (blocking).
    pub fn bcp_sync(
        &self,
        clause_data: &[i64],
        num_clauses: usize,
        assignments: &[i8],
    ) -> Result<BcpResult> {
        self.submit_bcp(clause_data, num_clauses, assignments)?;
        self.sync();
        
        // Wait for result
        loop {
            if let Some(result) = self.poll_result()? {
                return Ok(result);
            }
            std::hint::spin_loop();
        }
    }
}

impl Drop for GpuWorker {
    fn drop(&mut self) {
        if self.initialized {
            unsafe { ffi::gpu_worker_shutdown() }
        }
    }
}

impl Default for GpuWorker {
    fn default() -> Self {
        Self::new().unwrap_or(Self { initialized: false })
    }
}
