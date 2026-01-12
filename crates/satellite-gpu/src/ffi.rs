//! FFI bindings to the C GPU worker API.

use std::os::raw::c_int;

/// BCP result from GPU
#[repr(C)]
pub struct GpuBcpResult {
    pub has_conflict: c_int,
    pub conflict_clause_id: usize,
}

/// GPU error codes
pub const GPU_OK: i32 = 0;
pub const GPU_ERROR: i32 = -1;
pub const GPU_NOT_AVAILABLE: i32 = -2;

#[cfg(feature = "cuda")]
#[link(name = "gpu_worker")]
extern "C" {
    pub fn gpu_worker_init() -> i32;
    pub fn gpu_worker_shutdown();
    pub fn gpu_worker_is_available() -> i32;
    pub fn gpu_worker_device_count() -> i32;
    pub fn gpu_worker_submit_bcp(
        clause_data: *const i64,
        num_clauses: usize,
        assignments: *const i8,
        num_vars: usize,
    ) -> i32;
    pub fn gpu_worker_poll_result(result: *mut GpuBcpResult) -> i32;
    pub fn gpu_worker_sync();
    pub fn gpu_worker_memory_info(used: *mut usize, total: *mut usize) -> i32;
}

#[cfg(feature = "hip")]
#[link(name = "gpu_worker_hip")]
extern "C" {
    pub fn gpu_worker_init() -> i32;
    pub fn gpu_worker_shutdown();
    pub fn gpu_worker_is_available() -> i32;
    pub fn gpu_worker_device_count() -> i32;
    pub fn gpu_worker_submit_bcp(
        clause_data: *const i64,
        num_clauses: usize,
        assignments: *const i8,
        num_vars: usize,
    ) -> i32;
    pub fn gpu_worker_poll_result(result: *mut GpuBcpResult) -> i32;
    pub fn gpu_worker_sync();
    pub fn gpu_worker_memory_info(used: *mut usize, total: *mut usize) -> i32;
}

// Stub implementations when no GPU feature is enabled
#[cfg(not(any(feature = "cuda", feature = "hip")))]
pub mod stubs {
    use super::*;
    
    pub unsafe fn gpu_worker_init() -> i32 {
        GPU_NOT_AVAILABLE
    }
    
    pub unsafe fn gpu_worker_shutdown() {}
    
    pub unsafe fn gpu_worker_is_available() -> i32 {
        0
    }
    
    pub unsafe fn gpu_worker_device_count() -> i32 {
        0
    }
    
    pub unsafe fn gpu_worker_submit_bcp(
        _clause_data: *const i64,
        _num_clauses: usize,
        _assignments: *const i8,
        _num_vars: usize,
    ) -> i32 {
        GPU_NOT_AVAILABLE
    }
    
    pub unsafe fn gpu_worker_poll_result(_result: *mut GpuBcpResult) -> i32 {
        GPU_NOT_AVAILABLE
    }
    
    pub unsafe fn gpu_worker_sync() {}
    
    pub unsafe fn gpu_worker_memory_info(used: *mut usize, total: *mut usize) -> i32 {
        if !used.is_null() {
            *used = 0;
        }
        if !total.is_null() {
            *total = 0;
        }
        GPU_NOT_AVAILABLE
    }
}

#[cfg(not(any(feature = "cuda", feature = "hip")))]
pub use stubs::*;
