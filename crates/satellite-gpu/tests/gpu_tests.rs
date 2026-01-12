//! Unit tests for satellite-gpu.

use satellite_gpu::{GpuWorker, GpuStatus, GpuError};

#[test]
fn test_worker_creation() {
    // Without GPU features, this should fail gracefully
    let result = GpuWorker::new();
    
    // Either succeeds or returns NotAvailable
    match result {
        Ok(worker) => {
            assert!(worker.device_count() >= 0);
        }
        Err(GpuError::NotAvailable) => {
            // Expected when no GPU is available
        }
        Err(e) => {
            panic!("Unexpected error: {:?}", e);
        }
    }
}

#[test]
fn test_status_enum() {
    assert!(GpuStatus::Ready.is_available());
    assert!(GpuStatus::Busy.is_available());
    assert!(!GpuStatus::Unavailable.is_available());
    assert!(!GpuStatus::NotCompiled.is_available());
}

#[test]
fn test_memory_info() {
    use satellite_gpu::GpuMemoryInfo;
    
    let info = GpuMemoryInfo {
        used: 1024 * 1024 * 100,  // 100 MB
        total: 1024 * 1024 * 1000, // 1000 MB
    };
    
    assert_eq!(info.usage_percent(), 10.0);
    assert_eq!(info.free(), 1024 * 1024 * 900);
}

#[test]
fn test_bcp_result() {
    use satellite_gpu::BcpResult;
    
    let no_conflict = BcpResult::no_conflict();
    assert!(!no_conflict.has_conflict);
    assert!(no_conflict.conflict_clause_id.is_none());
    
    let conflict = BcpResult::conflict(42);
    assert!(conflict.has_conflict);
    assert_eq!(conflict.conflict_clause_id, Some(42));
}

#[test]
fn test_worker_default() {
    // Default should not panic
    let worker = GpuWorker::default();
    let _ = worker.status();
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    
    #[test]
    fn test_cuda_init() {
        let worker = GpuWorker::new().expect("CUDA init failed");
        assert!(worker.is_available());
        assert!(worker.device_count() > 0);
    }
    
    #[test]
    fn test_cuda_memory() {
        let worker = GpuWorker::new().expect("CUDA init failed");
        let info = worker.memory_info().expect("Memory info failed");
        assert!(info.total > 0);
    }
    
    #[test]
    fn test_cuda_bcp() {
        let worker = GpuWorker::new().expect("CUDA init failed");
        
        // Simple clause: (x1 OR x2)
        // Represented as [1, 2, 0] (0-terminated)
        let clauses: Vec<i64> = vec![1, 2, 0];
        let assignments: Vec<i8> = vec![1, 0];  // x1=true, x2=unassigned
        
        let result = worker.bcp_sync(&clauses, 1, &assignments)
            .expect("BCP failed");
        
        // Clause is satisfied (x1=true)
        assert!(!result.has_conflict);
    }
    
    #[test]
    fn test_cuda_bcp_conflict() {
        let worker = GpuWorker::new().expect("CUDA init failed");
        
        // Clause: (NOT x1) but x1=true → conflict
        let clauses: Vec<i64> = vec![-1, 0];
        let assignments: Vec<i8> = vec![1];  // x1=true
        
        let result = worker.bcp_sync(&clauses, 1, &assignments)
            .expect("BCP failed");
        
        assert!(result.has_conflict);
    }
}
