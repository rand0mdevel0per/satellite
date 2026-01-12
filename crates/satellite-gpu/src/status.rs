//! GPU status and result types.

/// GPU availability status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuStatus {
    /// GPU is available and initialized
    Ready,
    /// GPU is currently busy processing
    Busy,
    /// GPU is not available (no GPU or driver issues)
    Unavailable,
    /// GPU support not compiled in
    NotCompiled,
}

impl GpuStatus {
    pub fn is_available(&self) -> bool {
        matches!(self, GpuStatus::Ready | GpuStatus::Busy)
    }
}

/// GPU memory information
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuMemoryInfo {
    /// Used memory in bytes
    pub used: usize,
    /// Total memory in bytes
    pub total: usize,
}

impl GpuMemoryInfo {
    /// Get memory usage as a percentage
    pub fn usage_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }

    /// Get free memory in bytes
    pub fn free(&self) -> usize {
        self.total.saturating_sub(self.used)
    }
}

/// BCP result from GPU
#[derive(Debug, Clone, Copy)]
pub struct BcpResult {
    /// Whether a conflict was detected
    pub has_conflict: bool,
    /// ID of the conflicting clause (if any)
    pub conflict_clause_id: Option<usize>,
}

impl BcpResult {
    pub fn no_conflict() -> Self {
        Self {
            has_conflict: false,
            conflict_clause_id: None,
        }
    }

    pub fn conflict(clause_id: usize) -> Self {
        Self {
            has_conflict: true,
            conflict_clause_id: Some(clause_id),
        }
    }
}
