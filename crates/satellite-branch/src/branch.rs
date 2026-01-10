//! Branch data structure.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Unique branch identifier.
pub type BranchId = u64;

/// Branch status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BranchStatus {
    /// Branch is active (being explored).
    Active = 0,
    /// Branch failed (UNSAT in this branch).
    Failed = 1,
    /// Branch solved (SAT found).
    Solved = 2,
}

impl BranchStatus {
    /// Creates from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Active),
            1 => Some(Self::Failed),
            2 => Some(Self::Solved),
            _ => None,
        }
    }
}

/// A branch in the constraint tree.
pub struct Branch {
    /// Unique identifier.
    id: BranchId,
    /// Parent branch ID (None for root).
    parent_id: Option<BranchId>,
    /// Reference count (number of active children).
    refcount: AtomicU32,
    /// Branch status.
    status: AtomicU32,
    /// Depth in the tree.
    depth: u32,
    /// Number of jobs completed in this branch.
    jobs_completed: AtomicU64,
    /// Confidence score for priority calculation.
    confidence: AtomicU64, // Stored as f64 bits
}

impl Branch {
    /// Creates a new root branch.
    pub fn new_root() -> Self {
        Self {
            id: 0,
            parent_id: None,
            refcount: AtomicU32::new(0),
            status: AtomicU32::new(BranchStatus::Active as u32),
            depth: 0,
            jobs_completed: AtomicU64::new(0),
            confidence: AtomicU64::new(1.0f64.to_bits()),
        }
    }

    /// Creates a new child branch.
    pub fn new_child(id: BranchId, parent_id: BranchId, parent_depth: u32) -> Self {
        Self {
            id,
            parent_id: Some(parent_id),
            refcount: AtomicU32::new(0),
            status: AtomicU32::new(BranchStatus::Active as u32),
            depth: parent_depth + 1,
            jobs_completed: AtomicU64::new(0),
            confidence: AtomicU64::new(1.0f64.to_bits()),
        }
    }

    /// Returns the branch ID.
    pub fn id(&self) -> BranchId {
        self.id
    }

    /// Returns the parent branch ID.
    pub fn parent_id(&self) -> Option<BranchId> {
        self.parent_id
    }

    /// Returns the depth.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Returns the current status.
    pub fn status(&self) -> BranchStatus {
        BranchStatus::from_u8(self.status.load(Ordering::Acquire) as u8)
            .unwrap_or(BranchStatus::Active)
    }

    /// Sets the status.
    pub fn set_status(&self, status: BranchStatus) {
        self.status.store(status as u32, Ordering::Release);
    }

    /// Returns whether the branch has failed.
    pub fn is_failed(&self) -> bool {
        self.status() == BranchStatus::Failed
    }

    /// Returns whether the branch is active.
    pub fn is_active(&self) -> bool {
        self.status() == BranchStatus::Active
    }

    /// Increments the reference count.
    pub fn inc_refcount(&self) -> u32 {
        self.refcount.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Decrements the reference count and returns the new value.
    pub fn dec_refcount(&self) -> u32 {
        self.refcount.fetch_sub(1, Ordering::SeqCst) - 1
    }

    /// Returns the current reference count.
    pub fn refcount(&self) -> u32 {
        self.refcount.load(Ordering::Acquire)
    }

    /// Increments jobs completed.
    pub fn inc_jobs_completed(&self) -> u64 {
        self.jobs_completed.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Returns jobs completed.
    pub fn jobs_completed(&self) -> u64 {
        self.jobs_completed.load(Ordering::Relaxed)
    }

    /// Returns the confidence score.
    pub fn confidence(&self) -> f64 {
        f64::from_bits(self.confidence.load(Ordering::Relaxed))
    }

    /// Sets the confidence score.
    pub fn set_confidence(&self, confidence: f64) {
        self.confidence.store(confidence.to_bits(), Ordering::Relaxed);
    }
}

impl Clone for Branch {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            parent_id: self.parent_id,
            refcount: AtomicU32::new(self.refcount.load(Ordering::Relaxed)),
            status: AtomicU32::new(self.status.load(Ordering::Relaxed)),
            depth: self.depth,
            jobs_completed: AtomicU64::new(self.jobs_completed.load(Ordering::Relaxed)),
            confidence: AtomicU64::new(self.confidence.load(Ordering::Relaxed)),
        }
    }
}
