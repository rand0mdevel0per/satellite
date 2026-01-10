//! Branch lifecycle management.

use std::sync::atomic::{AtomicU64, Ordering};
use satellite_lockfree::ConcurrentSkiplist;
use super::branch::{Branch, BranchId, BranchStatus};

/// Manages the lifecycle of all branches.
pub struct BranchManager {
    /// Next branch ID to allocate.
    next_id: AtomicU64,
    /// All branches indexed by ID.
    branches: ConcurrentSkiplist<BranchId, Branch>,
    /// Active branch count.
    active_count: AtomicU64,
    /// Failed branch count.
    failed_count: AtomicU64,
}

impl BranchManager {
    /// Creates a new branch manager with a root branch.
    pub fn new() -> Self {
        let manager = Self {
            next_id: AtomicU64::new(1),
            branches: ConcurrentSkiplist::new(),
            active_count: AtomicU64::new(1),
            failed_count: AtomicU64::new(0),
        };

        // Create root branch
        let root = Branch::new_root();
        manager.branches.insert(0, root);

        manager
    }

    /// Creates child branches for a constraint split.
    ///
    /// Returns the IDs of the created branches.
    pub fn fork(&self, parent_id: BranchId, num_children: usize) -> Vec<BranchId> {
        let parent_depth = self
            .branches
            .get(&parent_id)
            .map(|b| b.depth())
            .unwrap_or(0);

        let mut child_ids = Vec::with_capacity(num_children);

        for _ in 0..num_children {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let child = Branch::new_child(id, parent_id, parent_depth);
            self.branches.insert(id, child);
            child_ids.push(id);
            self.active_count.fetch_add(1, Ordering::Relaxed);
        }

        // Update parent refcount
        if let Some(parent) = self.branches.get(&parent_id) {
            for _ in 0..num_children {
                parent.inc_refcount();
            }
        }

        child_ids
    }

    /// Marks a branch as failed.
    pub fn mark_failed(&self, branch_id: BranchId) {
        if let Some(branch) = self.branches.get(&branch_id) {
            if branch.status() == BranchStatus::Active {
                branch.set_status(BranchStatus::Failed);
                self.active_count.fetch_sub(1, Ordering::Relaxed);
                self.failed_count.fetch_add(1, Ordering::Relaxed);

                // Handle parent refcount
                if let Some(parent_id) = branch.parent_id() {
                    if let Some(parent) = self.branches.get(&parent_id) {
                        let new_count = parent.dec_refcount();
                        if new_count == 0 && parent.is_active() {
                            // Recursively mark parent as failed
                            self.mark_failed(parent_id);
                        }
                    }
                }
            }
        }
    }

    /// Marks a branch as solved.
    pub fn mark_solved(&self, branch_id: BranchId) {
        if let Some(branch) = self.branches.get(&branch_id) {
            branch.set_status(BranchStatus::Solved);
            self.active_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Checks if a branch is failed.
    pub fn is_failed(&self, branch_id: BranchId) -> bool {
        self.branches
            .get(&branch_id)
            .map(|b| b.is_failed())
            .unwrap_or(true)
    }

    /// Returns the number of active branches.
    pub fn active_count(&self) -> u64 {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Returns the number of failed branches.
    pub fn failed_count(&self) -> u64 {
        self.failed_count.load(Ordering::Relaxed)
    }

    /// Returns total branches.
    pub fn total_count(&self) -> u64 {
        self.next_id.load(Ordering::Relaxed)
    }
}

impl Default for BranchManager {
    fn default() -> Self {
        Self::new()
    }
}
