//! Reference counting for branch lifecycle.

use super::branch::{Branch, BranchId, BranchStatus};

/// Callback for branch events.
pub trait BranchCallback: Send + Sync {
    /// Called when a branch transitions to failed.
    fn on_branch_failed(&self, branch_id: BranchId, parent_id: Option<BranchId>);

    /// Called when a branch is solved.
    fn on_branch_solved(&self, branch_id: BranchId);
}

/// Handles reference counting for branch failure propagation.
pub struct RefCountHandler<C: BranchCallback> {
    callback: C,
}

impl<C: BranchCallback> RefCountHandler<C> {
    /// Creates a new handler.
    pub fn new(callback: C) -> Self {
        Self { callback }
    }

    /// Handles a child branch failure.
    ///
    /// Decrements parent refcount and propagates failure if all children failed.
    pub fn on_child_fail(&self, parent: &Branch) {
        let new_count = parent.dec_refcount();

        if new_count == 0 {
            // All children failed, mark parent as failed
            parent.set_status(BranchStatus::Failed);

            // Notify callback to check grandparent
            self.callback
                .on_branch_failed(parent.id(), parent.parent_id());
        }
    }

    /// Handles a branch being solved.
    pub fn on_branch_solved(&self, branch: &Branch) {
        branch.set_status(BranchStatus::Solved);
        self.callback.on_branch_solved(branch.id());
    }
}

/// Simple callback implementation that does nothing.
#[derive(Default)]
pub struct NoOpCallback;

impl BranchCallback for NoOpCallback {
    fn on_branch_failed(&self, _branch_id: BranchId, _parent_id: Option<BranchId>) {}
    fn on_branch_solved(&self, _branch_id: BranchId) {}
}
