//! Job scheduler for the daemon.

use satellite_protocol::{
    JobId, JobState, JobStatus, ProgressStats, SubmitJobRequest,
    AddConstraintsRequest, SnapshotData,
};
use satellite_kit::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// A running job.
struct RunningJob {
    id: JobId,
    state: JobState,
    stats: ProgressStats,
    // TODO: Add actual solver instance
}

/// Job scheduler.
pub struct Scheduler {
    next_job_id: AtomicU64,
    jobs: HashMap<JobId, RunningJob>,
}

impl Scheduler {
    /// Creates a new scheduler.
    pub fn new() -> Self {
        Self {
            next_job_id: AtomicU64::new(1),
            jobs: HashMap::new(),
        }
    }

    /// Submits a new job.
    pub fn submit_job(&mut self, req: SubmitJobRequest) -> Result<JobId> {
        let job_id = self.next_job_id.fetch_add(1, Ordering::Relaxed);

        let job = RunningJob {
            id: job_id,
            state: JobState::Queued,
            stats: ProgressStats {
                total_branches: 0,
                active_branches: 0,
                failed_branches: 0,
                learned_clauses: 0,
                elapsed_ms: 0,
                cpu_utilization: 0.0,
                gpu_utilization: 0.0,
                memory_bytes: 0,
            },
        };

        self.jobs.insert(job_id, job);

        // TODO: Start actual solving in background task

        Ok(job_id)
    }

    /// Gets job status.
    pub fn get_status(&self, job_id: JobId) -> Option<JobStatus> {
        self.jobs.get(&job_id).map(|job| JobStatus {
            job_id: job.id,
            state: job.state,
            stats: job.stats.clone(),
        })
    }

    /// Cancels a job.
    pub fn cancel_job(&mut self, job_id: JobId) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            job.state = JobState::Cancelled;
        }
    }

    /// Gets a snapshot of a job.
    pub fn get_snapshot(&self, job_id: JobId) -> Option<SnapshotData> {
        self.jobs.get(&job_id).map(|_job| SnapshotData {
            job_id,
            snapshot_json: "{}".to_string(), // TODO: Actual snapshot
        })
    }

    /// Adds constraints to an existing job.
    pub fn add_constraints(&mut self, req: AddConstraintsRequest) -> Result<()> {
        if !self.jobs.contains_key(&req.job_id) {
            return Err(satellite_kit::Error::Internal(
                format!("Job {} not found", req.job_id),
            ));
        }

        // TODO: Actually add constraints to the running solver

        Ok(())
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}
