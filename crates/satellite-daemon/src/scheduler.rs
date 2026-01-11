//! Job scheduler for the daemon.

use satellite_kit::Result;
use satellite_protocol::{
    AddConstraintsRequest, JobId, JobState, JobStatus, ProgressStats, SnapshotData,
    SubmitJobRequest,
};
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
            return Err(satellite_kit::Error::Internal(format!(
                "Job {} not found",
                req.job_id
            )));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_submit_job() {
        let mut scheduler = Scheduler::new();
        
        let req = SubmitJobRequest {
            problem_json: r#"{"clauses":[[1,2],[-1,3]]}"#.to_string(),
            timeout_secs: None,
            use_gpu: false,
            cpu_workers: 0,
        };
        
        let job_id = scheduler.submit_job(req).unwrap();
        assert!(job_id > 0, "Job ID should be positive");
    }

    #[test]
    fn test_scheduler_get_status() {
        let mut scheduler = Scheduler::new();
        
        let req = SubmitJobRequest {
            problem_json: "{}".to_string(),
            timeout_secs: None,
            use_gpu: false,
            cpu_workers: 0,
        };
        
        let job_id = scheduler.submit_job(req).unwrap();
        
        let status = scheduler.get_status(job_id);
        assert!(status.is_some());
        
        let status = status.unwrap();
        assert_eq!(status.job_id, job_id);
        assert_eq!(status.state, JobState::Queued);
    }

    #[test]
    fn test_scheduler_cancel_job() {
        let mut scheduler = Scheduler::new();
        
        let req = SubmitJobRequest {
            problem_json: "{}".to_string(),
            timeout_secs: None,
            use_gpu: false,
            cpu_workers: 0,
        };
        
        let job_id = scheduler.submit_job(req).unwrap();
        scheduler.cancel_job(job_id);
        
        let status = scheduler.get_status(job_id).unwrap();
        assert_eq!(status.state, JobState::Cancelled);
    }

    #[test]
    fn test_scheduler_get_status_nonexistent() {
        let scheduler = Scheduler::new();
        
        let status = scheduler.get_status(99999);
        assert!(status.is_none());
    }

    #[test]
    fn test_scheduler_multiple_jobs() {
        let mut scheduler = Scheduler::new();
        
        let id1 = scheduler.submit_job(SubmitJobRequest {
            problem_json: "{}".to_string(),
            timeout_secs: None,
            use_gpu: false,
            cpu_workers: 0,
        }).unwrap();
        
        let id2 = scheduler.submit_job(SubmitJobRequest {
            problem_json: "{}".to_string(),
            timeout_secs: None,
            use_gpu: false,
            cpu_workers: 0,
        }).unwrap();
        
        let id3 = scheduler.submit_job(SubmitJobRequest {
            problem_json: "{}".to_string(),
            timeout_secs: None,
            use_gpu: false,
            cpu_workers: 0,
        }).unwrap();
        
        // IDs should be unique and sequential
        assert!(id1 < id2);
        assert!(id2 < id3);
        
        // All should have status
        assert!(scheduler.get_status(id1).is_some());
        assert!(scheduler.get_status(id2).is_some());
        assert!(scheduler.get_status(id3).is_some());
    }
}
