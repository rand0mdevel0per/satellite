//! Protocol message types.

use rkyv::{Archive, Deserialize, Serialize};

/// Unique job identifier.
pub type JobId = u64;

/// Client request messages.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub enum ClientMessage {
    /// Submit a new solving job.
    SubmitJob(SubmitJobRequest),
    /// Query job status.
    QueryStatus(JobId),
    /// Cancel a running job.
    CancelJob(JobId),
    /// Request a snapshot.
    RequestSnapshot(JobId),
    /// Add incremental constraints.
    AddConstraints(AddConstraintsRequest),
    /// Ping for keepalive.
    Ping,
}

/// Server response messages.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub enum ServerMessage {
    /// Job accepted.
    JobAccepted { job_id: JobId },
    /// Job status update.
    Status(JobStatus),
    /// Job completed with result.
    Result(JobResult),
    /// Snapshot data.
    Snapshot(SnapshotData),
    /// Error response.
    Error(ErrorResponse),
    /// Pong response.
    Pong,
}

/// Request to submit a new job.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct SubmitJobRequest {
    /// Problem in Advanced-CNF JSON format.
    pub problem_json: String,
    /// Optional timeout in seconds.
    pub timeout_secs: Option<u64>,
    /// Whether to enable GPU.
    pub use_gpu: bool,
    /// Number of CPU workers (0 = auto).
    pub cpu_workers: u32,
}

/// Request to add constraints to an existing job.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct AddConstraintsRequest {
    /// Job to add constraints to.
    pub job_id: JobId,
    /// New clauses to add.
    pub clauses: Vec<Vec<i64>>,
}

/// Job status information.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct JobStatus {
    /// Job identifier.
    pub job_id: JobId,
    /// Current state.
    pub state: JobState,
    /// Progress stats.
    pub stats: ProgressStats,
}

/// Job state.
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub enum JobState {
    /// Job is queued.
    Queued,
    /// Job is running.
    Running,
    /// Job completed successfully.
    Completed,
    /// Job was cancelled.
    Cancelled,
    /// Job failed with error.
    Failed,
}

/// Progress statistics.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct ProgressStats {
    /// Total branches explored.
    pub total_branches: u64,
    /// Active branches.
    pub active_branches: u64,
    /// Failed branches.
    pub failed_branches: u64,
    /// Learned clauses.
    pub learned_clauses: u64,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: u64,
    /// CPU utilization (0.0 - 1.0).
    pub cpu_utilization: f32,
    /// GPU utilization (0.0 - 1.0).
    pub gpu_utilization: f32,
    /// Memory usage in bytes.
    pub memory_bytes: u64,
}

/// Job result.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub enum JobResult {
    /// Satisfiable with model.
    Sat {
        job_id: JobId,
        /// Variable assignments (DIMACS format).
        model: Vec<i64>,
    },
    /// Unsatisfiable.
    Unsat { job_id: JobId },
    /// Unknown (timeout or resource limit).
    Unknown {
        job_id: JobId,
        reason: String,
    },
}

/// Snapshot data.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct SnapshotData {
    /// Job identifier.
    pub job_id: JobId,
    /// Snapshot JSON.
    pub snapshot_json: String,
}

/// Error response.
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub struct ErrorResponse {
    /// Error code.
    pub code: ErrorCode,
    /// Error message.
    pub message: String,
}

/// Error codes.
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub enum ErrorCode {
    /// Job not found.
    JobNotFound,
    /// Invalid request.
    InvalidRequest,
    /// Resource exhausted.
    ResourceExhausted,
    /// Internal error.
    InternalError,
    /// GPU error.
    GpuError,
}
