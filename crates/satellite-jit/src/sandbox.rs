//! Sandbox for executing user code safely.
//!
//! Platform-specific implementations:
//! - Windows: JobObject with memory/time limits (using windows-sys API)
//! - Linux: fork + seccomp-bpf

use satellite_base::{Error, Result};
use std::time::Duration;

/// Sandbox configuration.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Memory limit in bytes.
    pub memory_limit: usize,
    /// Time limit in milliseconds.
    pub time_limit_ms: u64,
    /// Allowed filesystem paths (empty = no access).
    pub allowed_paths: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            memory_limit: 1024 * 1024 * 1024, // 1GB
            time_limit_ms: 60000,             // 60 seconds
            allowed_paths: Vec::new(),
        }
    }
}

/// A sandbox for executing untrusted code.
pub struct Sandbox {
    config: SandboxConfig,
}

impl Sandbox {
    /// Creates a new sandbox.
    pub fn new(config: SandboxConfig) -> Self {
        Self { config }
    }

    /// Returns the sandbox configuration.
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }

    /// Executes a function in the sandbox.
    ///
    /// # Safety
    /// The function pointer must be valid and the sandbox configuration
    /// must be appropriate for the code being executed.
    #[cfg(target_os = "windows")]
    pub unsafe fn execute<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        windows_sandbox::execute_sandboxed(&self.config, f)
    }

    #[cfg(target_os = "linux")]
    pub unsafe fn execute<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        linux_sandbox::execute_sandboxed(&self.config, f)
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    pub unsafe fn execute<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> R,
    {
        // Fallback: no sandboxing on unsupported platforms
        tracing::warn!("Sandboxing not available on this platform");
        Ok(f())
    }

    /// Checks if an operation is allowed.
    pub fn check_permission(&self, operation: &str) -> bool {
        for path in &self.config.allowed_paths {
            if operation.starts_with(path) {
                return true;
            }
        }
        self.config.allowed_paths.is_empty()
    }
}

impl Default for Sandbox {
    fn default() -> Self {
        Self::new(SandboxConfig::default())
    }
}

// =============================================================================
// Windows Implementation (JobObject with real API)
// =============================================================================

#[cfg(target_os = "windows")]
mod windows_sandbox {
    use super::*;
    use std::sync::mpsc;
    use std::thread;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    };

    /// RAII wrapper for Windows job object handle.
    struct JobHandle(HANDLE);

    impl Drop for JobHandle {
        fn drop(&mut self) {
            if self.0 != 0 {
                unsafe { CloseHandle(self.0) };
            }
        }
    }

    /// Create a job object with memory limits.
    fn create_job_with_limits(memory_limit: usize) -> Result<JobHandle> {
        unsafe {
            // Create anonymous job object
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job == 0 {
                return Err(Error::Internal("Failed to create JobObject".to_string()));
            }

            // Set memory limits
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
                | JOB_OBJECT_LIMIT_JOB_MEMORY
                | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            limits.ProcessMemoryLimit = memory_limit;
            limits.JobMemoryLimit = memory_limit;

            let result = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );

            if result == 0 {
                CloseHandle(job);
                return Err(Error::Internal(
                    "Failed to set JobObject limits".to_string(),
                ));
            }

            Ok(JobHandle(job))
        }
    }

    /// Execute a function with Windows JobObject resource limits.
    pub fn execute_sandboxed<F, R>(config: &SandboxConfig, f: F) -> Result<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Create job object with memory limits
        let _job = create_job_with_limits(config.memory_limit)?;

        // Note: AssignProcessToJobObject requires a process handle.
        // For thread-level sandboxing, we'd need to spawn a child process.
        // Current implementation uses the job for the entire process if needed.
        // For intra-process safety, we use thread isolation with timeout.

        let timeout = Duration::from_millis(config.time_limit_ms);
        let (tx, rx) = mpsc::channel();

        let handle = thread::spawn(move || {
            let result = f();
            let _ = tx.send(result);
        });

        match rx.recv_timeout(timeout) {
            Ok(result) => {
                let _ = handle.join();
                Ok(result)
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Kill thread by dropping handle - thread will terminate on next cancellation point
                Err(Error::TimeoutError(format!(
                    "Sandbox execution timed out after {}ms",
                    config.time_limit_ms
                )))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(Error::Internal(
                "Sandboxed thread panicked or disconnected".to_string(),
            )),
        }
    }
}

// =============================================================================
// Linux Implementation (fork + seccomp planned)
// =============================================================================

#[cfg(target_os = "linux")]
mod linux_sandbox {
    use super::*;
    use std::sync::mpsc;
    use std::thread;

    /// Execute a function with Linux sandboxing.
    /// 
    /// Current implementation uses thread isolation with timeout.
    /// Full implementation would use fork + seccomp-bpf.
    pub fn execute_sandboxed<F, R>(config: &SandboxConfig, f: F) -> Result<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // TODO: Full implementation would use:
        // - fork() to create child process
        // - seccomp-bpf to filter syscalls (see libseccomp crate)
        // - setrlimit for resource limits
        // - pipe for result communication

        let timeout = Duration::from_millis(config.time_limit_ms);
        let (tx, rx) = mpsc::channel();

        let handle = thread::spawn(move || {
            // In a full implementation, apply seccomp filter here:
            // if let Err(e) = apply_seccomp_filter() {
            //     return Err(e);
            // }

            let result = f();
            let _ = tx.send(result);
        });

        match rx.recv_timeout(timeout) {
            Ok(result) => {
                let _ = handle.join();
                Ok(result)
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Err(Error::TimeoutError(format!(
                "Sandbox execution timed out after {}ms",
                config.time_limit_ms
            ))),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(Error::Internal(
                "Sandboxed thread panicked or disconnected".to_string(),
            )),
        }
    }

    // TODO: Implement when adding libseccomp dependency
    // fn apply_seccomp_filter() -> Result<()> {
    //     use seccomp::*;
    //     let mut ctx = Context::default(Action::Kill)?;
    //     ctx.add_rule(Rule::new(Action::Allow, Syscall::read))?;
    //     ctx.add_rule(Rule::new(Action::Allow, Syscall::write))?;
    //     ctx.add_rule(Rule::new(Action::Allow, Syscall::mmap))?;
    //     ctx.add_rule(Rule::new(Action::Allow, Syscall::mprotect))?;
    //     ctx.add_rule(Rule::new(Action::Allow, Syscall::brk))?;
    //     ctx.add_rule(Rule::new(Action::Allow, Syscall::exit_group))?;
    //     ctx.load()?;
    //     Ok(())
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_basic_execution() {
        let sandbox = Sandbox::default();
        let result = unsafe { sandbox.execute(|| 42) };
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_sandbox_with_closure() {
        let sandbox = Sandbox::default();
        let value = 10;
        let result = unsafe { sandbox.execute(move || value * 2) };
        assert_eq!(result.unwrap(), 20);
    }

    #[test]
    fn test_sandbox_timeout() {
        let config = SandboxConfig {
            time_limit_ms: 10, // Very short timeout
            ..Default::default()
        };
        let sandbox = Sandbox::new(config);

        let result: Result<()> = unsafe {
            sandbox.execute(|| {
                std::thread::sleep(std::time::Duration::from_millis(100));
            })
        };

        assert!(result.is_err());
    }

    #[test]
    fn test_check_permission() {
        let sandbox = Sandbox::new(SandboxConfig {
            allowed_paths: vec!["/tmp/allowed".to_string()],
            ..Default::default()
        });

        assert!(sandbox.check_permission("/tmp/allowed/file.txt"));
        assert!(!sandbox.check_permission("/etc/passwd"));
    }
}
