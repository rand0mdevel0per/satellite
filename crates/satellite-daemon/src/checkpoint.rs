//! Checkpoint and restore functionality.

use satellite_format::Snapshot;
use satellite_kit::Result;
use std::fs;
use std::path::PathBuf;

/// Checkpoint manager for long-running solves.
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
}

impl CheckpointManager {
    /// Creates a new checkpoint manager.
    pub fn new(checkpoint_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&checkpoint_dir)?;
        Ok(Self { checkpoint_dir })
    }

    /// Saves a checkpoint.
    pub fn save(&self, job_id: u64, snapshot: &Snapshot) -> Result<PathBuf> {
        let path = self
            .checkpoint_dir
            .join(format!("job_{}.checkpoint", job_id));
        let json = snapshot
            .to_json()
            .map_err(|e| satellite_kit::Error::Serialization(e.to_string()))?;
        fs::write(&path, json)?;
        Ok(path)
    }

    /// Loads a checkpoint.
    pub fn load(&self, job_id: u64) -> Result<Option<Snapshot>> {
        let path = self
            .checkpoint_dir
            .join(format!("job_{}.checkpoint", job_id));
        if !path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&path)?;
        let snapshot = Snapshot::from_json(&json)
            .map_err(|e| satellite_kit::Error::Serialization(e.to_string()))?;

        Ok(Some(snapshot))
    }

    /// Lists all checkpoints.
    pub fn list(&self) -> Result<Vec<u64>> {
        let mut job_ids = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            if let Some(name) = name.to_str() {
                if name.starts_with("job_") && name.ends_with(".checkpoint") {
                    if let Ok(id) = name[4..name.len() - 11].parse() {
                        job_ids.push(id);
                    }
                }
            }
        }

        Ok(job_ids)
    }

    /// Deletes a checkpoint.
    pub fn delete(&self, job_id: u64) -> Result<()> {
        let path = self
            .checkpoint_dir
            .join(format!("job_{}.checkpoint", job_id));
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }
}
