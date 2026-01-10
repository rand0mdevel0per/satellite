//! CLI configuration.

use std::path::PathBuf;

/// CLI configuration loaded from file or environment.
#[derive(Debug, Clone, Default)]
pub struct CliConfig {
    /// Default daemon URL.
    pub daemon_url: Option<String>,
    /// Default number of workers.
    pub default_workers: Option<usize>,
    /// Cache directory.
    pub cache_dir: Option<PathBuf>,
}

impl CliConfig {
    /// Loads configuration from file.
    pub fn load() -> Self {
        // TODO: Load from ~/.satellite/config.toml
        Self::default()
    }
}
