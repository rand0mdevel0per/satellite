//! Satellite CLI - Command-line interface for the SAT solver.

mod commands;
mod config;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "satellite")]
#[command(author, version, about = "High-Performance Parallel SAT Solver", long_about = None)]
struct Cli {
    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Solve a SAT problem
    Solve(commands::solve::SolveArgs),
    /// Convert between file formats
    Convert(commands::convert::ConvertArgs),
    /// Profile solver performance
    Profile(commands::profile::ProfileArgs),
    /// Connect to a daemon for distributed solving
    Connect {
        /// Daemon WebSocket URL
        #[arg(long, default_value = "ws://localhost:8080")]
        url: String,
    },
    /// Run batch processing
    Batch(commands::batch::BatchArgs),
    /// Install/Update components (frontends)
    Install(commands::install::InstallArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let filter = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(filter))
        .with_env_filter(EnvFilter::new(filter))
        .init();

    if let Err(e) = init_workspace() {
        tracing::warn!("Failed to initialize workspace: {}", e);
    }

    match cli.command {
        Commands::Solve(args) => commands::solve::run(args),
        Commands::Convert(args) => commands::convert::run(args),
        Commands::Profile(args) => commands::profile::run(args),
        Commands::Connect { url } => {
            println!("Connecting to daemon at {}...", url);
            Ok(())
        }
        Commands::Batch(args) => commands::batch::run(args),
        Commands::Install(args) => commands::install::run(args),
    }
}

fn init_workspace() -> anyhow::Result<()> {
    if let Some(home) = home::home_dir() {
        let satellite_dir = home.join(".satellite");
        let frontends_dir = satellite_dir.join("frontends");
        
        if !satellite_dir.exists() {
            std::fs::create_dir(&satellite_dir)?;
        }
        
        let needs_install = if !frontends_dir.exists() {
            std::fs::create_dir(&frontends_dir)?;
            true
        } else {
            // Check if empty
            match std::fs::read_dir(&frontends_dir) {
                Ok(mut iter) => iter.next().is_none(),
                Err(_) => true,
            }
        };

        if needs_install {
            tracing::info!("Frontends not detected. Attempting to install from workspace...");
            // We ignore errors here so the tool keeps running even if install fails
            if let Err(e) = commands::install::run(commands::install::InstallArgs { force: false }) {
                tracing::warn!("Auto-install failed (non-fatal): {}", e);
            }
        }
    }
    Ok(())
}
