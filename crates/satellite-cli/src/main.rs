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
        .init();

    match cli.command {
        Commands::Solve(args) => commands::solve::run(args),
        Commands::Convert(args) => commands::convert::run(args),
        Commands::Profile(args) => commands::profile::run(args),
        Commands::Connect { url } => {
            println!("Connecting to daemon at {}...", url);
            // TODO: Implement daemon connection
            Ok(())
        }
    }
}
