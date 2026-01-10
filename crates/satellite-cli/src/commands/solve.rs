//! Solve command.

use clap::Args;
use satellite_kit::Solver;
use satellite_format::{AdvancedCnf, DimacsCnf};
use std::path::PathBuf;
use std::fs;
use std::time::Instant;

#[derive(Args)]
pub struct SolveArgs {
    /// Input file path
    #[arg(required = true)]
    pub input: PathBuf,

    /// Input format (auto-detect from extension if not specified)
    #[arg(short, long, value_parser = ["dimacs", "cnf", "json", "advanced-cnf"])]
    pub format: Option<String>,

    /// Output file for the solution
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Timeout in seconds
    #[arg(short, long)]
    pub timeout: Option<u64>,

    /// Disable GPU acceleration
    #[arg(long)]
    pub no_gpu: bool,

    /// Number of CPU workers
    #[arg(long)]
    pub workers: Option<usize>,

    /// Output profiling data to file
    #[arg(long)]
    pub profile: Option<PathBuf>,
}

pub fn run(args: SolveArgs) -> anyhow::Result<()> {
    tracing::info!("Loading problem from {:?}", args.input);

    let content = fs::read_to_string(&args.input)?;
    let format = args.format.as_deref().unwrap_or_else(|| {
        args.input
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("cnf")
    });

    let problem = match format {
        "dimacs" | "cnf" => {
            let dimacs = DimacsCnf::from_str(&content)?;
            tracing::info!("Loaded DIMACS: {} vars, {} clauses", dimacs.num_vars, dimacs.clauses.len());
            dimacs.to_advanced_cnf()
        }
        "json" | "advanced-cnf" => {
            AdvancedCnf::from_json(&content)?
        }
        _ => {
            anyhow::bail!("Unknown format: {}", format);
        }
    };

    tracing::info!(
        "Problem: {} variables, {} clauses",
        problem.variables.len(),
        problem.clauses.len()
    );

    // Create and run solver
    let mut solver = Solver::new();

    // Add clauses from problem
    for clause in &problem.clauses {
        solver.add_clause(clause.literals.iter().copied());
    }

    let start = Instant::now();
    let result = solver.solve()?;
    let elapsed = start.elapsed();

    match result {
        satellite_kit::SatResult::Sat(model) => {
            println!("s SATISFIABLE");
            if args.output.is_some() {
                // Write model to output file
                let model_str: String = model
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| if v { (i + 1) as i64 } else { -((i + 1) as i64) })
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                if let Some(output) = &args.output {
                    fs::write(output, format!("v {} 0\n", model_str))?;
                }
            }
        }
        satellite_kit::SatResult::Unsat => {
            println!("s UNSATISFIABLE");
        }
        satellite_kit::SatResult::Unknown(reason) => {
            println!("s UNKNOWN ({})", reason);
        }
    }

    tracing::info!("Solved in {:?}", elapsed);
    println!("c Time: {:.3}s", elapsed.as_secs_f64());

    Ok(())
}
