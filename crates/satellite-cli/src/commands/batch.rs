use clap::Args;
use std::path::{Path, PathBuf};
use std::fs;
use std::time::Instant;
use satellite_kit::Solver;
use satellite_format::AdvancedCnf;
use satellite_cdcl::{SatResult, CdclSolver, CdclConfig};
use walkdir::WalkDir;
use rayon::prelude::*;
use serde::Serialize;

#[derive(Args, Debug)]
pub struct BatchArgs {
    /// Input directory containing problem files
    #[arg(long)]
    pub input_dir: PathBuf,

    /// Output directory for results
    #[arg(long)]
    pub output_dir: PathBuf,

    /// Number of parallel workers (0 = auto)
    #[arg(long, default_value_t = 0)]
    pub workers: usize,

    /// Timeout in seconds per problem
    #[arg(long)]
    pub timeout: Option<u64>,
}

#[derive(Serialize)]
struct BatchResult {
    file: String,
    status: String,
    time_ms: u128,
    variables: usize,
    clauses: usize,
}

pub fn run(args: BatchArgs) -> anyhow::Result<()> {
    tracing::info!("Starting batch processing from {:?}", args.input_dir);

    if !args.output_dir.exists() {
        fs::create_dir_all(&args.output_dir)?;
    }

    // Collect files
    let mut files = Vec::new();
    for entry in WalkDir::new(&args.input_dir) {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "json" || ext == "cnf" {
                    files.push(path.to_owned());
                }
            }
        }
    }

    tracing::info!("Found {} files to process", files.len());

    // Setup thread pool if workers specified
    if args.workers > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.workers)
            .build_global()?;
    }

    // Process in parallel
    files.par_iter().for_each(|file_path| {
        if let Err(e) = process_file(file_path, &args.output_dir, args.timeout) {
            tracing::error!("Failed to process {:?}: {}", file_path, e);
        }
    });

    tracing::info!("Batch processing complete");
    Ok(())
}

fn process_file(path: &Path, output_dir: &Path, timeout: Option<u64>) -> anyhow::Result<()> {
    let start = Instant::now();
    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
    
    tracing::info!("Processing {}", filename);

    // Load problem
    // Note: For now assuming AdvancedCNF JSON. DIMACS support would require detection.
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    
    let problem = if extension == "cnf" {
        // TODO: Add DIMACS parser to satellite-kit/format
        tracing::warn!("DIMACS format not yet fully supported in batch mode, skipping {}", filename);
        return Ok(());
    } else {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        match AdvancedCnf::from_reader(reader) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to parse {}: {}", filename, e);
                return Ok(());
            }
        }
    };

    // Use CdclSolver directly for batch processing of existing files
    let mut config = CdclConfig::default();
    // if let Some(t) = timeout { ... }

    let mut cdcl = CdclSolver::with_config(&problem, config);
    let result = cdcl.solve()?;
    
    let duration = start.elapsed();
    
    let status_str = match result {
        SatResult::Sat(_) => "SAT",
        SatResult::Unsat => "UNSAT",
        SatResult::Unknown(r) => "UNKNOWN",
    };

    let file_stem = path.file_stem().unwrap_or_default();
    let result_file = output_dir.join(format!("{}.json", file_stem.to_string_lossy()));
    
    let batch_result = BatchResult {
        file: filename.clone(),
        status: status_str.to_string(),
        time_ms: duration.as_millis(),
        variables: problem.variables.len(),
        clauses: problem.clauses.len(),
    };

    let f = std::fs::File::create(result_file)?;
    serde_json::to_writer_pretty(f, &batch_result)?;

    tracing::info!("Finished {} [{}] in {:?}ms", filename, status_str, duration.as_millis());
    Ok(())
}
