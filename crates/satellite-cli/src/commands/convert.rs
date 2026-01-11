//! Convert command.

use clap::Args;
use satellite_format::{AdvancedCnf, DimacsCnf};
use std::fs;
use std::path::PathBuf;

#[derive(Args)]
pub struct ConvertArgs {
    /// Input file path
    #[arg(required = true)]
    pub input: PathBuf,

    /// Output file path
    #[arg(required = true)]
    pub output: PathBuf,

    /// Input format
    #[arg(short = 'f', long, value_parser = ["dimacs", "json", "jsonc"])]
    pub from: Option<String>,

    /// Output format
    #[arg(short = 't', long, value_parser = ["dimacs", "json"])]
    pub to: Option<String>,
}

pub fn run(args: ConvertArgs) -> anyhow::Result<()> {
    let input_format = args.from.as_deref().unwrap_or_else(|| {
        args.input
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("cnf")
    });

    let output_format = args.to.as_deref().unwrap_or_else(|| {
        args.output
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("json")
    });

    let content = fs::read_to_string(&args.input)?;

    // Parse input
    let advanced_cnf = match input_format {
        "dimacs" | "cnf" => {
            let dimacs = DimacsCnf::from_str(&content)?;
            dimacs.to_advanced_cnf()
        }
        "json" => AdvancedCnf::from_json(&content)?,
        "jsonc" | "sat" => {
            let canonical: satellite_format::CanonicalCnf = serde_json::from_str(&content)?;
            canonical.try_into().map_err(|e: String| anyhow::anyhow!(e))?
        }
        _ => anyhow::bail!("Unknown input format: {}", input_format),
    };

    // Write output
    match output_format {
        "dimacs" | "cnf" => {
            // Convert back to DIMACS
            let dimacs = DimacsCnf {
                num_vars: advanced_cnf.variables.len(),
                clauses: advanced_cnf
                    .clauses
                    .iter()
                    .map(|c| c.literals.clone())
                    .collect(),
            };
            fs::write(&args.output, dimacs.to_dimacs())?;
        }
        "json" => {
            let json = advanced_cnf.to_json()?;
            fs::write(&args.output, json)?;
        }
        _ => anyhow::bail!("Unknown output format: {}", output_format),
    }

    println!(
        "Converted {} -> {}",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}
