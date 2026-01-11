use clap::Args;
use std::process::Command;
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct InstallArgs {
    /// Re-install even if already exists
    #[arg(long)]
    pub force: bool,
}

pub fn run(args: InstallArgs) -> anyhow::Result<()> {
    tracing::info!("Starting Satellite frontend installation...");

    let home = home::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot find home directory"))?;
    let frontend_dir = home.join(".satellite").join("frontends");
    
    if !frontend_dir.exists() {
        std::fs::create_dir_all(&frontend_dir)?;
    }

    let frontends = vec![
        ("satellite-python-frontend", "satellite_python_frontend"),
        ("satellite-rust-frontend", "satellite_rust_frontend"),
        ("satellite-cpp-frontend", "satellite_cpp_frontend"),
    ];

    let current_dir = std::env::current_dir()?;
    let workspace_root = find_workspace_root(&current_dir)
        .ok_or_else(|| anyhow::anyhow!("Not inside a cargo workspace"))?;

    for (pkg, dll_name) in frontends {
        let dll_filename = if cfg!(target_os = "windows") {
            format!("{}.dll", dll_name)
        } else if cfg!(target_os = "macos") {
            format!("lib{}.dylib", dll_name)
        } else {
            format!("lib{}.so", dll_name)
        };

        let target_path = frontend_dir.join(&dll_filename);
        
        if target_path.exists() && !args.force {
            tracing::info!("Frontend {} already installed.", pkg);
            continue;
        }

        tracing::info!("Building {}...", pkg);
        
        // cargo build -p <pkg> --release
        let status = Command::new("cargo")
            .arg("build")
            .arg("-p")
            .arg(pkg)
            .arg("--release")
            .current_dir(&workspace_root)
            .status()?;
            
        if !status.success() {
            tracing::error!("Failed to build {}", pkg);
            continue;
        }
        
        // Find artifacts
        // Assuming ./target/release/
        let target_dir = workspace_root.join("target").join("release");
        let src_dll = target_dir.join(&dll_filename);
        
        if src_dll.exists() {
            tracing::info!("Installing to {}...", target_path.display());
            std::fs::copy(&src_dll, &target_path)?;
        } else {
            tracing::error!("Could not find built DLL at {}", src_dll.display());
        }
    }
    
    tracing::info!("Installation complete.");
    Ok(())
}

fn find_workspace_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        if current.join("Cargo.toml").exists() {
            // Check if it's a workspace? Simple check if it has [workspace]
            // For now, assume any Cargo.toml up the tree might be it.
            // But better: use `cargo locate-project --workspace --message-format plain`
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}
