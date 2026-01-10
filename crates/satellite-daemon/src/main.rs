//! Satellite Daemon - Distributed solving server.

mod server;
mod scheduler;
mod checkpoint;

use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let addr = std::env::var("SATELLITE_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    tracing::info!("Starting Satellite daemon on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    server::run(listener).await?;

    Ok(())
}
