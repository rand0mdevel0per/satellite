//! WebSocket server for distributed solving.

use futures_util::{SinkExt, StreamExt};
use satellite_protocol::{ClientMessage, ProtocolCodec, ServerMessage};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio_tungstenite::accept_async;

use crate::scheduler::Scheduler;

/// Server state.
pub struct ServerState {
    scheduler: Scheduler,
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
        }
    }
}

/// Runs the WebSocket server.
pub async fn run(listener: TcpListener) -> anyhow::Result<()> {
    let state = Arc::new(RwLock::new(ServerState::new()));

    while let Ok((stream, addr)) = listener.accept().await {
        tracing::info!("New connection from {}", addr);

        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, state).await {
                tracing::error!("Connection error: {}", e);
            }
        });
    }

    Ok(())
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    state: Arc<RwLock<ServerState>>,
) -> anyhow::Result<()> {
    let ws_stream = accept_async(stream).await?;
    let (mut write, mut read) = ws_stream.split();

    while let Some(msg) = read.next().await {
        let msg = msg?;
        if msg.is_binary() {
            let bytes = msg.into_data();

            match ProtocolCodec::decode_client(&bytes) {
                Ok(client_msg) => {
                    let response = handle_message(client_msg, &state).await;
                    let response_bytes = ProtocolCodec::encode_server(&response)?;
                    write
                        .send(tokio_tungstenite::tungstenite::Message::Binary(
                            response_bytes.into(),
                        ))
                        .await?;
                }
                Err(e) => {
                    tracing::warn!("Failed to decode message: {}", e);
                }
            }
        }
    }

    Ok(())
}

async fn handle_message(msg: ClientMessage, state: &Arc<RwLock<ServerState>>) -> ServerMessage {
    match msg {
        ClientMessage::Ping => ServerMessage::Pong,
        ClientMessage::SubmitJob(req) => {
            let mut state = state.write().await;
            match state.scheduler.submit_job(req) {
                Ok(job_id) => ServerMessage::JobAccepted { job_id },
                Err(e) => ServerMessage::Error(satellite_protocol::ErrorResponse {
                    code: satellite_protocol::ErrorCode::InternalError,
                    message: e.to_string(),
                }),
            }
        }
        ClientMessage::QueryStatus(job_id) => {
            let state = state.read().await;
            match state.scheduler.get_status(job_id) {
                Some(status) => ServerMessage::Status(status),
                None => ServerMessage::Error(satellite_protocol::ErrorResponse {
                    code: satellite_protocol::ErrorCode::JobNotFound,
                    message: format!("Job {} not found", job_id),
                }),
            }
        }
        ClientMessage::CancelJob(job_id) => {
            let mut state = state.write().await;
            state.scheduler.cancel_job(job_id);
            ServerMessage::Pong // ACK
        }
        ClientMessage::RequestSnapshot(job_id) => {
            let state = state.read().await;
            match state.scheduler.get_snapshot(job_id) {
                Some(snapshot) => ServerMessage::Snapshot(snapshot),
                None => ServerMessage::Error(satellite_protocol::ErrorResponse {
                    code: satellite_protocol::ErrorCode::JobNotFound,
                    message: format!("Job {} not found", job_id),
                }),
            }
        }
        ClientMessage::AddConstraints(req) => {
            let mut state = state.write().await;
            match state.scheduler.add_constraints(req) {
                Ok(_) => ServerMessage::Pong,
                Err(e) => ServerMessage::Error(satellite_protocol::ErrorResponse {
                    code: satellite_protocol::ErrorCode::InvalidRequest,
                    message: e.to_string(),
                }),
            }
        }
    }
}
