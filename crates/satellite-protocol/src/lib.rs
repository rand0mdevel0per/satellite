//! # satellite-protocol
//!
//! Communication protocol for CLI ↔ Daemon using rkyv for efficient serialization.
//!
//! This crate defines the message types and serialization logic for:
//! - Job submission
//! - Progress updates
//! - Result retrieval
//! - Checkpoint/restore operations

pub mod codec;
pub mod messages;

pub use codec::ProtocolCodec;
pub use messages::*;
