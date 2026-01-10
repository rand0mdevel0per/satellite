//! # satellite-protocol
//!
//! Communication protocol for CLI ↔ Daemon using rkyv for efficient serialization.
//!
//! This crate defines the message types and serialization logic for:
//! - Job submission
//! - Progress updates
//! - Result retrieval
//! - Checkpoint/restore operations

pub mod messages;
pub mod codec;

pub use messages::*;
pub use codec::ProtocolCodec;
