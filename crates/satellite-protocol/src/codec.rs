//! Protocol codec for framing and serialization.

use crate::messages::{ClientMessage, ServerMessage};
use satellite_base::{Error, Result};

/// Codec for encoding/decoding protocol messages.
pub struct ProtocolCodec;

impl ProtocolCodec {
    /// Encodes a client message to bytes.
    pub fn encode_client(msg: &ClientMessage) -> Result<Vec<u8>> {
        rkyv::to_bytes::<rkyv::rancor::Error>(msg)
            .map(|v| v.to_vec())
            .map_err(|e| Error::Serialization(e.to_string()))
    }

    /// Decodes a client message from bytes.
    pub fn decode_client(bytes: &[u8]) -> Result<ClientMessage> {
        rkyv::from_bytes::<ClientMessage, rkyv::rancor::Error>(bytes)
            .map_err(|e| Error::Serialization(e.to_string()))
    }

    /// Encodes a server message to bytes.
    pub fn encode_server(msg: &ServerMessage) -> Result<Vec<u8>> {
        rkyv::to_bytes::<rkyv::rancor::Error>(msg)
            .map(|v| v.to_vec())
            .map_err(|e| Error::Serialization(e.to_string()))
    }

    /// Decodes a server message from bytes.
    pub fn decode_server(bytes: &[u8]) -> Result<ServerMessage> {
        rkyv::from_bytes::<ServerMessage, rkyv::rancor::Error>(bytes)
            .map_err(|e| Error::Serialization(e.to_string()))
    }

    /// Frames a message with length prefix (4 bytes, big-endian).
    #[must_use]
    pub fn frame(data: &[u8]) -> Vec<u8> {
        let len = data.len() as u32;
        let mut framed = Vec::with_capacity(4 + data.len());
        framed.extend_from_slice(&len.to_be_bytes());
        framed.extend_from_slice(data);
        framed
    }

    /// Reads the frame length from a 4-byte prefix.
    #[must_use]
    pub fn read_frame_len(header: &[u8; 4]) -> u32 {
        u32::from_be_bytes(*header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::*;

    #[test]
    fn test_roundtrip_client() {
        let msg = ClientMessage::Ping;
        let encoded = ProtocolCodec::encode_client(&msg).unwrap();
        let decoded = ProtocolCodec::decode_client(&encoded).unwrap();
        assert!(matches!(decoded, ClientMessage::Ping));
    }

    #[test]
    fn test_roundtrip_server() {
        let msg = ServerMessage::Pong;
        let encoded = ProtocolCodec::encode_server(&msg).unwrap();
        let decoded = ProtocolCodec::decode_server(&encoded).unwrap();
        assert!(matches!(decoded, ServerMessage::Pong));
    }
}
