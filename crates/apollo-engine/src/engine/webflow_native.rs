//! Allocation-safe Native Messaging framing and user-agent transport.

use std::fs;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::engine::webflow_types::{WebFlowEvent, MAX_WEBFLOW_MESSAGE_BYTES};

const MAX_UNIX_SOCKET_PATH_BYTES: usize = 103;
const WEBFLOW_AGENT_SOCKET_NAME: &str = "com.eduardocortez.apollo-webflow";
const BRIDGE_TIMEOUT: Duration = Duration::from_secs(2);
const EVENTS_PER_SECOND: u16 = 64;

pub fn read_native_frame(reader: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut header = [0u8; 4];
    let mut read = 0usize;
    while read < header.len() {
        match reader.read(&mut header[read..]) {
            Ok(0) if read == 0 => return Ok(None),
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated native message header",
                ))
            }
            Ok(count) => read += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    let length = u32::from_ne_bytes(header) as usize;
    if length == 0 || length > MAX_WEBFLOW_MESSAGE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "native message length is outside WebFlow bounds",
        ));
    }
    let mut payload = vec![0u8; length];
    reader.read_exact(&mut payload)?;
    Ok(Some(payload))
}

pub fn write_native_frame(writer: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    if payload.is_empty() || payload.len() > MAX_WEBFLOW_MESSAGE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "native message length is outside WebFlow bounds",
        ));
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "native message too large"))?;
    writer.write_all(&length.to_ne_bytes())?;
    writer.write_all(payload)?;
    writer.flush()
}

#[derive(Debug, Clone, Copy)]
pub struct EventTokenBucket {
    window_started_ms: u64,
    remaining: u16,
}

impl EventTokenBucket {
    pub const fn new(now_ms: u64) -> Self {
        Self {
            window_started_ms: now_ms,
            remaining: EVENTS_PER_SECOND,
        }
    }

    pub fn admit(&mut self, now_ms: u64) -> bool {
        if now_ms.saturating_sub(self.window_started_ms) >= 1_000 {
            self.window_started_ms = now_ms;
            self.remaining = EVENTS_PER_SECOND;
        }
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        true
    }
}

pub fn webflow_agent_socket_path_for(base: &str, uid: u32) -> io::Result<PathBuf> {
    let path = Path::new(base).join(format!("{WEBFLOW_AGENT_SOCKET_NAME}-{uid}.sock"));
    if path.as_os_str().as_encoded_bytes().len() > MAX_UNIX_SOCKET_PATH_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "WebFlow agent socket path exceeds macOS sockaddr_un limit",
        ));
    }
    Ok(path)
}

pub fn webflow_agent_socket_path() -> io::Result<PathBuf> {
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
    webflow_agent_socket_path_for(&base, unsafe { libc::geteuid() } as u32)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeAck {
    pub accepted: bool,
    pub code: u8,
}

impl BridgeAck {
    pub const ACCEPTED: Self = Self {
        accepted: true,
        code: 0,
    };
    pub const REJECTED: Self = Self {
        accepted: false,
        code: 1,
    };
}

pub struct ContextWebFlowServer {
    listener: UnixListener,
    path: PathBuf,
    owner_uid: u32,
}

impl ContextWebFlowServer {
    pub fn bind_default() -> io::Result<Self> {
        Self::bind_at(webflow_agent_socket_path()?)
    }

    pub fn bind_at(path: PathBuf) -> io::Result<Self> {
        let owner_uid = unsafe { libc::geteuid() } as u32;
        let parent = path.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "WebFlow socket has no parent")
        })?;
        let parent_metadata = fs::metadata(parent)?;
        if parent_metadata.uid() != owner_uid {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "WebFlow socket parent is not owned by the active user",
            ));
        }
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            if !metadata.file_type().is_socket() || metadata.uid() != owner_uid {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "unsafe stale WebFlow socket node",
                ));
            }
            fs::remove_file(&path)?;
        }
        let listener = UnixListener::bind(&path)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
        Ok(Self {
            listener,
            path,
            owner_uid,
        })
    }

    pub fn serve_once(
        &self,
        forward: impl FnOnce(WebFlowEvent) -> io::Result<()>,
    ) -> io::Result<()> {
        let (mut stream, _) = self.listener.accept()?;
        stream.set_read_timeout(Some(BRIDGE_TIMEOUT))?;
        stream.set_write_timeout(Some(BRIDGE_TIMEOUT))?;
        if peer_uid(&stream) != Some(self.owner_uid) {
            let payload = serde_json::to_vec(&BridgeAck::REJECTED)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            let _ = write_native_frame(&mut stream, &payload);
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "WebFlow bridge peer UID mismatch",
            ));
        }
        let payload = read_native_frame(&mut stream)?.ok_or_else(|| {
            io::Error::new(io::ErrorKind::UnexpectedEof, "WebFlow bridge disconnected")
        })?;
        let event = WebFlowEvent::from_bounded_json(&payload)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let result = forward(event);
        let ack = if result.is_ok() {
            BridgeAck::ACCEPTED
        } else {
            BridgeAck::REJECTED
        };
        let response = serde_json::to_vec(&ack)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        write_native_frame(&mut stream, &response)?;
        result
    }
}

impl Drop for ContextWebFlowServer {
    fn drop(&mut self) {
        if fs::symlink_metadata(&self.path).is_ok_and(|metadata| {
            metadata.file_type().is_socket() && metadata.uid() == self.owner_uid
        }) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub fn send_event_to_context_agent(event: &WebFlowEvent) -> io::Result<BridgeAck> {
    send_event_to_context_agent_at(&webflow_agent_socket_path()?, event)
}

pub fn send_event_to_context_agent_at(path: &Path, event: &WebFlowEvent) -> io::Result<BridgeAck> {
    let payload = event
        .bounded_json()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut stream = UnixStream::connect(path)?;
    stream.set_read_timeout(Some(BRIDGE_TIMEOUT))?;
    stream.set_write_timeout(Some(BRIDGE_TIMEOUT))?;
    write_native_frame(&mut stream, &payload)?;
    let response = read_native_frame(&mut stream)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "WebFlow agent returned no response",
        )
    })?;
    serde_json::from_slice(&response)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

pub fn process_bridge_payload(
    payload: &[u8],
    bucket: &mut EventTokenBucket,
    now_ms: u64,
    forward: impl FnOnce(WebFlowEvent) -> io::Result<()>,
) -> BridgeAck {
    let Ok(event) = WebFlowEvent::from_bounded_json(payload) else {
        return BridgeAck::REJECTED;
    };
    if !bucket.admit(now_ms) {
        return BridgeAck::REJECTED;
    }
    if forward(event).is_ok() {
        BridgeAck::ACCEPTED
    } else {
        BridgeAck::REJECTED
    }
}

fn peer_uid(stream: &UnixStream) -> Option<u32> {
    #[cfg(target_os = "macos")]
    {
        let mut uid: libc::uid_t = 0;
        let mut gid: libc::gid_t = 0;
        let result = unsafe { libc::getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) };
        (result == 0).then_some(uid as u32)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = stream;
        Some(unsafe { libc::geteuid() } as u32)
    }
}
