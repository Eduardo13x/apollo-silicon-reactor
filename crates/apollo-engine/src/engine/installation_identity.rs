use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InstallationId(pub u64);

impl InstallationId {
    pub const UNKNOWN: Self = Self(0);

    pub fn is_known(self) -> bool {
        self != Self::UNKNOWN
    }
}

pub fn load_or_create(path: &Path) -> io::Result<InstallationId> {
    let mut entropy = File::open("/dev/urandom")?;
    load_or_create_from(path, &mut entropy)
}

fn read_existing(path: &Path) -> io::Result<InstallationId> {
    let raw = std::fs::read_to_string(path)?;
    let text = raw.trim();
    if text.len() != 16 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "installation identity must be exactly 16 hexadecimal digits",
        ));
    }

    let value = u64::from_str_radix(text, 16).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid installation identity: {error}"),
        )
    })?;
    let id = InstallationId(value);
    if !id.is_known() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "zero is reserved for unknown installation identity",
        ));
    }
    Ok(id)
}

fn load_or_create_from(path: &Path, entropy: &mut impl Read) -> io::Result<InstallationId> {
    if path.exists() {
        return read_existing(path);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut bytes = [0_u8; 8];
    entropy.read_exact(&mut bytes)?;
    let id = InstallationId(u64::from_le_bytes(bytes));
    if !id.is_known() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "entropy produced the reserved unknown installation identity",
        ));
    }

    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);

    match options.open(path) {
        Ok(mut file) => {
            writeln!(file, "{:016x}", id.0)?;
            file.sync_all()?;
            Ok(id)
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => read_existing(path),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn creates_once_and_reloads_the_same_nonzero_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("installation_id");
        let mut first_entropy = &0x1020_3040_5060_7080_u64.to_le_bytes()[..];
        let first = load_or_create_from(&path, &mut first_entropy).unwrap();
        let mut ignored_entropy = &0x8877_6655_4433_2211_u64.to_le_bytes()[..];
        let second = load_or_create_from(&path, &mut ignored_entropy).unwrap();

        assert_eq!(first, InstallationId(0x1020_3040_5060_7080));
        assert_eq!(second, first);
        #[cfg(unix)]
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn zero_entropy_is_rejected_instead_of_creating_portable_authority() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("installation_id");
        let mut zero = &[0_u8; 8][..];

        assert_eq!(
            load_or_create_from(&path, &mut zero).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        assert!(!path.exists());
    }

    #[test]
    fn malformed_existing_identity_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("installation_id");
        std::fs::write(&path, "not-an-id\n").unwrap();
        let mut entropy = &0x1020_3040_5060_7080_u64.to_le_bytes()[..];

        assert_eq!(
            load_or_create_from(&path, &mut entropy).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }
}
