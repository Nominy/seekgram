//! Settings (plain JSON) and secrets (encrypted with Windows DPAPI for the current user).

use std::{fs, io::Write, path::Path};

use serde::{Deserialize, Serialize};

use crate::paths;

/// Default loopback port for the control API and the MCP endpoint.
pub const DEFAULT_PORT: u16 = 28741;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Loopback port of the control API and `/mcp`. Applied at the next service start.
    pub port: u16,
    /// Whether the indexer should run whenever it can.
    pub indexing: bool,
    /// Model worker processes; each holds its own copy of the model. 0 runs one.
    pub workers: u8,
    /// Media messages downloaded and prepared at once.
    pub concurrency: u8,
    /// Chat where `?` searches are answered: `None` is Saved Messages, otherwise a marked chat ID.
    pub search_chat: Option<i64>,
    /// The private group last used for searching. Kept while searches go to Saved Messages,
    /// so switching back reuses it instead of creating another group.
    pub search_group: Option<i64>,
    /// Start the service when the user logs in to Windows.
    pub autostart: bool,
    /// Serve the MCP endpoint for AI agents. Off by default: most users never need it.
    pub mcp: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { port: DEFAULT_PORT, indexing: true, workers: 0, concurrency: 6, search_chat: None, search_group: None, autostart: true, mcp: false }
    }
}

impl Settings {
    /// Missing or unreadable settings are the defaults; a damaged file is not an error
    /// worth refusing to start over.
    pub fn load() -> Self {
        fs::read(paths::settings_file()).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default()
    }
    pub fn save(&self) -> Result<(), String> {
        write_atomic(&paths::settings_file(), &serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?)
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Secrets {
    /// From my.telegram.org; identifies this install to Telegram.
    pub api_id: Option<i32>,
    pub api_hash: Option<String>,
    /// Bearer token for the control API and MCP. Created on first use.
    pub token: String,
}

impl Secrets {
    /// Loads the secrets, creating the access token on first use.
    pub fn load() -> Result<Self, String> {
        let path = paths::secrets_file();
        let mut secrets: Secrets = match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&unprotect(&bytes)?).map_err(|e| format!("secrets are unreadable: {e}"))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Secrets::default(),
            Err(error) => return Err(format!("cannot read secrets: {error}")),
        };
        if secrets.token.is_empty() {
            secrets.token = format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple());
            secrets.save()?;
        }
        Ok(secrets)
    }
    pub fn save(&self) -> Result<(), String> {
        let plain = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        write_atomic(&paths::secrets_file(), &protect(&plain)?)
    }
}

/// Writes through a temporary file so a crash never leaves a half-written file behind.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let directory = path.parent().ok_or("path has no parent")?;
    fs::create_dir_all(directory).map_err(|e| format!("cannot create {}: {e}", directory.display()))?;
    let temporary = path.with_extension("tmp");
    let mut file = fs::File::create(&temporary).map_err(|e| format!("cannot write {}: {e}", temporary.display()))?;
    file.write_all(bytes).and_then(|_| file.sync_all()).map_err(|e| format!("cannot write {}: {e}", temporary.display()))?;
    drop(file);
    fs::rename(&temporary, path).map_err(|e| format!("cannot replace {}: {e}", path.display()))
}

#[cfg(windows)]
fn dpapi(input: &[u8], protect: bool) -> Result<Vec<u8>, String> {
    use windows::Win32::{Foundation::{HLOCAL, LocalFree}, Security::Cryptography::{CRYPT_INTEGER_BLOB, CryptProtectData, CryptUnprotectData}};
    let source = CRYPT_INTEGER_BLOB { cbData: input.len() as u32, pbData: input.as_ptr() as *mut u8 };
    let mut output = CRYPT_INTEGER_BLOB::default();
    // SAFETY: `source` points at `input` for the duration of the call; on success the
    // system allocates `output`, which is copied and then released with LocalFree.
    unsafe {
        let result = if protect { CryptProtectData(&source, None, None, None, None, 0, &mut output) }
            else { CryptUnprotectData(&source, None, None, None, None, 0, &mut output) };
        result.map_err(|e| format!("Windows could not {} the secrets: {e}", if protect { "encrypt" } else { "decrypt" }))?;
        let bytes = std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec();
        let _ = LocalFree(Some(HLOCAL(output.pbData as *mut _)));
        Ok(bytes)
    }
}
#[cfg(windows)]
fn protect(plain: &[u8]) -> Result<Vec<u8>, String> { dpapi(plain, true) }
#[cfg(windows)]
fn unprotect(sealed: &[u8]) -> Result<Vec<u8>, String> { dpapi(sealed, false) }

#[cfg(not(windows))]
fn protect(_: &[u8]) -> Result<Vec<u8>, String> { Err("secret storage is only implemented for Windows".into()) }
#[cfg(not(windows))]
fn unprotect(_: &[u8]) -> Result<Vec<u8>, String> { Err("secret storage is only implemented for Windows".into()) }
