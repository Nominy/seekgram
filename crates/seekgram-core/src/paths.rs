//! Every file Seekgram owns lives under one data directory, so an install can be
//! inspected, backed up or removed as a unit. Nothing is written next to the executable.

use std::path::PathBuf;

#[cfg(not(target_os = "linux"))]
/// `%LOCALAPPDATA%\Seekgram`, or `SEEKGRAM_DATA_DIR` for a second, isolated instance.
pub fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SEEKGRAM_DATA_DIR") { return PathBuf::from(dir); }
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    base.join("Seekgram")
}

/// Linux data directory. An empty path signals that neither XDG_DATA_HOME nor
/// an absolute HOME was available. Always validate it before accessing data.
#[cfg(target_os = "linux")]
pub fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SEEKGRAM_DATA_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(base) = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).filter(|p| p.is_absolute()) {
        return base.join("seekgram");
    }
    std::env::var_os("HOME").map(PathBuf::from).filter(|p| p.is_absolute())
        .map(|home| home.join(".local/share/seekgram")).unwrap_or_default()
}

/// Validates storage before starting the service or creating secrets. The umask also
/// covers files created by SQLite and other dependencies that do not specify modes.
#[cfg(target_os = "linux")]
pub fn prepare_data_dir() -> Result<(), String> {
    // SAFETY: no pointer arguments. All Seekgram data directories are owner-only.
    unsafe { libc::umask(0o077); }
    private_data_dir(&data_dir(), true)
}

/// Checks the storage tree before existing files are opened by the service.
/// SQLite and other libraries may follow existing links when opening their files.
#[cfg(target_os = "linux")]
pub fn verify_data_tree() -> Result<(), String> {
    private_data_dir(&data_dir(), false)?;
    fn walk(dir: &std::path::Path) -> Result<(), String> {
        for entry in std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))? {
            let path = entry.map_err(|e| format!("cannot read {}: {e}", dir.display()))?.path();
            let meta = std::fs::symlink_metadata(&path).map_err(|e| format!("cannot inspect {}: {e}", path.display()))?;
            if path == control_socket() { verify_control_socket(&path, &meta)?; }
            else { verify_owned_node(&path, &meta)?; }
            if meta.is_dir() { walk(&path)?; }
        }
        Ok(())
    }
    walk(&data_dir())
}

#[cfg(target_os = "linux")]
pub(crate) fn private_data_dir(path: &std::path::Path, create: bool) -> Result<(), String> {
    use std::{fs, path::Component, os::unix::fs::{DirBuilderExt, MetadataExt}};

    if !path.is_absolute() || path.parent().is_none() || path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err("Set SEEKGRAM_DATA_DIR to a private absolute directory, or set XDG_DATA_HOME or HOME.".into());
    }
    // SAFETY: no arguments, returns the effective UID.
    let uid = unsafe { libc::geteuid() };
    let mut current = PathBuf::new();
    let mut components = path.components().peekable();
    while let Some(component) = components.next() {
        current.push(component);
        let leaf = components.peek().is_none();
        let meta = match fs::symlink_metadata(&current) {
            Ok(meta) => meta,
            Err(e) if create && e.kind() == std::io::ErrorKind::NotFound => {
                match fs::DirBuilder::new().mode(0o700).create(&current) {
                    Ok(()) => (),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
                    Err(e) => return Err(format!("cannot create {}: {e}", current.display())),
                }
                fs::symlink_metadata(&current).map_err(|e| format!("cannot inspect {}: {e}", current.display()))?
            }
            Err(e) => return Err(format!("cannot inspect {}: {e}", current.display())),
        };
        let mode = meta.mode();
        // A root-owned sticky directory (e.g., /tmp for an explicit test override)
        // prevents other users from replacing entries they do not own.
        let root_sticky = meta.uid() == 0 && mode & 0o1000 != 0;
        let safe = meta.is_dir() && if leaf {
            meta.uid() == uid && mode & 0o7777 == 0o700
        } else {
            (meta.uid() == 0 || meta.uid() == uid) && (mode & 0o022 == 0 || root_sticky)
        };
        if !safe {
            return Err(format!("Unsafe Seekgram data path {}: use real, trusted directories and mode 0700 on the data directory.", current.display()));
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
pub(crate) fn verify_owned_node(path: &std::path::Path, meta: &std::fs::Metadata) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    // SAFETY: no arguments, returns the effective UID.
    let uid = unsafe { libc::geteuid() };
    let safe = meta.uid() == uid && if meta.is_file() {
        meta.nlink() == 1 && meta.mode() & 0o7777 == 0o600
    } else {
        meta.is_dir() && meta.mode() & 0o7777 == 0o700
    };
    if safe { Ok(()) } else {
        Err(format!("Unsafe Seekgram storage entry {}: use owner-only regular files (0600) and directories (0700), with no links.", path.display()))
    }
}

/// The private Linux control endpoint, never opened by other OS users.
#[cfg(target_os = "linux")]
pub fn control_socket() -> PathBuf { data_dir().join("control.sock") }

#[cfg(target_os = "linux")]
pub(crate) fn verify_control_socket(path: &std::path::Path, meta: &std::fs::Metadata) -> Result<(), String> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};
    // SAFETY: no arguments, returns the effective UID.
    let uid = unsafe { libc::geteuid() };
    if meta.file_type().is_socket() && meta.uid() == uid && meta.nlink() == 1
        && matches!(meta.mode() & 0o7777, 0o600 | 0o700)
    {
        Ok(())
    } else {
        Err(format!("Unsafe Seekgram control socket {}.", path.display()))
    }
}

/// Settings the user can change; plain JSON.
pub fn settings_file() -> PathBuf { data_dir().join("settings.json") }
/// Telegram API key and local token: DPAPI-encrypted on Windows, unencrypted in a mode 0600 file on Linux.
pub fn secrets_file() -> PathBuf { data_dir().join("secrets.bin") }
/// The Telegram login. Whoever holds this file is logged in as the user.
pub fn session_db() -> PathBuf { data_dir().join("telegram.sqlite3") }
/// The login as grammers 0.7 saved it, read once to fill a new `session_db`.
pub fn legacy_session_file() -> PathBuf { data_dir().join("telegram.session") }
/// Held by the running service; a second service refuses to start.
pub fn lock_file() -> PathBuf { data_dir().join("service.lock") }
/// The embedding index: `index.sqlite3`.
pub fn index_dir() -> PathBuf { data_dir().join("index") }
/// Media fetched to show or to search by; size-capped, safe to delete.
pub fn cache_dir() -> PathBuf { data_dir().join("cache") }
pub fn logs_dir() -> PathBuf { data_dir().join("logs") }
/// Downloaded helper programs: `ffmpeg.exe`, `ffprobe.exe`.
pub fn tools_dir() -> PathBuf { data_dir().join("tools") }
/// ONNX Runtime: `onnxruntime.dll` and the `DirectML.dll` it must load.
pub fn runtime_dir() -> PathBuf { data_dir().join("runtime") }
/// The embedding model: `tokenizer.json`, the graphs and their weights (about 1.5 GB).
pub fn model_dir() -> PathBuf { data_dir().join("model") }
/// Scratch space for media being indexed; emptied at service start.
pub fn temp_dir() -> PathBuf { data_dir().join("tmp") }
/// When the last check for a new version succeeded (`crates/seekgram-service/src/update.rs`).
pub fn update_file() -> PathBuf { data_dir().join("update.json") }
/// What Seekgram Setup's install, update and uninstall steps did (`crates/seekgram-service/src/install.rs`).
pub fn install_log() -> PathBuf { logs_dir().join("install.log") }

/// Paths of the prepared tools. Only valid once `Status.setup.state == Ready`.
pub fn ffmpeg() -> PathBuf {
    tools_dir().join("ffmpeg.exe")
}

pub fn ffprobe() -> PathBuf {
    tools_dir().join("ffprobe.exe")
}


#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::{DirBuilderExt, PermissionsExt, symlink}};

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("sg-{}", uuid::Uuid::new_v4().simple()));
            fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
    }

    #[test]
    fn creates_private_data_and_rejects_insecure_directory() {
        let parent = Temp::new();
        let data = parent.0.join("data");
        private_data_dir(&data, true).unwrap();
        assert_eq!(fs::metadata(&data).unwrap().permissions().mode() & 0o777, 0o700);
        fs::set_permissions(&data, fs::Permissions::from_mode(0o750)).unwrap();
        assert!(private_data_dir(&data, false).is_err());
        assert!(private_data_dir(std::path::Path::new("/"), false).is_err());
        assert!(private_data_dir(std::path::Path::new("relative"), false).is_err());
    }

    #[test]
    fn rejects_symlinks_and_hardlinked_or_world_readable_files() {
        let parent = Temp::new();
        let data = parent.0.join("data");
        let link = parent.0.join("link");
        private_data_dir(&data, true).unwrap();
        symlink(&data, &link).unwrap();
        assert!(private_data_dir(&link, false).is_err());
        assert!(private_data_dir(&link.join("new"), true).is_err());
        assert!(!data.join("new").exists());

        let claim = data.join("test");
        fs::write(&claim, b"fixture").unwrap();
        fs::set_permissions(&claim, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(verify_owned_node(&claim, &fs::symlink_metadata(&claim).unwrap()).is_err());
        fs::set_permissions(&claim, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(verify_owned_node(&claim, &fs::symlink_metadata(&claim).unwrap()).is_ok());
        fs::hard_link(&claim, data.join("hard")).unwrap();
        assert!(verify_owned_node(&claim, &fs::symlink_metadata(&claim).unwrap()).is_err());
    }

    #[test]
    fn rejects_non_socket_at_control_path() {
        let parent = Temp::new();
        let socket = parent.0.join("control.sock");
        fs::write(&socket, b"not a socket").unwrap();
        fs::set_permissions(&socket, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(verify_control_socket(&socket, &fs::symlink_metadata(&socket).unwrap()).is_err());
    }
}
