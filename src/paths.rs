//! Every file Seekgram owns lives under one data directory, so an install can be
//! inspected, backed up or removed as a unit. Nothing is written next to the executable.

use std::path::PathBuf;

/// `%LOCALAPPDATA%\Seekgram`, or `SEEKGRAM_DATA_DIR` for a second, isolated instance.
pub fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SEEKGRAM_DATA_DIR") { return PathBuf::from(dir); }
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    base.join("Seekgram")
}

/// Settings the user can change; plain JSON.
pub fn settings_file() -> PathBuf { data_dir().join("settings.json") }
/// Telegram API key and the local access token, encrypted for the Windows user.
pub fn secrets_file() -> PathBuf { data_dir().join("secrets.bin") }
/// The Telegram login. Whoever holds this file is logged in as the user.
pub fn session_file() -> PathBuf { data_dir().join("telegram.session") }
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
