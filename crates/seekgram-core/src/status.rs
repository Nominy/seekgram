//! The status document the service publishes on `GET /api/status` and the panel shows.
//!
//! Every field here is part of the contract between the service and the app; the
//! service owns one section per subsystem and updates it through its `Shared` state.

use serde::{Deserialize, Serialize};

use crate::config::Settings;

/// Where the Telegram login stands. Each step names what the user has to supply next.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountStep {
    /// No API key stored yet.
    #[default]
    ApiKey,
    /// Talking to Telegram; nothing to ask.
    Connecting,
    Phone,
    Code,
    /// Two-step verification password.
    Password,
    Ready,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Account {
    pub step: AccountStep,
    /// Why the last login action failed, in words for the user.
    pub error: Option<String>,
    /// Display name and ID once `step` is `ready`.
    pub name: Option<String>,
    pub id: Option<i64>,
    /// The hint Telegram gives for the two-step password, when it asks for one.
    pub password_hint: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SetupState {
    /// Not checked yet.
    #[default]
    Pending,
    Running,
    Ready,
    Failed,
}

/// The step of preparing the model that is running or failed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SetupStep {
    /// Looking for the graphics card.
    #[default]
    Checking,
    /// Downloading ffmpeg and ffprobe.
    Ffmpeg,
    /// Downloading ONNX Runtime and DirectML (15 MB).
    Runtime,
    /// Downloading the embedding model (about 1.5 GB; 2.9 GB more if full precision is needed).
    Model,
    /// Loading the model and embedding a test text, picture, sound and video clip.
    SelfTest,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Gpu {
    pub name: String,
    pub vram_mb: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Setup {
    pub state: SetupState,
    pub step: SetupStep,
    /// One line about what is happening right now, for the user. Once ready: the device and
    /// engine the model runs on, e.g. "Model ready on NVIDIA GeForce RTX 4070 (DirectML)".
    pub detail: String,
    /// Download progress of the current step, when it is a download of known size.
    pub bytes_done: Option<u64>,
    pub bytes_total: Option<u64>,
    pub error: Option<String>,
    /// The graphics card DirectML uses, if any. Without one the model runs on the processor.
    pub gpu: Option<Gpu>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexerState {
    /// Not running: switched off, or waiting for login or setup.
    #[default]
    Off,
    Starting,
    Running,
    /// Caught up; waiting for the next pass.
    Idle,
    /// Telegram asked for a pause.
    FloodWait,
    Stopping,
    Failed,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CurrentChat {
    pub id: i64,
    pub title: String,
    /// The newest message already scanned, and the chat's newest message.
    pub message_id: i32,
    pub top_message_id: i32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Indexer {
    pub state: IndexerState,
    /// Why it is off or failed, in words for the user (e.g. "waiting for login").
    pub reason: Option<String>,
    pub chats_done: u32,
    pub chats_total: u32,
    pub chat: Option<CurrentChat>,
    /// Since this service started.
    pub messages_scanned: u64,
    pub items_added: u64,
    pub errors: u64,
    pub bytes_downloaded: u64,
    /// Media messages being downloaded or embedded right now.
    pub in_flight: u32,
    /// Items added per minute over the last few minutes.
    pub items_per_minute: f64,
    /// Seconds left of a pause Telegram asked for.
    pub flood_wait_seconds: Option<u64>,
    /// Model workers that loaded successfully.
    pub workers: u32,
    /// Unix seconds.
    pub started_at: Option<u64>,
    pub last_activity_at: Option<u64>,
}

/// What the index holds, read from the database every few seconds.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexCounts {
    pub images: u64,
    pub video_segments: u64,
    pub audio_segments: u64,
    pub chats: u64,
    pub failures: u64,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchChat {
    /// `saved` or `group`.
    pub kind: String,
    pub id: Option<i64>,
    pub title: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateState {
    /// Nothing happening: between checks, switched off, or not possible for this copy (`reason`).
    #[default]
    Idle,
    Checking,
    Downloading,
    /// Downloaded; the service is stopping so the new version can be put in place.
    Ready,
    Failed,
}

/// Automatic updates (`crates/seekgram-service/src/update.rs`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Update {
    pub state: UpdateState,
    /// The version being downloaded or installed.
    pub version: Option<String>,
    /// Download progress, 0 to 100, while `downloading`.
    pub percent: Option<u8>,
    /// Why the last check or download failed, in words for the user.
    pub error: Option<String>,
    /// Why this copy does not update itself (not installed with Seekgram Setup), if so.
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Status {
    pub version: String,
    pub pid: u32,
    pub account: Account,
    pub setup: Setup,
    pub indexer: Indexer,
    pub index: IndexCounts,
    pub search_chat: SearchChat,
    pub update: Update,
    pub settings: Settings,
    /// e.g. `http://127.0.0.1:28741/mcp`
    pub mcp_url: String,
    pub data_dir: String,
}

/// One line for the panel while something is happening; `None` when nothing is.
pub fn describe_status(update: &Update) -> Option<String> {
    let version = update.version.as_deref().unwrap_or("");
    match update.state {
        UpdateState::Idle => None,
        UpdateState::Checking => Some("Checking for a new version".to_owned()),
        UpdateState::Downloading => Some(match update.percent {
            Some(percent) => format!("Downloading Seekgram {version} ({percent}%)"),
            None => format!("Downloading Seekgram {version}"),
        }),
        UpdateState::Ready => Some(format!("Installing Seekgram {version}; it restarts in a moment")),
        UpdateState::Failed => Some(update.error.clone().unwrap_or_else(|| "The update failed. Seekgram tries again later.".to_owned())),
    }
}
