//! The state every part of the service shares, and the status document the app shows.
//!
//! `Status` is the single source of truth for the control panel: each subsystem owns
//! one section and updates it through `Shared::update`. It is also what
//! `GET /api/status` returns, so every field here is part of the app's contract.

use std::sync::{Arc, Mutex, RwLock};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot, watch};

use crate::config::{Secrets, Settings};

use super::telegram::TelegramService;

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

/// The step of preparing the model worker that is running or failed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SetupStep {
    #[default]
    Checking,
    /// Downloading the `uv` Python manager.
    Uv,
    /// Downloading ffmpeg and ffprobe.
    Ffmpeg,
    /// Installing Python and the model's packages (PyTorch with CUDA is about 3 GB).
    Packages,
    /// Downloading the embedding model (about 1.5 GB).
    Model,
    /// Loading the model and embedding a test image, clip and sound.
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
    /// One line about what is happening right now, for the user.
    pub detail: String,
    /// Download progress of the current step, when it is a download of known size.
    pub bytes_done: Option<u64>,
    pub bytes_total: Option<u64>,
    pub error: Option<String>,
    /// The NVIDIA GPU found, if any. Without one indexing is refused.
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
    pub settings: Settings,
    /// e.g. `http://127.0.0.1:28741/mcp`
    pub mcp_url: String,
    pub data_dir: String,
}

/// A login step submitted through the control API.
pub enum LoginCommand {
    ApiKey { api_id: i32, api_hash: String },
    Phone(String),
    Code(String),
    Password(String),
    /// Ends the Telegram session and forgets it locally. The index is kept.
    Logout,
}

/// A login step and the signal that Telegram has answered it. The outcome is read
/// from `Status.account` afterwards.
pub struct LoginRequest {
    pub command: LoginCommand,
    pub done: oneshot::Sender<()>,
}

pub struct Shared {
    status: RwLock<Status>,
    pub secrets: Mutex<Secrets>,
    /// The logged-in Telegram client; `None` until `account.step` is `ready`.
    pub telegram: tokio::sync::RwLock<Option<Arc<TelegramService>>>,
    pub login: mpsc::Sender<LoginRequest>,
    /// Flips to `true` once; every task then finishes its current unit and returns.
    pub shutdown: watch::Sender<bool>,
    /// Bumped whenever settings, login or setup change, so waiting tasks re-evaluate.
    pub changed: watch::Sender<u64>,
}

impl Shared {
    pub fn new(settings: Settings, secrets: Secrets, login: mpsc::Sender<LoginRequest>) -> Arc<Self> {
        let status = Status {
            version: env!("CARGO_PKG_VERSION").to_owned(),
            pid: std::process::id(),
            mcp_url: format!("http://127.0.0.1:{}/mcp", settings.port),
            data_dir: crate::paths::data_dir().to_string_lossy().into_owned(),
            settings,
            ..Status::default()
        };
        Arc::new(Self {
            status: RwLock::new(status),
            secrets: Mutex::new(secrets),
            telegram: tokio::sync::RwLock::new(None),
            login,
            shutdown: watch::channel(false).0,
            changed: watch::channel(0).0,
        })
    }

    /// A copy of the whole status document.
    pub fn status(&self) -> Status { self.status.read().unwrap_or_else(|e| e.into_inner()).clone() }

    /// Reads one thing out of the status without copying all of it.
    pub fn read<T>(&self, read: impl FnOnce(&Status) -> T) -> T { read(&self.status.read().unwrap_or_else(|e| e.into_inner())) }

    /// Changes the status. Keep the closure short: it runs under the lock.
    pub fn update(&self, change: impl FnOnce(&mut Status)) { change(&mut self.status.write().unwrap_or_else(|e| e.into_inner())); }

    /// Tells waiting tasks that login, setup or settings changed.
    pub fn notify(&self) { self.changed.send_modify(|generation| *generation += 1); }

    pub fn settings(&self) -> Settings { self.read(|status| status.settings.clone()) }

    /// Applies and persists a settings change.
    pub fn set_settings(&self, change: impl FnOnce(&mut Settings)) -> Result<(), String> {
        let mut settings = self.settings();
        change(&mut settings);
        settings.save()?;
        self.update(|status| status.settings = settings);
        self.notify();
        Ok(())
    }

    pub fn stopping(&self) -> bool { *self.shutdown.borrow() }

    /// Resolves when the service is shutting down.
    pub async fn stopped(&self) {
        let mut shutdown = self.shutdown.subscribe();
        while !*shutdown.borrow_and_update() { if shutdown.changed().await.is_err() { return; } }
    }

    /// The logged-in client, or the reason there is none.
    pub async fn telegram(&self) -> Result<Arc<TelegramService>, String> {
        self.telegram.read().await.clone().ok_or_else(|| "Telegram is not logged in yet; open Seekgram to log in".to_owned())
    }
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
