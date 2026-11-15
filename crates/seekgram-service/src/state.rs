//! The state every part of the service shares.
//!
//! The status types live in `seekgram_core::status` so the panel can use them without
//! this crate; each subsystem updates its section through `Shared::update`.

use std::sync::{Arc, Mutex, RwLock};

use tokio::sync::{mpsc, oneshot, watch};

pub use seekgram_core::status::*;

use crate::config::{Secrets, Settings};

use super::telegram::TelegramService;

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
