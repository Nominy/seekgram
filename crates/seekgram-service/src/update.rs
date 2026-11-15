//! Automatic updates for copies installed with Seekgram Setup (Velopack). Owns `Status.update`.
//!
//! Two minutes after the service starts, and then every six hours (counted from the last
//! successful check, which is remembered across restarts), the service asks the GitHub
//! releases for a newer version while `settings.check_updates` is on. A newer version is
//! downloaded in the background; then open panels are closed, the service shuts down
//! the usual way, and `apply_pending` hands over to Velopack's updater, which puts the new
//! version in place and starts it again as `seekgram service`.
//!
//! `SEEKGRAM_UPDATE_URL` replaces the GitHub releases with another Velopack feed, a local
//! folder or a URL, and checks on every start regardless of the last check; it is for testing.

use std::{
    fs,
    sync::{Arc, Mutex},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use velopack::{sources, UpdateCheck, UpdateManager, VelopackAsset};

use crate::{config::write_atomic, paths};

use super::state::{unix_now, Shared, Update, UpdateState};

const RELEASES: &str = "https://github.com/Nominy/seekgram";
const FIRST_CHECK_AFTER: u64 = 2 * 60;
const CHECK_EVERY: u64 = 6 * 60 * 60;
/// After a failed check or download (usually no network yet), try again sooner.
const RETRY_AFTER: u64 = 60 * 60;
/// Waits are cut into slices so a PC waking from sleep checks on time.
const LONGEST_NAP: u64 = 10 * 60;

/// A downloaded update, handed from `run` to `apply_pending` once everything has stopped.
static PENDING: Mutex<Option<(UpdateManager, VelopackAsset)>> = Mutex::new(None);

/// `update.json` in the data folder: when the last check succeeded.
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Checked {
    checked_at: u64,
}

fn last_check() -> u64 {
    fs::read(paths::update_file()).ok().and_then(|bytes| serde_json::from_slice::<Checked>(&bytes).ok()).map_or(0, |checked| checked.checked_at)
}

fn remember_check() {
    if let Ok(bytes) = serde_json::to_vec(&Checked { checked_at: unix_now() }) {
        if let Err(error) = write_atomic(&paths::update_file(), &bytes) { eprintln!("Update: {error}"); }
    }
}

fn set(shared: &Shared, change: impl FnOnce(&mut Update)) {
    shared.update(|status| change(&mut status.update));
}

/// The update manager for this copy, or why it cannot update itself.
fn manager(feed: Option<&str>) -> Result<UpdateManager, String> {
    let made = match feed {
        Some(feed) => UpdateManager::new(sources::AutoSource::new(feed), None, None),
        // Every 0.x release is a pre-release on GitHub, so pre-releases must count.
        None => UpdateManager::new(sources::GithubSource::new(RELEASES, None, true), None, None),
    };
    made.map_err(|error| {
        eprintln!("Update: not checking for updates: {error}");
        match error {
            velopack::Error::NotInstalled(_) => "This copy of Seekgram was not installed with Seekgram Setup, so it does not update itself.".to_owned(),
            _ => "Seekgram cannot update itself here because its installation could not be read.".to_owned(),
        }
    })
}

/// A Velopack error, in words for the user.
fn describe(error: &velopack::Error, checking: bool) -> String {
    match error {
        velopack::Error::Network(_) if checking => "Seekgram could not reach GitHub to check for a new version. It tries again in an hour.".to_owned(),
        velopack::Error::Network(_) => "The download of the new version was interrupted. Seekgram tries again in an hour.".to_owned(),
        velopack::Error::ChecksumInvalid(..) | velopack::Error::SizeInvalid(..) => "The new version arrived damaged. Seekgram downloads it again in an hour.".to_owned(),
        _ if checking => "Checking for a new version failed. Seekgram tries again in an hour.".to_owned(),
        _ => "The new version could not be downloaded. Seekgram tries again in an hour.".to_owned(),
    }
}

/// Blocking: asks the feed for a newer version and downloads it. `None` if there is none.
fn check_and_download(shared: &Arc<Shared>, manager: &UpdateManager) -> Result<Option<VelopackAsset>, String> {
    set(shared, |update| *update = Update { state: UpdateState::Checking, ..Update::default() });
    let found = manager.check_for_updates().map_err(|error| {
        eprintln!("Update: the check failed: {error}");
        describe(&error, true)
    })?;
    remember_check();
    let UpdateCheck::UpdateAvailable(info) = found else {
        eprintln!("Update: Seekgram {} is the latest version", env!("CARGO_PKG_VERSION"));
        set(shared, |update| *update = Update::default());
        return Ok(None);
    };
    let version = info.TargetFullRelease.Version.clone();
    eprintln!("Update: downloading Seekgram {version}");
    set(shared, |update| *update = Update { state: UpdateState::Downloading, version: Some(version.clone()), percent: Some(0), ..Update::default() });
    let (sender, receiver) = std::sync::mpsc::channel::<i16>();
    let watcher = shared.clone();
    let progress = std::thread::Builder::new().name("update-progress".into()).spawn(move || {
        for percent in receiver {
            let percent = u8::try_from(percent.clamp(0, 100)).unwrap_or(0);
            set(&watcher, |update| update.percent = Some(percent));
        }
    });
    let downloaded = manager.download_updates(&info, Some(sender));
    if let Ok(progress) = progress { let _ = progress.join(); }
    downloaded.map_err(|error| {
        eprintln!("Update: the download failed: {error}");
        describe(&error, false)
    })?;
    eprintln!("Update: Seekgram {version} is downloaded");
    Ok(Some(info.TargetFullRelease))
}

/// Runs until shutdown, or until an update is downloaded: then it asks the service to stop.
pub async fn run(shared: Arc<Shared>) {
    let feed = std::env::var("SEEKGRAM_UPDATE_URL").ok().filter(|feed| !feed.trim().is_empty());
    let manager = match manager(feed.as_deref()) {
        Ok(manager) => manager,
        Err(reason) => {
            set(&shared, |update| update.reason = Some(reason));
            return;
        }
    };
    if let Some(feed) = &feed { eprintln!("Update: using the feed {feed} instead of GitHub"); }
    let earliest = unix_now() + FIRST_CHECK_AFTER;
    let mut next = if feed.is_some() { earliest } else { earliest.max(last_check() + CHECK_EVERY) };
    let mut changed = shared.changed.subscribe();
    loop {
        let enabled = shared.settings().check_updates;
        let wait = next.saturating_sub(unix_now());
        if !enabled || wait > 0 {
            let nap = Duration::from_secs(if enabled { wait.min(LONGEST_NAP) } else { LONGEST_NAP });
            tokio::select! {
                _ = shared.stopped() => return,
                _ = changed.changed() => {}
                _ = tokio::time::sleep(nap) => {}
            }
            continue;
        }
        let attempt = {
            let (shared, manager) = (shared.clone(), manager.clone());
            tokio::task::spawn_blocking(move || check_and_download(&shared, &manager))
        };
        // A check or download cannot be cancelled; shutdown leaves it behind and Velopack
        // throws the partial file away next time.
        let outcome = tokio::select! {
            _ = shared.stopped() => return,
            outcome = attempt => outcome.unwrap_or_else(|_| Err("Checking for a new version failed. Seekgram tries again in an hour.".to_owned())),
        };
        match outcome {
            Ok(None) => next = unix_now() + CHECK_EVERY,
            Ok(Some(asset)) if shared.settings().check_updates => {
                let version = asset.Version.clone();
                eprintln!("Update: stopping to install Seekgram {version}");
                set(&shared, |update| *update = Update { state: UpdateState::Ready, version: Some(version), ..Update::default() });
                *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((manager, asset));
                // Panels would only show a stopped service; the new version opens none by itself.
                super::tray::close_panels();
                shared.shutdown.send_replace(true);
                return;
            }
            // Switched off while downloading: keep the download for when it is switched on again.
            Ok(Some(_)) => {
                set(&shared, |update| *update = Update::default());
                next = unix_now();
            }
            Err(error) => {
                set(&shared, |update| *update = Update { state: UpdateState::Failed, error: Some(error), ..Update::default() });
                next = unix_now() + RETRY_AFTER;
            }
        }
    }
}

/// Called by the service as its last act, after everything has stopped and the lock is
/// released. If an update is waiting, starts Velopack's updater, which waits for this
/// process to exit, installs the new version and starts it again as `seekgram service`.
pub fn apply_pending() {
    let Some((manager, asset)) = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    // The updater must outlive this process, so it must not join the job that ends the
    // service's children with it.
    super::release_children();
    match manager.wait_exit_then_apply_updates(&asset, true, true, ["service"]) {
        Ok(()) => eprintln!("Update: the updater installs Seekgram {} now", asset.Version),
        Err(error) => eprintln!("Update: the updater could not be started: {error}"),
    }
}

