//! What Seekgram Setup (Velopack) runs this program for. Setup installs it to
//! `%LOCALAPPDATA%\Seekgram.App` (`current\seekgram.exe`, `Update.exe`) and calls it back:
//!
//! - after installing and after each update (`--veloapp-install`, `--veloapp-updated`):
//!   the start at logon is registered again, unless the user switched it off;
//! - before uninstalling (`--veloapp-uninstall`): the start at logon is removed. The data
//!   folder is left alone.
//!
//! Velopack's uninstaller ends every process running from the install folder before it
//! calls the uninstall hook, so the service cannot be asked to stop there; `install.ps1
//! -Uninstall` asks it first. Ended mid-write, the index is still consistent (SQLite).
//!
//! After an interactive install, Setup opens Seekgram (`VELOPACK_FIRSTRUN`): the panel,
//! which starts the service. `SEEKGRAM_NO_LAUNCH` skips that. A silent install opens nothing.
//!
//! The install folder must not be the data folder (`%LOCALAPPDATA%\Seekgram`): Velopack
//! deletes its install folder on uninstall, and updates must not touch the login or index.

use std::io::Write;

use velopack::VelopackApp;

use crate::{autostart, config::Settings, paths};

/// Appends to `logs/install.log`; Setup shows nothing a hook prints.
fn log(line: &str) {
    let path = paths::install_log();
    if let Some(folder) = path.parent() { let _ = std::fs::create_dir_all(folder); }
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(file, "{} {line}", crate::timestamp());
    }
}

/// Handles Setup's callbacks, which exit the process; returns for every other start.
pub fn handle_setup_events() {
    VelopackApp::build()
        // The service installs updates itself once it has stopped cleanly; a panel or a model
        // worker starting meanwhile must not do it behind the service's back.
        .set_auto_apply_on_startup(false)
        .on_after_install_fast_callback(|version| {
            log(&format!("Installed Seekgram {version}"));
            register_autostart();
        })
        .on_after_update_fast_callback(|version| {
            log(&format!("Updated to Seekgram {version}"));
            register_autostart();
        })
        .on_before_uninstall_fast_callback(|version| {
            log(&format!("Uninstalling Seekgram {version}"));
            match autostart::remove_if_ours() {
                Ok(()) => log("Start at logon removed"),
                Err(error) => log(&error),
            }
            log(&format!("The data folder {} is kept", paths::data_dir().display()));
        })
        .on_first_run(|_| {
            if std::env::var_os("SEEKGRAM_NO_LAUNCH").is_some_and(|value| !value.is_empty()) {
                log("Installed without opening Seekgram (SEEKGRAM_NO_LAUNCH)");
                std::process::exit(0);
            }
        })
        .run();
}

/// Registers the start at logon for this install, keeping the user's choice. A test copy
/// with its own data folder leaves the entry alone: it would replace the real one.
fn register_autostart() {
    let command = autostart::command().unwrap_or_default();
    if std::env::var_os("SEEKGRAM_DATA_DIR").is_some() {
        log(&format!("Start at logon left alone because SEEKGRAM_DATA_DIR is set; it would be: {command}"));
    } else if !Settings::load().autostart {
        log("Start at logon stays off, as chosen in Seekgram");
    } else {
        match autostart::set(true) {
            Ok(()) => log(&format!("Start at logon: {command}")),
            Err(error) => log(&error),
        }
    }
}
