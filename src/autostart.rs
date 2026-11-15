//! Starting the service when the user logs in to Windows: a per-user `Run` entry,
//! which needs no administrator rights and is visible in Task Manager's Startup tab.

const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
const VALUE: &str = "Seekgram";

/// Adds or removes the logon entry for this executable.
#[cfg(windows)]
pub fn set(enabled: bool) -> Result<(), String> {
    use winreg::{enums::HKEY_CURRENT_USER, RegKey};
    let (key, _) = RegKey::predef(HKEY_CURRENT_USER).create_subkey(RUN_KEY)
        .map_err(|error| format!("Windows did not allow changing the startup setting: {error}"))?;
    if enabled {
        let exe = std::env::current_exe().map_err(|error| format!("Cannot find the Seekgram program file: {error}"))?;
        key.set_value(VALUE, &format!("\"{}\" service", exe.display()))
            .map_err(|error| format!("Windows did not allow changing the startup setting: {error}"))
    } else {
        match key.delete_value(VALUE) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(format!("Windows did not allow changing the startup setting: {error}")),
            _ => Ok(()),
        }
    }
}

#[cfg(not(windows))]
pub fn set(_: bool) -> Result<(), String> { Err("Starting at logon is only implemented for Windows.".to_owned()) }
