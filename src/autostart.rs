//! Starting the service when the user logs in to Windows: a per-user `Run` entry,
//! which needs no administrator rights and is visible in Task Manager's Startup tab.
//!
//! The entry names this executable. Installed with Seekgram Setup that is
//! `%LOCALAPPDATA%\Seekgram.App\current\seekgram.exe`, a path updates keep.

const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
const VALUE: &str = "Seekgram";

fn refused(error: impl std::fmt::Display) -> String { format!("Windows did not allow changing the startup setting: {error}") }

/// The `Run` value for this executable: `"<exe>" service`.
pub fn command() -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|error| format!("Cannot find the Seekgram program file: {error}"))?;
    Ok(format!("\"{}\" service", exe.display()))
}

/// Adds or removes the logon entry for this executable.
#[cfg(windows)]
pub fn set(enabled: bool) -> Result<(), String> {
    use winreg::{enums::HKEY_CURRENT_USER, RegKey};
    let (key, _) = RegKey::predef(HKEY_CURRENT_USER).create_subkey(RUN_KEY).map_err(refused)?;
    if enabled {
        key.set_value(VALUE, &command()?).map_err(refused)
    } else {
        match key.delete_value(VALUE) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(refused(error)),
            _ => Ok(()),
        }
    }
}

/// Removes the logon entry only if it starts this executable, so uninstalling one copy
/// does not switch off another one installed elsewhere.
#[cfg(windows)]
pub fn remove_if_ours() -> Result<(), String> {
    use winreg::{enums::{HKEY_CURRENT_USER, KEY_READ, KEY_SET_VALUE}, RegKey};
    let key = match RegKey::predef(HKEY_CURRENT_USER).open_subkey_with_flags(RUN_KEY, KEY_READ | KEY_SET_VALUE) {
        Ok(key) => key,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(refused(error)),
    };
    let Ok(current) = key.get_value::<String, _>(VALUE) else { return Ok(()) };
    if !current.eq_ignore_ascii_case(&command()?) { return Ok(()); }
    match key.delete_value(VALUE) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(refused(error)),
        _ => Ok(()),
    }
}

#[cfg(not(windows))]
pub fn set(_: bool) -> Result<(), String> { Err("Starting at logon is only implemented for Windows.".to_owned()) }

#[cfg(not(windows))]
pub fn remove_if_ours() -> Result<(), String> { Ok(()) }
