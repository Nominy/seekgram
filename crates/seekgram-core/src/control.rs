//! The running service seen from outside: one-shot calls to its control API, and
//! starting it. Used by the CLI modes and the panel.

use std::time::{Duration, Instant};

use crate::{config::{Secrets, Settings}, paths};

/// One request to the running service's control API.
pub fn call(method: reqwest::Method, path: &str) -> Result<String, String> {
    let token = Secrets::load()?.token;
    let url = format!("http://127.0.0.1:{}{path}", Settings::load().port);
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(10)).build().map_err(|e| e.to_string())?;
    let response = client.request(method, url).bearer_auth(token).send().map_err(|_| "Seekgram is not running.".to_owned())?;
    let status = response.status();
    let body = response.text().map_err(|e| e.to_string())?;
    if status.is_success() { Ok(body) } else { Err(format!("Seekgram answered {status}: {body}")) }
}

pub fn running() -> bool { call(reqwest::Method::GET, "/api/status").is_ok() }

/// Starts the service as a detached background process and waits until it answers.
pub fn start_service() -> Result<(), String> {
    if running() { return Ok(()); }
    let exe = std::env::current_exe().map_err(|error| format!("Cannot find the Seekgram program file: {error}"))?;
    let mut command = std::process::Command::new(exe);
    command.arg("service").stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }
    command.spawn().map_err(|error| format!("Seekgram could not be started: {error}"))?;
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if running() { return Ok(()); }
        std::thread::sleep(Duration::from_millis(250));
    }
    Err(format!("Seekgram did not start. See {}", paths::logs_dir().join("service.log").display()))
}
