//! The running service seen from outside: one-shot calls to its control API, and
//! starting it. Used by the CLI modes and the panel.

use std::time::{Duration, Instant};

use crate::{config::Secrets, paths};
#[cfg(not(target_os = "linux"))]
use crate::config::Settings;

/// One request to the running service's control API.
#[cfg(not(target_os = "linux"))]
pub fn call(method: reqwest::Method, path: &str) -> Result<String, String> {
    let token = Secrets::load()?.token;
    let url = format!("http://127.0.0.1:{}{path}", Settings::load().port);
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(10)).build().map_err(|e| e.to_string())?;
    let response = client.request(method, url).bearer_auth(token).send().map_err(|_| "Seekgram is not running.".to_owned())?;
    let status = response.status();
    let body = response.text().map_err(|e| e.to_string())?;
    if status.is_success() { Ok(body) } else { Err(format!("Seekgram answered {status}: {body}")) }
}

#[cfg(target_os = "linux")]
pub fn call(method: reqwest::Method, path: &str) -> Result<String, String> {
    linux_call(method, path, None, Duration::from_secs(10))
}

#[cfg(target_os = "linux")]
pub fn login_step(path: &str, body: serde_json::Value) -> Result<String, String> {
    linux_call(reqwest::Method::POST, path, Some(body), Duration::from_secs(100))
}

#[cfg(target_os = "linux")]
fn linux_call(method: reqwest::Method, path: &str, body: Option<serde_json::Value>, timeout: Duration) -> Result<String, String> {
    if !path.starts_with("/api/") {
        return Err("Unsupported Seekgram control endpoint.".into());
    }
    paths::private_data_dir(&paths::data_dir(), false)?;
    let socket = paths::control_socket();
    let meta = std::fs::symlink_metadata(&socket).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            "Seekgram is not running.".to_owned()
        } else {
            format!("Cannot inspect the Seekgram control socket: {error}")
        }
    })?;
    paths::verify_control_socket(&socket, &meta)?;
    let token = Secrets::load_existing()?.token;
    // The address provides the HTTP Host; the transport uses only the private socket.
    let url = format!("http://localhost{path}");
    let client = reqwest::blocking::Client::builder()
        .unix_socket(socket).no_proxy().redirect(reqwest::redirect::Policy::none()).timeout(timeout)
        .build().map_err(|_| "Cannot create the Seekgram control client.")?;
    let mut request = client.request(method, url).bearer_auth(token);
    let is_login = body.is_some();
    if let Some(body) = body { request = request.json(&body); }
    let response = request.send().map_err(|_| "Seekgram did not answer the local control request.")?;
    let status = response.status();
    if !status.is_success() {
        // Do not print login response bodies: a failure could contain input.
        if is_login { return Err(format!("The login request failed ({status}).")); }
        let body = response.text().map_err(|_| "Cannot read Seekgram's response.")?;
        return Err(format!("Seekgram answered {status}: {body}"));
    }
    response.text().map_err(|_| "Cannot read Seekgram's response.".into())
}

pub fn running() -> bool { call(reqwest::Method::GET, "/api/status").is_ok() }

/// Starts the service as a detached background process and waits until it answers.
pub fn start_service() -> Result<(), String> {
    #[cfg(target_os = "linux")]
    paths::prepare_data_dir()?;
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
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: setsid is async-signal-safe and the child opens no Rust I/O here.
        unsafe { command.pre_exec(|| {
            if libc::setsid() < 0 { return Err(std::io::Error::last_os_error()); }
            Ok(())
        }); }
    }
    command.spawn().map_err(|error| format!("Seekgram could not be started: {error}"))?;
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if running() { return Ok(()); }
        std::thread::sleep(Duration::from_millis(250));
    }
    #[cfg(target_os = "linux")]
    return Err("Seekgram did not start. Run `seekgram service` in the foreground to see the error.".into());
    #[cfg(not(target_os = "linux"))]
    Err(format!("Seekgram did not start. See {}", paths::logs_dir().join("service.log").display()))
}
