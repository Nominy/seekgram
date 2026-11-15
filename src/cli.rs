//! Mode dispatch: the panel, the service, or a one-shot command against a running service.

use std::time::{Duration, Instant};

use crate::{config::{Secrets, Settings}, paths};

const HELP: &str = "\
Seekgram: search your Telegram photos, videos and audio by meaning

usage: seekgram            open the control panel
       seekgram service    run the background service (what starts at logon)
       seekgram start      start the background service and return
       seekgram stop       stop the background service
       seekgram status     print the service status as JSON
       seekgram --version";

/// A GUI-subsystem process has no console of its own. When started from a terminal
/// without redirection, borrow the terminal so the CLI modes can print.
#[cfg(windows)]
fn attach_console() {
    use std::os::windows::io::IntoRawHandle;
    use windows::Win32::{Foundation::HANDLE, System::Console::{AttachConsole, GetStdHandle, SetStdHandle, ATTACH_PARENT_PROCESS, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE}};
    // SAFETY: plain Win32 calls; handles opened here are leaked on purpose, they stay
    // the process's standard streams until exit.
    unsafe {
        let redirected = GetStdHandle(STD_OUTPUT_HANDLE).is_ok_and(|handle| !handle.is_invalid() && !handle.0.is_null());
        if redirected || AttachConsole(ATTACH_PARENT_PROCESS).is_err() { return; }
        for target in [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
            if let Ok(console) = std::fs::OpenOptions::new().write(true).open("CONOUT$") {
                let _ = SetStdHandle(target, HANDLE(console.into_raw_handle()));
            }
        }
    }
}

#[cfg(not(windows))]
fn attach_console() {}

/// One request to the running service's control API.
fn call(method: reqwest::Method, path: &str) -> Result<String, String> {
    let token = Secrets::load()?.token;
    let url = format!("http://127.0.0.1:{}{path}", Settings::load().port);
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(10)).build().map_err(|e| e.to_string())?;
    let response = client.request(method, url).bearer_auth(token).send().map_err(|_| "Seekgram is not running.".to_owned())?;
    let status = response.status();
    let body = response.text().map_err(|e| e.to_string())?;
    if status.is_success() { Ok(body) } else { Err(format!("Seekgram answered {status}: {body}")) }
}

fn running() -> bool { call(reqwest::Method::GET, "/api/status").is_ok() }

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

fn report(result: Result<String, String>) -> i32 {
    match result {
        Ok(text) => { if !text.is_empty() { println!("{text}"); } 0 }
        Err(error) => { eprintln!("{error}"); 1 }
    }
}

/// Whether a panel is already open; if so it is brought to the front instead of opening
/// a second window. The tray icon opens the panel on every click, so this matters.
#[cfg(windows)]
fn focus_existing_panel() -> bool {
    use windows::{core::w, Win32::{Foundation::{GetLastError, ERROR_ALREADY_EXISTS}, System::Threading::CreateMutexW, UI::WindowsAndMessaging::{FindWindowW, IsIconic, SetForegroundWindow, ShowWindow, SW_RESTORE}}};
    // SAFETY: plain Win32 calls. The mutex handle is leaked on purpose: it marks this
    // process as the open panel until it exits.
    unsafe {
        if CreateMutexW(None, false, w!("Local\\SeekgramPanel")).is_err() || GetLastError() != ERROR_ALREADY_EXISTS { return false; }
        if let Ok(window) = FindWindowW(None, w!("Seekgram")) {
            if IsIconic(window).as_bool() { let _ = ShowWindow(window, SW_RESTORE); }
            let _ = SetForegroundWindow(window);
        }
        true
    }
}

#[cfg(not(windows))]
fn focus_existing_panel() -> bool { false }

pub fn run() -> i32 {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.first().map(String::as_str) {
        None => {
            if !focus_existing_panel() { crate::app::run(); }
            0
        }
        Some("service") => crate::service::main(),
        Some(command) => {
            attach_console();
            match command {
                "start" => report(start_service().map(|()| "Seekgram is running.".to_owned())),
                "stop" => {
                    if !running() { return report(Ok("Seekgram is not running.".to_owned())); }
                    let stopped = call(reqwest::Method::POST, "/api/shutdown").and_then(|_| {
                        let deadline = Instant::now() + Duration::from_secs(30);
                        while Instant::now() < deadline {
                            if !running() { return Ok("Seekgram stopped.".to_owned()); }
                            std::thread::sleep(Duration::from_millis(250));
                        }
                        Err("Seekgram is still shutting down.".to_owned())
                    });
                    report(stopped)
                }
                "status" => report(call(reqwest::Method::GET, "/api/status")),
                "--version" | "-V" => report(Ok(format!("seekgram {}", env!("CARGO_PKG_VERSION")))),
                "--help" | "-h" => report(Ok(HELP.to_owned())),
                other => report(Err(format!("Unknown command: {other}\n\n{HELP}"))),
            }
        }
    }
}
