//! Linux terminal login. Credential input is read only from the controlling terminal.
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{fd::AsRawFd, unix::fs::{MetadataExt, OpenOptionsExt}},
    time::{Duration, Instant},
};

use seekgram_core::{control::{call, login_step}, paths, status::{AccountStep, Status}};
use serde_json::json;

const WAIT: Duration = Duration::from_secs(100);

fn read_status(body: &str) -> Result<Status, String> {
    // Status supports missing fields for the Windows UI; a login response must
    // explicitly identify its account step before we solicit any credentials.
    let value: serde_json::Value = serde_json::from_str(body).map_err(|_| "Invalid login status from Seekgram.")?;
    if value.pointer("/account/step").is_none() {
        return Err("Invalid login status from Seekgram.".into());
    }
    serde_json::from_value(value).map_err(|_| "Invalid login status from Seekgram.".into())
}

fn current_status() -> Result<Status, String> {
    read_status(&call(reqwest::Method::GET, "/api/status")?)
}

pub(super) fn run() -> Result<String, String> {
    let mut terminal = OpenOptions::new().read(true).write(true).open("/dev/tty")
        .map_err(|_| "Login requires an interactive terminal.")?;
    let mut status = current_status()?;
    if status.account.step == AccountStep::Ready { return Ok("Seekgram is already logged in.".into()); }

    // Serialize interactive sessions for the same data directory. All untrusted
    // input stays out of arguments, environment variables and standard streams.
    let path = paths::data_dir().join("login.lock");
    let lock = OpenOptions::new().read(true).write(true).create(true).truncate(false).mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(path)
        .map_err(|_| "Cannot open the local login lock.")?;
    let meta = lock.metadata().map_err(|_| "Cannot inspect the local login lock.")?;
    // SAFETY: no arguments, returns the effective UID.
    if !meta.is_file() || meta.nlink() != 1 || meta.mode() & 0o7777 != 0o600 || meta.uid() != unsafe { libc::geteuid() } {
        return Err("Unsafe local login lock. Check the data directory permissions.".into());
    }
    // SAFETY: lock refers to a valid open file and is held until run returns.
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err("Another Seekgram login is already in progress.".into());
    }

    status = current_status()?;
    if status.account.step == AccountStep::Ready { return Ok("Seekgram is already logged in.".into()); }

    eprintln!("On Linux, the API credentials and reusable Telegram session are stored unencrypted on disk in your private data directory.");
    if !prompt(&mut terminal, "Type yes to continue with this local login (input hidden): ")?.trim().eq_ignore_ascii_case("yes") {
        return Err("Login cancelled without reading any credentials.".into());
    }
    let mut connecting_since = Instant::now();
    loop {
        use AccountStep::*;
        if status.account.error.is_some() {
            // Account errors and hints come from Telegram. Do not print text that
            // might contain credential input. A new prompt permits trying again.
            eprintln!("Seekgram could not complete the last login step. Check what you entered.");
        }
        let (endpoint, payload) = match status.account.step {
            Ready => return Ok("Seekgram is logged in.".into()),
            Connecting => {
                if connecting_since.elapsed() >= WAIT {
                    return Err("Seekgram is still connecting to Telegram. Try again later.".into());
                }
                std::thread::sleep(Duration::from_millis(500));
                status = current_status()?;
                // Do not repeat an error every time we poll.
                if status.account.step == Connecting { status.account.error = None; }
                continue;
            }
            ApiKey => {
                let id = prompt(&mut terminal, "Telegram API ID (my.telegram.org): ")?;
                let Ok(api_id) = id.trim().parse::<i32>() else {
                    eprintln!("The API ID must be a positive integer.");
                    continue;
                };
                if api_id <= 0 { eprintln!("The API ID must be a positive integer."); continue; }
                let api_hash = prompt(&mut terminal, "Telegram API hash (hidden): ")?;
                if api_hash.trim().len() != 32 || !api_hash.trim().bytes().all(|b| b.is_ascii_hexdigit()) {
                    eprintln!("The API hash must contain 32 hexadecimal characters.");
                    continue;
                }
                ("/api/login/api_key", json!({"api_id": api_id, "api_hash": api_hash.trim()}))
            }
            Phone => {
                let phone = prompt(&mut terminal, "Telegram phone number with country code (hidden): ")?;
                let number = phone.trim().chars().filter(|c| *c == '+' || c.is_ascii_digit()).collect::<String>();
                if !number.starts_with('+') || number[1..].len() < 7 || !number[1..].bytes().all(|b| b.is_ascii_digit()) {
                    eprintln!("Enter a phone number in international format, starting with +.");
                    continue;
                }
                ("/api/login/phone", json!({"phone": number}))
            }
            Code => {
                let code = prompt(&mut terminal, "Code Telegram sent you (hidden): ")?;
                let digits = code.chars().filter(char::is_ascii_digit).collect::<String>();
                if digits.is_empty() { eprintln!("Enter the numeric login code from Telegram."); continue; }
                ("/api/login/code", json!({"code": digits}))
            }
            Password => {
                let password = prompt(&mut terminal, "Telegram two-step verification password (hidden): ")?;
                if password.is_empty() { eprintln!("Enter your two-step verification password."); continue; }
                ("/api/login/password", json!({"password": password}))
            }
        };
        status = read_status(&login_step(endpoint, payload)?)?;
        connecting_since = Instant::now();
    }
}

struct RestoreTerminal<'a> {
    terminal: &'a mut File,
    previous: libc::termios,
}

impl Drop for RestoreTerminal<'_> {
    fn drop(&mut self) {
        // SAFETY: previous was filled by tcgetattr for this live descriptor.
        unsafe { libc::tcsetattr(self.terminal.as_raw_fd(), libc::TCSAFLUSH, &self.previous); }
        let _ = writeln!(self.terminal);
    }
}

fn prompt(terminal: &mut File, label: &str) -> Result<String, String> {
    let fd = terminal.as_raw_fd();
    // SAFETY: these are read-only queries; fd refers to an open terminal.
    if unsafe { libc::tcgetpgrp(fd) != libc::getpgrp() } {
        return Err("Login requires a foreground terminal.".into());
    }
    let mut previous = std::mem::MaybeUninit::uninit();
    // SAFETY: the terminal descriptor and pointer to writable termios storage are valid.
    if unsafe { libc::tcgetattr(fd, previous.as_mut_ptr()) } != 0 {
        return Err("Cannot read the terminal settings; no credentials were read.".into());
    }
    // SAFETY: tcgetattr succeeded.
    let previous = unsafe { previous.assume_init() };
    let mut hidden = previous;
    // Handle the common Ctrl-C / Ctrl-D cancellation as input so the guard
    // restores the console rather than leaving the user's terminal echo off.
    hidden.c_lflag &= !(libc::ECHO | libc::ECHONL | libc::ICANON | libc::ISIG | libc::IEXTEN);
    hidden.c_iflag &= !(libc::IXON | libc::ISTRIP);
    hidden.c_cc[libc::VMIN] = 1;
    hidden.c_cc[libc::VTIME] = 0;
    // SAFETY: hidden is initialized and fd refers to the open controlling terminal.
    if unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &hidden) } != 0 {
        return Err("Cannot disable terminal echo; no credentials were read.".into());
    }
    let guard = RestoreTerminal { terminal, previous };
    guard.terminal.write_all(label.as_bytes()).and_then(|()| guard.terminal.flush())
        .map_err(|_| "Cannot display the terminal prompt.")?;

    let mut bytes = Vec::new();
    let mut byte = [0];
    loop {
        guard.terminal.read_exact(&mut byte).map_err(|_| "Cannot read from the terminal.")?;
        match byte[0] {
            3 | 4 => return Err("Login cancelled.".into()),
            b'\n' | b'\r' => break,
            8 | 127 => {
                // Delete one UTF-8 code point, including multibyte characters.
                while bytes.pop().is_some_and(|b: u8| b & 0xc0 == 0x80) {}
            }
            b if bytes.len() < 4096 => bytes.push(b),
            _ => return Err("Login input is too long.".into()),
        }
    }
    String::from_utf8(bytes).map_err(|_| "Login input must be valid UTF-8.".into())
}
