//! The background service: one process that owns the Telegram login, the index, the
//! model workers and the loopback API. The panel and CLI are only clients of it.

pub mod account;
pub mod api;
pub mod cache;
pub mod chat_search;
pub mod indexer;
pub mod install;
pub mod mcp;
pub mod provision;
pub mod search;
pub mod state;
pub mod telegram;
pub mod tray;
pub mod update;
pub mod workers;

use std::{fs, io::{BufRead, BufReader, Write}, time::Duration};

use fs2::FileExt;
use tokio::sync::mpsc;

use seekgram_core::{config::{Secrets, Settings}, paths};
use seekgram_core::{autostart, config};
use seekgram_embed as embed;

use state::Shared;

/// Keeps a child process from flashing a console window: the service has none of its own.
pub fn hidden(command: &mut tokio::process::Command) -> &mut tokio::process::Command {
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

const LOG_LIMIT: u64 = 5 * 1024 * 1024;

/// `2026-01-31T12:34:56Z` from the system clock, without a date library.
pub fn timestamp() -> String {
    let seconds = state::unix_now();
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Civil-from-days (Howard Hinnant's algorithm), valid for the whole Unix era.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rest / 3_600, rest % 3_600 / 60, rest % 60)
}

/// Sends everything written to stderr, by this process's own code, to
/// `logs/service.log` with a timestamp per line. The log is rotated once at 5 MB.
#[cfg(windows)]
fn redirect_stderr() -> Result<(), String> {
    use std::os::windows::io::IntoRawHandle;
    use windows::Win32::{Foundation::HANDLE, System::Console::{SetStdHandle, STD_ERROR_HANDLE}};
    fs::create_dir_all(paths::logs_dir()).map_err(|e| format!("cannot create the log folder: {e}"))?;
    let path = paths::logs_dir().join("service.log");
    let (reader, writer) = std::io::pipe().map_err(|e| format!("cannot create the log pipe: {e}"))?;
    // SAFETY: the handle is a valid pipe write end that is intentionally never closed;
    // it stays the process's stderr for its whole life.
    unsafe { SetStdHandle(STD_ERROR_HANDLE, HANDLE(writer.into_raw_handle())) }.map_err(|e| format!("cannot redirect the log: {e}"))?;
    std::thread::Builder::new().name("log".into()).spawn(move || {
        let open = || fs::OpenOptions::new().create(true).append(true).open(&path);
        let Ok(mut file) = open() else { return };
        for line in BufReader::new(reader).lines().map_while(Result::ok) {
            if file.metadata().map(|meta| meta.len() > LOG_LIMIT).unwrap_or(false) {
                drop(file);
                let _ = fs::rename(&path, path.with_extension("log.1"));
                let Ok(next) = open() else { return };
                file = next;
            }
            let _ = writeln!(file, "{} {line}", timestamp());
        }
    }).map_err(|e| format!("cannot start the log writer: {e}"))?;
    Ok(())
}

#[cfg(not(windows))]
fn redirect_stderr() -> Result<(), String> { Ok(()) }

/// The job `own_children` made, kept so `release_children` can change it.
#[cfg(windows)]
static JOB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Puts the service in a job that Windows ends when the service does, however it ends.
/// Without it a killed or crashed service leaves its model workers and ffmpeg running,
/// holding the GPU and locking files an update needs to replace.
#[cfg(windows)]
fn own_children() -> Result<(), String> {
    use windows::Win32::System::{
        JobObjects::{AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE},
        Threading::GetCurrentProcess,
    };
    // SAFETY: plain Win32 calls with a correctly sized, initialised structure. The job
    // handle is leaked on purpose: closing it, at process exit, is what ends the children.
    unsafe {
        let job = CreateJobObjectW(None, None).map_err(|e| format!("cannot create the process group: {e}"))?;
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, std::ptr::from_ref(&limits).cast(), size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32)
            .map_err(|e| format!("cannot configure the process group: {e}"))?;
        AssignProcessToJobObject(job, GetCurrentProcess()).map_err(|e| format!("cannot join the process group: {e}"))?;
        let _ = JOB.set(job.0 as usize);
        Ok(())
    }
}

#[cfg(not(windows))]
fn own_children() -> Result<(), String> { Ok(()) }

/// Lets processes started from now on outlive the service (Velopack's updater). Children
/// already in the job still end with it.
#[cfg(windows)]
fn release_children() {
    use windows::Win32::{Foundation::HANDLE, System::JobObjects::{JobObjectExtendedLimitInformation, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK}};
    let Some(&job) = JOB.get() else { return };
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK;
    // SAFETY: the job handle is still open (it is never closed) and the structure is correctly sized.
    let changed = unsafe { SetInformationJobObject(HANDLE(job as *mut _), JobObjectExtendedLimitInformation, std::ptr::from_ref(&limits).cast(), size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32) };
    if let Err(error) = changed { eprintln!("Seekgram: cannot release the process group: {error}"); }
}

#[cfg(not(windows))]
fn release_children() {}

/// Runs the service until it is asked to stop. Returns the process exit code.
pub fn main() -> i32 {
    if let Err(error) = fs::create_dir_all(paths::data_dir()) {
        eprintln!("Seekgram cannot create its data folder {}: {error}", paths::data_dir().display());
        return 1;
    }
    // Held for the life of the process; a second service would fight over the login and the index.
    let lock = match fs::OpenOptions::new().create(true).read(true).write(true).truncate(false).open(paths::lock_file()) {
        Ok(lock) => lock,
        Err(error) => { eprintln!("Seekgram cannot open its lock file: {error}"); return 1; }
    };
    if lock.try_lock_exclusive().is_err() {
        eprintln!("Seekgram is already running.");
        return 2;
    }
    if let Err(error) = redirect_stderr().and_then(|()| own_children()) {
        eprintln!("Seekgram: {error}");
        return 1;
    }
    eprintln!("Seekgram {} starting", env!("CARGO_PKG_VERSION"));
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(error) => { eprintln!("Cannot start the async runtime: {error}"); return 1; }
    };
    let code = match runtime.block_on(run()) {
        Ok(()) => 0,
        Err(error) => { eprintln!("Seekgram stopped with an error: {error}"); 1 }
    };
    // Blocking tasks (SQLite scans) must not keep a stopped service alive.
    runtime.shutdown_timeout(Duration::from_secs(5));
    eprintln!("Seekgram stopped");
    drop(lock);
    update::apply_pending();
    code
}

async fn run() -> Result<(), String> {
    // Media left behind by a crash mid-download is useless; the checkpoint re-fetches it.
    let _ = fs::remove_dir_all(paths::temp_dir());
    let secrets = Secrets::load()?;
    let (login, commands) = mpsc::channel(8);
    let shared = Shared::new(Settings::load(), secrets, login);
    let workers = workers::Workers::new(shared.clone());

    let tasks = [
        tokio::spawn(provision::run(shared.clone())),
        tokio::spawn(account::run(shared.clone(), workers.clone(), commands)),
        tokio::spawn(indexer::run(shared.clone(), workers.clone())),
        tokio::spawn(update::run(shared.clone())),
    ];
    tokio::task::spawn_blocking(cache::enforce);
    tray::spawn(shared.clone());
    let interrupted = shared.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() { interrupted.shutdown.send_replace(true); }
    });

    let served = api::serve(shared.clone(), workers.clone()).await;
    // Reached on shutdown, or when the API could not start; either way everything stops.
    shared.shutdown.send_replace(true);
    for task in tasks {
        if tokio::time::timeout(Duration::from_secs(15), task).await.is_err() { eprintln!("A task did not stop in time"); }
    }
    workers.stop().await;
    served
}
