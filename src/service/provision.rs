//! Gets the model worker running on a PC that has nothing but an NVIDIA driver: no
//! Python, no ffmpeg, no terminal. Everything is downloaded into the data folder at
//! pinned versions, so every install runs the same software the index format was
//! validated with, and nothing already on the PC (a uv or ffmpeg on PATH, another
//! Python) can change the result.
//!
//! A marker file records what was prepared; while it matches this build, later starts
//! only look for the graphics card and are ready at once.

use std::{
    collections::VecDeque,
    fs, io,
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
    sync::Notify,
};

use crate::{config, paths};

use super::{
    hidden,
    state::{Gpu, Setup, SetupState, SetupStep, Shared},
};

const WORKER_PY: &str = include_str!("../../worker/worker.py");
const PYPROJECT: &str = include_str!("../../worker/pyproject.toml");
const UV_LOCK: &str = include_str!("../../worker/uv.lock");

/// A release archive pinned by checksum, and the programs taken out of it.
struct Archive {
    name: &'static str,
    url: &'static str,
    sha256: &'static str,
    /// Path inside the zip, and the file name it gets in the tools folder.
    members: &'static [(&'static str, &'static str)],
}

const UV_VERSION: &str = "0.13.8";
/// How `uv --version` starts for the pinned release.
const UV_BANNER: &str = "uv 0.13.8 ";
static UV: Archive = Archive {
    name: "uv",
    url: "https://github.com/astral-sh/uv/releases/download/0.13.8/uv-x86_64-pc-windows-msvc.zip",
    sha256: "529e78a9cec6cd7e3e014f7ecd8bda0c113969d4ed0f2efeedb85c30943325b8",
    members: &[("uv.exe", "uv.exe")],
};

/// The gyan.dev "essentials" build the index format was validated with. It includes
/// libx264, ffv1 and the vp9_metadata bitstream filter, which the indexer relies on.
const FFMPEG_VERSION: &str = "8.0.1-essentials_build";
const FFMPEG_BANNER: &str = "ffmpeg version 8.0.1-essentials_build-";
const FFPROBE_BANNER: &str = "ffprobe version 8.0.1-essentials_build-";
static FFMPEG: Archive = Archive {
    name: "ffmpeg",
    url: "https://github.com/GyanD/codexffmpeg/releases/download/8.0.1/ffmpeg-8.0.1-essentials_build.zip",
    sha256: "e2aaeaa0fdbc397d4794828086424d4aaa2102cef1fb6874f6ffd29c0b88b673",
    members: &[
        ("ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe", "ffmpeg.exe"),
        ("ffmpeg-8.0.1-essentials_build/bin/ffprobe.exe", "ffprobe.exe"),
    ],
};

/// Must equal `MODEL` and `REVISION` in `worker.py`; a test holds them together.
const MODEL: &str = "google/embeddinggemma-2";
const MODEL_REVISION: &str = "f9972ac22e696a105c482202646979e33e4b5a1e";
/// Repository files the worker never opens: the tokenizer loads from `tokenizer.json`,
/// and the validated setup never had the SentencePiece copy.
const MODEL_SKIP: &[&str] = &["tokenizer.model"];

/// Runs inside the worker environment with the script on stdin, so no quoting can break
/// it. It reports only the kind of failure: tracebacks would carry local paths.
const DOWNLOAD_SCRIPT: &str = r#"
import errno, sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=sys.argv[3:], token=False)
except Exception as error:
    kind = type(error).__name__
    if isinstance(error, PermissionError):
        kind = "blocked"
    elif isinstance(error, OSError) and (error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112):
        kind = "disk_full"
    print("failure=" + kind, file=sys.stderr, flush=True)
    sys.exit(1)
"#;

const GIB: u64 = 1 << 30;
/// PyTorch with CUDA is a 3 GB download that unpacks to well over twice that.
const PACKAGES_SPACE: u64 = 9 * GIB;
const MODEL_SPACE: u64 = 2 * GIB;
const TOOLS_SPACE: u64 = GIB;
/// Cards sold as "4 GB" report 4096 MiB; the margin only absorbs rounding.
const MIN_VRAM_MB: u64 = 4000;

const DOWNLOAD_ATTEMPTS: u32 = 3;
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);
const TOOL_TIMEOUT: Duration = Duration::from_secs(30);
/// Long enough for the 3 GB of packages on a slow connection (about 2 Mbit/s); uv gives
/// up on a stalled connection by itself, so this only bounds a run that never ends.
const PACKAGES_TIMEOUT: Duration = Duration::from_secs(4 * 3600);
const MODEL_TIMEOUT: Duration = Duration::from_secs(4 * 3600);
/// The first load reads several GB of libraries from disk, often through antivirus.
const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const TAIL_LINES: usize = 40;

static REPAIR: Notify = Notify::const_new();

/// A failed setup is tried again by itself: the usual cause at logon is a network that
/// is not up yet, and a user who never opens the panel would otherwise stay stuck.
const RETRY_AFTER: Duration = Duration::from_secs(5 * 60);

/// Runs until shutdown: prepares the worker environment, then stays idle.
/// Re-runs when `repair()` is called, and periodically after a failure. Owns `Status.setup`.
pub async fn run(shared: Arc<Shared>) {
    let mut repairing = false;
    loop {
        setup(&shared, |setup| {
            setup.state = SetupState::Running;
            setup.error = None;
        });
        shared.notify();
        // Dropping the unfinished preparation kills whatever child process it is waiting
        // on, so shutdown never waits for a download or an install.
        let outcome = tokio::select! {
            _ = shared.stopped() => return,
            outcome = prepare(&shared, repairing) => outcome,
        };
        let failed = outcome.is_err();
        match &outcome {
            Ok(()) => eprintln!("Setup: ready"),
            Err(error) => eprintln!("Setup failed: {error}"),
        }
        setup(&shared, |setup| {
            setup.bytes_done = None;
            setup.bytes_total = None;
            match outcome {
                Ok(()) => {
                    setup.state = SetupState::Ready;
                    setup.detail = "The model worker is ready".to_owned();
                }
                Err(error) => {
                    setup.state = SetupState::Failed;
                    setup.error = Some(error);
                }
            }
        });
        shared.notify();
        tokio::select! {
            _ = shared.stopped() => return,
            _ = REPAIR.notified() => {
                let _ = fs::remove_file(marker_file());
                repairing = true;
            }
            _ = tokio::time::sleep(RETRY_AFTER), if failed => repairing = false,
        }
    }
}

/// Asks `run` to verify and repair everything again (control API: POST /api/setup/repair).
pub fn repair() {
    REPAIR.notify_one();
}

/// Paths of the prepared tools. Only valid once `Status.setup.state == Ready`.
pub fn python() -> PathBuf {
    paths::worker_dir().join(".venv").join("Scripts").join("python.exe")
}

pub fn worker_script() -> PathBuf {
    paths::worker_dir().join("worker.py")
}

pub fn ffmpeg() -> PathBuf {
    paths::tools_dir().join("ffmpeg.exe")
}

pub fn ffprobe() -> PathBuf {
    paths::tools_dir().join("ffprobe.exe")
}

/// Environment for worker processes: the model is read from Seekgram's own cache and
/// never fetched, and the tools folder comes first on PATH so anything that shells out
/// to ffmpeg gets the pinned build.
pub fn worker_env() -> Vec<(String, String)> {
    let mut path = paths::tools_dir().into_os_string();
    if let Some(inherited) = std::env::var_os("PATH") {
        path.push(";");
        path.push(inherited);
    }
    vec![
        ("HF_HOME".to_owned(), paths::models_dir().to_string_lossy().into_owned()),
        // Named outright as well: a cache folder set in the user's own environment would
        // otherwise take precedence over HF_HOME.
        ("HF_HUB_CACHE".to_owned(), paths::models_dir().join("hub").to_string_lossy().into_owned()),
        ("HF_HUB_OFFLINE".to_owned(), "1".to_owned()),
        ("HF_HUB_DISABLE_TELEMETRY".to_owned(), "1".to_owned()),
        ("PYTHONUTF8".to_owned(), "1".to_owned()),
        ("PYTHONNOUSERSITE".to_owned(), "1".to_owned()),
        ("PATH".to_owned(), path.to_string_lossy().into_owned()),
    ]
}

fn uv() -> PathBuf {
    paths::tools_dir().join("uv.exe")
}

fn marker_file() -> PathBuf {
    paths::worker_dir().join("ready.json")
}

/// The model's folder in the Hugging Face cache layout.
fn model_repo() -> PathBuf {
    paths::models_dir().join("hub").join(format!("models--{}", MODEL.replace('/', "--")))
}

fn model_snapshot() -> PathBuf {
    model_repo().join("snapshots").join(MODEL_REVISION)
}

fn setup(shared: &Shared, change: impl FnOnce(&mut Setup)) {
    shared.update(|status| change(&mut status.setup));
}

fn enter(shared: &Shared, step: SetupStep, detail: &str) {
    eprintln!("Setup: {detail}");
    setup(shared, |setup| {
        setup.step = step;
        setup.detail = detail.to_owned();
        setup.bytes_done = None;
        setup.bytes_total = None;
    });
}

/// What a finished setup consisted of. Any difference from the running build (new
/// worker files after an update, a new tool or model pin) makes the setup run again.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Marker {
    worker_files: String,
    uv: String,
    ffmpeg: String,
    model: String,
}

impl Marker {
    fn of(worker_files: [&str; 3], uv: &str, ffmpeg: &str, model: &str) -> Self {
        Self { worker_files: digest(&worker_files), uv: uv.to_owned(), ffmpeg: ffmpeg.to_owned(), model: model.to_owned() }
    }

    fn current() -> Self {
        Self::of([WORKER_PY, PYPROJECT, UV_LOCK], UV_VERSION, FFMPEG_VERSION, &format!("{MODEL}@{MODEL_REVISION}"))
    }

    fn matches(&self, saved: &str) -> bool {
        serde_json::from_str::<Marker>(saved).is_ok_and(|saved| saved == *self)
    }
}

/// SHA-256 over the parts, each prefixed with its length so that text moving from one
/// part to the next still changes the result.
fn digest(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn installed() -> bool {
    [python(), worker_script(), ffmpeg(), ffprobe()].iter().all(|file| file.is_file()) && model_snapshot().is_dir()
}

fn data_folder() -> String {
    paths::data_dir().display().to_string()
}

fn offline(what: &str) -> String {
    format!(
        "Seekgram could not download {what}. Check the internet connection (a VPN, proxy or firewall can block it), then use Repair in Seekgram to try again."
    )
}

fn blocked(doing: &str) -> String {
    format!(
        "Windows stopped Seekgram while {doing}. Antivirus software may be blocking it: add the folder {} to its exclusions, then use Repair in Seekgram.",
        data_folder()
    )
}

fn disk_full(doing: &str) -> String {
    format!(
        "The disk ran out of space while {doing}. Free up some space on the drive that holds {}, then use Repair in Seekgram.",
        data_folder()
    )
}

/// Explains a failed write in the data folder; the system's own wording goes to the log.
fn write_problem(doing: &str, error: &io::Error) -> String {
    eprintln!("Setup: {doing} failed: {error}");
    // 112 is ERROR_DISK_FULL.
    if error.raw_os_error() == Some(112) || error.kind() == io::ErrorKind::StorageFull { disk_full(doing) } else { blocked(doing) }
}

fn size_text(bytes: u64) -> String {
    let bytes = bytes as f64;
    if bytes >= 1e9 {
        format!("{:.1} GB", bytes / 1e9)
    } else if bytes >= 1e6 {
        format!("{:.0} MB", bytes / 1e6)
    } else {
        format!("{:.0} KB", bytes / 1e3)
    }
}

async fn prepare(shared: &Shared, reinstall: bool) -> Result<(), String> {
    enter(shared, SetupStep::Checking, "Looking for the NVIDIA graphics card");
    let gpu = find_gpu().await;
    setup(shared, |setup| setup.gpu = gpu.clone());
    let Some(gpu) = gpu else {
        return Err("Seekgram needs an NVIDIA graphics card with current drivers, and none was found on this PC. If it has one, install the current driver from nvidia.com/drivers, restart the PC, then use Repair in Seekgram.".to_owned());
    };
    if gpu.vram_mb < MIN_VRAM_MB {
        return Err(format!(
            "The graphics card {} has {:.1} GB of video memory; Seekgram needs an NVIDIA card with at least 4 GB.",
            gpu.name,
            gpu.vram_mb as f64 / 1024.0
        ));
    }

    let marker = Marker::current();
    if installed() && fs::read_to_string(marker_file()).is_ok_and(|saved| marker.matches(&saved)) {
        return Ok(());
    }

    for folder in [paths::tools_dir(), paths::worker_dir(), paths::models_dir()] {
        fs::create_dir_all(&folder).map_err(|error| write_problem("creating its folders", &error))?;
    }
    check_space()?;
    let client = reqwest::Client::builder()
        .user_agent(concat!("seekgram/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(|error| {
            eprintln!("Setup: cannot create the download client: {error}");
            offline("the setup files")
        })?;

    install_uv(shared, &client).await?;
    install_ffmpeg(shared, &client).await?;
    install_packages(shared, reinstall).await?;
    download_model(shared, &client).await?;
    self_test(shared).await?;

    let saved = serde_json::to_vec_pretty(&marker).map_err(|error| error.to_string());
    saved.and_then(|bytes| config::write_atomic(&marker_file(), &bytes)).map_err(|error| {
        eprintln!("Setup: cannot save the ready marker: {error}");
        blocked("finishing the setup")
    })
}

const GPU_QUERY: [&str; 2] = ["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"];

/// The NVIDIA card PyTorch will use. The driver installs `nvidia-smi` into System32,
/// which a service started from the registry may not have on PATH. With several cards
/// the one with the most memory is reported, which is normally the one CUDA uses first.
async fn find_gpu() -> Option<Gpu> {
    let windows = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    for program in [PathBuf::from("nvidia-smi"), windows.join("System32").join("nvidia-smi.exe")] {
        let mut command = Command::new(&program);
        command.args(GPU_QUERY).stdin(Stdio::null()).kill_on_drop(true);
        match tokio::time::timeout(TOOL_TIMEOUT, hidden(&mut command).output()).await {
            Ok(Ok(output)) if output.status.success() => {
                let best = parse_gpus(&String::from_utf8_lossy(&output.stdout)).into_iter().max_by_key(|gpu| gpu.vram_mb);
                if best.is_some() {
                    return best;
                }
                eprintln!("Setup: nvidia-smi listed no graphics card");
            }
            Ok(Ok(output)) => eprintln!("Setup: nvidia-smi failed ({})", output.status),
            Ok(Err(error)) => eprintln!("Setup: cannot run {}: {error}", program.display()),
            Err(_) => eprintln!("Setup: nvidia-smi did not answer"),
        }
    }
    None
}

/// Reads `name, memory in MiB` lines; lines without a memory figure are not usable cards.
fn parse_gpus(output: &str) -> Vec<Gpu> {
    output
        .lines()
        .filter_map(|line| {
            let (name, memory) = line.rsplit_once(',')?;
            let (name, vram_mb) = (name.trim(), memory.trim().parse::<u64>().ok()?);
            (!name.is_empty()).then(|| Gpu { name: name.to_owned(), vram_mb })
        })
        .collect()
}

/// Fails before anything large is downloaded when it cannot fit. Only what is still
/// missing is counted, so a repair of a complete install needs no extra room.
fn check_space() -> Result<(), String> {
    let mut needed = 0;
    if !python().is_file() {
        needed += PACKAGES_SPACE;
    }
    if !model_snapshot().is_dir() {
        needed += MODEL_SPACE;
    }
    if !ffmpeg().is_file() {
        needed += TOOLS_SPACE;
    }
    if needed == 0 {
        return Ok(());
    }
    let free = fs2::available_space(paths::data_dir()).map_err(|error| {
        eprintln!("Setup: cannot read the free disk space: {error}");
        format!("Seekgram could not find out how much disk space is free on the drive that holds {}.", data_folder())
    })?;
    if free >= needed {
        return Ok(());
    }
    Err(format!(
        "Setting up the model needs about {} of free disk space on the drive that holds {}, but only {} is free. Free up some space, then use Repair in Seekgram.",
        size_text(needed),
        data_folder(),
        size_text(free)
    ))
}

async fn install_uv(shared: &Shared, client: &reqwest::Client) -> Result<(), String> {
    enter(shared, SetupStep::Uv, "Checking the Python installer");
    if reports(&uv(), "--version", UV_BANNER).await {
        return Ok(());
    }
    setup(shared, |setup| setup.detail = "Downloading the Python installer (uv)".to_owned());
    install(shared, client, &UV).await?;
    if reports(&uv(), "--version", UV_BANNER).await { Ok(()) } else { Err(blocked("starting the Python installer")) }
}

async fn install_ffmpeg(shared: &Shared, client: &reqwest::Client) -> Result<(), String> {
    enter(shared, SetupStep::Ffmpeg, "Checking ffmpeg");
    if ffmpeg_ready().await {
        return Ok(());
    }
    setup(shared, |setup| setup.detail = "Downloading ffmpeg, which reads videos and sounds".to_owned());
    install(shared, client, &FFMPEG).await?;
    if ffmpeg_ready().await { Ok(()) } else { Err(blocked("starting ffmpeg")) }
}

async fn ffmpeg_ready() -> bool {
    reports(&ffmpeg(), "-version", FFMPEG_BANNER).await && reports(&ffprobe(), "-version", FFPROBE_BANNER).await
}

/// Whether the program exists, runs and is the pinned version. Running it is the only
/// cheap check that also catches a truncated or quarantined file.
async fn reports(program: &Path, flag: &str, banner: &str) -> bool {
    if !program.is_file() {
        return false;
    }
    let mut command = Command::new(program);
    command.arg(flag).stdin(Stdio::null()).kill_on_drop(true);
    match tokio::time::timeout(TOOL_TIMEOUT, hidden(&mut command).output()).await {
        Ok(Ok(output)) => output.status.success() && String::from_utf8_lossy(&output.stdout).starts_with(banner),
        _ => false,
    }
}

/// Downloads an archive and puts its programs into the tools folder.
async fn install(shared: &Shared, client: &reqwest::Client, archive: &'static Archive) -> Result<(), String> {
    let package = paths::tools_dir().join(format!("{}.zip", archive.name));
    download(shared, client, archive, &package).await?;
    setup(shared, |setup| {
        setup.detail = format!("Unpacking {}", archive.name);
        setup.bytes_done = None;
        setup.bytes_total = None;
    });
    let source = package.clone();
    let unpacked = tokio::task::spawn_blocking(move || extract(&source, archive.members)).await.unwrap_or_else(|error| Err(io::Error::other(error)));
    let _ = fs::remove_file(&package);
    unpacked.map_err(|error| write_problem(&format!("unpacking {}", archive.name), &error))
}

fn extract(package: &Path, members: &[(&str, &str)]) -> io::Result<()> {
    let mut archive = zip::ZipArchive::new(fs::File::open(package)?)?;
    for (member, name) in members {
        let target = paths::tools_dir().join(name);
        let partial = with_suffix(&target, ".part");
        let mut file = fs::File::create(&partial)?;
        io::copy(&mut archive.by_name(member)?, &mut file)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&partial, &target)?;
    }
    Ok(())
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

enum FetchError {
    /// The connection or the server; worth another attempt.
    Network(String),
    /// The local disk; another attempt would fail the same way.
    Disk(io::Error),
}

fn network(error: reqwest::Error) -> FetchError {
    let cause = std::error::Error::source(&error).map(|cause| format!(" ({cause})")).unwrap_or_default();
    FetchError::Network(format!("{error}{cause}"))
}

/// Downloads `archive` to `target`, continuing a partial file where the server allows
/// it. The file only gets its final name once its checksum is the pinned one.
async fn download(shared: &Shared, client: &reqwest::Client, archive: &Archive, target: &Path) -> Result<(), String> {
    if file_sha256(target).await.is_ok_and(|found| found == archive.sha256) {
        return Ok(());
    }
    let doing = format!("downloading {}", archive.name);
    let partial = with_suffix(target, ".part");
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        pause_before(attempt).await;
        let problem = match fetch(shared, client, archive.url, &partial).await {
            Ok(()) => match file_sha256(&partial).await {
                Ok(found) if found == archive.sha256 => {
                    return fs::rename(&partial, target).map_err(|error| write_problem(&doing, &error));
                }
                Ok(found) => {
                    // Nothing of a damaged file can be kept: the next attempt starts over.
                    let _ = fs::remove_file(&partial);
                    format!("the checksum is {found}, not the pinned one")
                }
                Err(error) => return Err(write_problem(&doing, &error)),
            },
            Err(FetchError::Disk(error)) => return Err(write_problem(&doing, &error)),
            Err(FetchError::Network(problem)) => problem,
        };
        eprintln!("Setup: {doing} failed, attempt {attempt} of {DOWNLOAD_ATTEMPTS}: {problem}");
    }
    Err(offline(archive.name))
}

async fn pause_before(attempt: u32) {
    if attempt > 1 {
        tokio::time::sleep(Duration::from_secs(5 * u64::from(attempt - 1))).await;
    }
}

async fn fetch(shared: &Shared, client: &reqwest::Client, url: &str, partial: &Path) -> Result<(), FetchError> {
    let offset = tokio::fs::metadata(partial).await.map(|meta| meta.len()).unwrap_or(0);
    let mut request = client.get(url);
    if offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let mut response = request.send().await.map_err(network)?;
    let status = response.status();
    let (mut file, mut done) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        (tokio::fs::OpenOptions::new().append(true).open(partial).await.map_err(FetchError::Disk)?, offset)
    } else if status.is_success() {
        // The server ignored the range and sent the whole file again.
        (tokio::fs::File::create(partial).await.map_err(FetchError::Disk)?, 0)
    } else {
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            // The partial file is not a prefix the server can continue.
            let _ = tokio::fs::remove_file(partial).await;
        }
        return Err(FetchError::Network(format!("the server answered {status}")));
    };
    let total = response.content_length().map(|rest| done + rest);
    let show = |done: u64| {
        setup(shared, |setup| {
            setup.bytes_done = Some(done);
            setup.bytes_total = total;
        });
    };
    show(done);
    let mut shown = Instant::now();
    while let Some(chunk) = response.chunk().await.map_err(network)? {
        file.write_all(&chunk).await.map_err(FetchError::Disk)?;
        done += chunk.len() as u64;
        if shown.elapsed() >= PROGRESS_INTERVAL {
            shown = Instant::now();
            show(done);
        }
    }
    file.flush().await.map_err(FetchError::Disk)?;
    show(done);
    Ok(())
}

async fn file_sha256(path: &Path) -> io::Result<String> {
    let path = path.to_owned();
    let hashing = tokio::task::spawn_blocking(move || -> io::Result<String> {
        let mut hasher = Sha256::new();
        io::copy(&mut fs::File::open(path)?, &mut hasher)?;
        Ok(format!("{:x}", hasher.finalize()))
    });
    hashing.await.map_err(io::Error::other)?
}

enum Stop {
    /// The program could not be started or watched.
    Launch(io::Error),
    Timeout,
}

/// Runs a prepared command to its end, handing every line it writes to stderr to
/// `each`. `input` becomes its stdin. The child dies with this future: on a timeout,
/// and when the service shuts down and the preparation is dropped.
async fn run_child(command: &mut Command, input: Option<&str>, limit: Duration, mut each: impl FnMut(&str)) -> Result<ExitStatus, Stop> {
    command.stdin(if input.is_some() { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::null()).stderr(Stdio::piped()).kill_on_drop(true);
    let mut child = hidden(command).spawn().map_err(Stop::Launch)?;
    let (stdin, stderr) = (child.stdin.take(), child.stderr.take());
    let work = async {
        if let (Some(mut stdin), Some(input)) = (stdin, input) {
            stdin.write_all(input.as_bytes()).await?;
            stdin.shutdown().await?;
        }
        if let Some(stderr) = stderr {
            let mut reader = BufReader::new(stderr);
            let mut line = Vec::new();
            // Read as bytes: one line that is not valid UTF-8 must not stop the reading,
            // or the child would block on a full pipe.
            while reader.read_until(b'\n', &mut line).await? != 0 {
                each(String::from_utf8_lossy(&line).trim_end());
                line.clear();
            }
        }
        child.wait().await
    };
    match tokio::time::timeout(limit, work).await {
        Ok(finished) => finished.map_err(Stop::Launch),
        Err(_) => Err(Stop::Timeout),
    }
}

/// Turns uv's output into the one line the panel shows. uv prints, when its output is
/// not a terminal, `Downloading NAME (SIZE)` and ` Downloaded NAME` for everything
/// larger than 1 MiB, and a summary line per phase.
#[derive(Default)]
struct UvProgress {
    /// Downloads announced and not finished yet, with their sizes where uv gave one.
    downloading: Vec<(String, Option<u64>)>,
}

impl UvProgress {
    /// The new line to show, when `line` changes what is going on.
    fn line(&mut self, line: &str) -> Option<String> {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Downloading ") {
            let sized = rest.strip_suffix(')').and_then(|rest| rest.rsplit_once(" (")).and_then(|(name, size)| Some((name, uv_size(size)?)));
            let (name, size) = match sized {
                Some((name, size)) => (name, Some(size)),
                None => (rest, None),
            };
            self.downloading.push((friendly_name(name), size));
            return Some(self.describe());
        }
        if let Some(name) = line.strip_prefix("Downloaded ") {
            let name = friendly_name(name);
            self.downloading.retain(|(active, _)| *active != name);
            return Some(if self.downloading.is_empty() { "Preparing the remaining packages".to_owned() } else { self.describe() });
        }
        if let Some(version) = line.strip_prefix("Using CPython ") {
            return Some(format!("Using Python {}", version.split_whitespace().next().unwrap_or_default()));
        }
        if line.starts_with("Creating virtual environment") {
            return Some("Creating the Python environment".to_owned());
        }
        if let Some(count) = line.strip_prefix("Prepared ").and_then(package_count) {
            return Some(format!("Installing {count} packages"));
        }
        if let Some(count) = line.strip_prefix("Installed ").and_then(package_count) {
            return Some(format!("Installed {count} packages"));
        }
        None
    }

    /// Names the largest running download, since that is the one being waited for.
    fn describe(&self) -> String {
        let Some((name, size)) = self.downloading.iter().max_by_key(|(_, size)| *size) else { return String::new() };
        let size = size.map(|bytes| format!(" ({})", size_text(bytes))).unwrap_or_default();
        match self.downloading.len() - 1 {
            0 => format!("Downloading {name}{size}"),
            1 => format!("Downloading {name}{size} and 1 more package"),
            others => format!("Downloading {name}{size} and {others} more packages"),
        }
    }
}

/// `3` from `3 packages in 1.2s`.
fn package_count(rest: &str) -> Option<u32> {
    let (count, unit) = rest.split_once(' ')?;
    unit.starts_with("package").then(|| count.parse().ok())?
}

/// uv calls its Python download `cpython-3.12.16-windows-x86_64-none (download)`.
fn friendly_name(name: &str) -> String {
    let name = name.trim().trim_end_matches(" (download)");
    match name.strip_prefix("cpython-").and_then(|rest| rest.split('-').next()) {
        Some(version) => format!("Python {version}"),
        None => name.to_owned(),
    }
}

/// Bytes from uv's binary-unit sizes such as `21.3MiB`.
fn uv_size(text: &str) -> Option<u64> {
    let (number, unit) = text.split_at(text.find(|character: char| character.is_ascii_alphabetic())?);
    let scale = match unit {
        "B" => 1.0,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((number.trim().parse::<f64>().ok()? * scale) as u64)
}

/// Says why uv failed, from the end of its output. Windows words its errors in the
/// system language, so only the error numbers and uv's own phrases are looked for.
fn uv_failure(output: &str) -> String {
    const DOING: &str = "installing the model's packages";
    let output = output.to_lowercase();
    let mentions = |phrases: &[&str]| phrases.iter().any(|phrase| output.contains(phrase));
    if mentions(&["os error 112)", "no space left"]) {
        disk_full(DOING)
    } else if mentions(&["os error 5)", "os error 32)", "os error 225)", "access is denied"]) {
        blocked(DOING)
    } else if mentions(&["error sending request", "failed to fetch", "failed to download", "dns error", "timed out", "connection", "network"]) {
        offline("Python and the model's packages")
    } else {
        format!(
            "Installing the model's packages failed. Use Repair in Seekgram to try again; if it keeps failing, the reason is at the end of {}.",
            paths::logs_dir().join("service.log").display()
        )
    }
}

/// Installs Python and the locked packages into the worker's own environment. The
/// worker files are written from this binary first, so an update replaces them.
async fn install_packages(shared: &Shared, reinstall: bool) -> Result<(), String> {
    enter(shared, SetupStep::Packages, "Installing Python and the model's packages (about 3 GB to download)");
    for (name, content) in [("worker.py", WORKER_PY), ("pyproject.toml", PYPROJECT), ("uv.lock", UV_LOCK)] {
        let path = paths::worker_dir().join(name);
        if fs::read(&path).is_ok_and(|existing| existing == content.as_bytes()) {
            continue;
        }
        config::write_atomic(&path, content.as_bytes()).map_err(|error| {
            eprintln!("Setup: cannot write the worker files: {error}");
            blocked("writing the worker files")
        })?;
    }

    let mut command = Command::new(uv());
    command
        .current_dir(paths::worker_dir())
        .args(["sync", "--frozen", "--no-dev"])
        .env("UV_PROJECT_ENVIRONMENT", paths::worker_dir().join(".venv"))
        .env("UV_PYTHON_INSTALL_DIR", paths::tools_dir().join("python"))
        .env("UV_CACHE_DIR", paths::data_dir().join("uv-cache"))
        .env("UV_PYTHON_PREFERENCE", "only-managed")
        // Keep the install inside the data folder: no registry entry, no python.exe on
        // the user's PATH, no settings from a uv the user may have configured.
        .env("UV_PYTHON_INSTALL_REGISTRY", "0")
        .env("UV_PYTHON_INSTALL_BIN", "0")
        .env("UV_NO_CONFIG", "1")
        .env("NO_COLOR", "1");
    if reinstall {
        // uv trusts an installed package's metadata; only a reinstall brings back files
        // that were deleted or quarantined afterwards. It reuses the download cache.
        command.arg("--reinstall");
    }

    let mut progress = UvProgress::default();
    let mut tail = VecDeque::with_capacity(TAIL_LINES);
    let outcome = run_child(&mut command, None, PACKAGES_TIMEOUT, |line| {
        if let Some(detail) = progress.line(line) {
            setup(shared, |setup| setup.detail = detail);
        }
        if tail.len() == TAIL_LINES {
            tail.pop_front();
        }
        tail.push_back(line.to_owned());
    })
    .await;
    match outcome {
        Ok(status) if status.success() && python().is_file() => Ok(()),
        Ok(status) => {
            let tail = Vec::from(tail).join("\n");
            eprintln!("Setup: uv sync failed ({status}):\n{tail}");
            Err(uv_failure(&tail))
        }
        Err(Stop::Timeout) => Err(format!(
            "Installing the model's packages did not finish within {} hours, which usually means a very slow connection. Use Repair in Seekgram to continue.",
            PACKAGES_TIMEOUT.as_secs() / 3600
        )),
        Err(Stop::Launch(error)) => {
            eprintln!("Setup: cannot run uv: {error}");
            Err(blocked("starting the Python installer"))
        }
    }
}

/// The files to download and their sizes, from the Hub's description of the pinned
/// revision. `None` when the answer is not about that revision or lacks a size.
fn model_files(info: &serde_json::Value) -> Option<Vec<(String, u64)>> {
    if info["sha"].as_str() != Some(MODEL_REVISION) {
        return None;
    }
    let files = info["siblings"].as_array()?.iter().filter(|file| !file["rfilename"].as_str().is_some_and(|name| MODEL_SKIP.contains(&name)));
    files.map(|file| Some((file["rfilename"].as_str()?.to_owned(), file["size"].as_u64()?))).collect()
}

async fn model_manifest(client: &reqwest::Client) -> Result<Vec<(String, u64)>, String> {
    let url = format!("https://huggingface.co/api/models/{MODEL}/revision/{MODEL_REVISION}?blobs=true");
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        pause_before(attempt).await;
        let answer = async { client.get(&url).send().await?.error_for_status()?.json::<serde_json::Value>().await };
        match answer.await {
            Ok(info) => match model_files(&info) {
                Some(files) if !files.is_empty() => return Ok(files),
                _ => eprintln!("Setup: the model's file list is not in the expected form"),
            },
            Err(error) => eprintln!("Setup: cannot read the model's file list, attempt {attempt} of {DOWNLOAD_ATTEMPTS}: {error}"),
        }
    }
    Err(offline("the embedding model"))
}

/// Whether every file of the model is in the cache at its full size.
fn model_complete(files: &[(String, u64)]) -> bool {
    let snapshot = model_snapshot();
    files.iter().all(|(name, size)| fs::metadata(snapshot.join(name)).is_ok_and(|meta| meta.len() == *size))
}

/// Bytes of real files under `folder`. Links are skipped, so a cache that links its
/// snapshot to its blobs is not counted twice. Each file is asked for its own size:
/// the folder listing does not see a file grow while it is being written.
fn tree_size(folder: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(folder) else { return 0 };
    let size = |entry: fs::DirEntry| match entry.file_type() {
        Ok(kind) if kind.is_dir() => tree_size(&entry.path()),
        Ok(kind) if kind.is_file() => fs::metadata(entry.path()).map(|meta| meta.len()).unwrap_or(0),
        _ => 0,
    };
    entries.flatten().map(size).sum()
}

/// Removes what an interrupted download left behind. The Hub client gives every
/// attempt a new temporary name, so these files can never be continued.
fn clear_partial_downloads() {
    let Ok(entries) = fs::read_dir(model_repo().join("blobs")) else { return };
    for path in entries.flatten().map(|entry| entry.path()) {
        if path.extension().is_some_and(|extension| extension == "incomplete") {
            let _ = fs::remove_file(path);
        }
    }
}

async fn download_model(shared: &Shared, client: &reqwest::Client) -> Result<(), String> {
    enter(shared, SetupStep::Model, "Checking the embedding model");
    let files = model_manifest(client).await?;
    if model_complete(&files) {
        return Ok(());
    }
    let total: u64 = files.iter().map(|(_, size)| size).sum();
    clear_partial_downloads();
    setup(shared, |setup| {
        setup.detail = format!("Downloading the embedding model ({})", size_text(total));
        setup.bytes_done = Some(0);
        setup.bytes_total = Some(total);
    });

    let mut command = Command::new(python());
    command
        .arg("-")
        .args([MODEL, MODEL_REVISION])
        .args(MODEL_SKIP)
        .envs(worker_env())
        .env("HF_HUB_OFFLINE", "0")
        // Plain HTTPS writes each file straight into the cache, where its growth can
        // be measured; the Xet transfer would also keep a second copy of the chunks.
        .env("HF_HUB_DISABLE_XET", "1")
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .env("HF_HUB_DISABLE_SYMLINKS_WARNING", "1");
    let repo = model_repo();
    let mut failure = None;
    let outcome = {
        let download = run_child(&mut command, Some(DOWNLOAD_SCRIPT), MODEL_TIMEOUT, |line| {
            if let Some(kind) = line.strip_prefix("failure=") {
                failure = Some(kind.to_owned());
            }
        });
        tokio::pin!(download);
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        loop {
            tokio::select! {
                outcome = &mut download => break outcome,
                _ = tick.tick() => {
                    let repo = repo.clone();
                    let done = tokio::task::spawn_blocking(move || tree_size(&repo)).await.unwrap_or(0);
                    setup(shared, |setup| setup.bytes_done = Some(done.min(total)));
                }
            }
        }
    };
    const DOING: &str = "downloading the embedding model";
    match outcome {
        Ok(status) if status.success() && model_complete(&files) => Ok(()),
        Ok(status) => {
            eprintln!("Setup: the model download failed ({status}, {})", failure.as_deref().unwrap_or("no reason given"));
            Err(match failure.as_deref() {
                Some("disk_full") => disk_full(DOING),
                Some("blocked") => blocked(DOING),
                // The Python environment itself did not start or lost its packages.
                None if !status.success() => blocked(DOING),
                _ => offline("the embedding model"),
            })
        }
        Err(Stop::Timeout) => Err(format!(
            "Downloading the embedding model did not finish within {} hours, which usually means a very slow connection. Use Repair in Seekgram to continue.",
            MODEL_TIMEOUT.as_secs() / 3600
        )),
        Err(Stop::Launch(error)) => {
            eprintln!("Setup: cannot run Python: {error}");
            Err(blocked(DOING))
        }
    }
}

/// What the worker's self-test said on stderr: `stage=...` before each part,
/// `device=...` once the model is loaded, and the Python error type if it gave up.
#[derive(Default)]
struct SelfTest {
    stage: Option<String>,
    error: Option<String>,
    /// PyTorch found no usable CUDA and loaded the model on the processor instead.
    on_cpu: bool,
}

impl SelfTest {
    /// The line to show when the test moves to its next part.
    fn line(&mut self, line: &str) -> Option<&'static str> {
        if let Some(stage) = line.strip_prefix("stage=") {
            self.stage = Some(stage.to_owned());
            return match stage {
                "loading_model" => Some("Loading the model onto the graphics card"),
                "text" => Some("Testing a text search"),
                "image" => Some("Testing a picture"),
                "audio" => Some("Testing a sound"),
                "video" => Some("Testing a video clip"),
                _ => None,
            };
        }
        if let Some(device) = line.strip_prefix("device=") {
            self.on_cpu = device != "cuda";
        }
        if let Some(error) = line.strip_prefix("worker initialization/self-test failed: ") {
            self.error = Some(error.to_owned());
        }
        None
    }

    fn passed(&self, status: ExitStatus) -> bool {
        status.success() && !self.on_cpu
    }

    /// The likely cause of a failed test, in words.
    fn failure(&self) -> String {
        const DRIVER: &str = "install the current driver from nvidia.com/drivers, restart the PC, then use Repair in Seekgram";
        let error = self.error.as_deref().unwrap_or_default();
        if self.on_cpu {
            return format!("PyTorch could not use the graphics card, which usually means the NVIDIA driver is too old for CUDA 12.8: {DRIVER}.");
        }
        if error.contains("OutOfMemory") {
            return "The graphics card ran out of memory while testing the model, which needs about 3 GB of it. Close games and other programs that use the graphics card, then use Repair in Seekgram.".to_owned();
        }
        match (self.stage.as_deref(), error) {
            (None, _) => blocked("starting the model worker"),
            (Some("loading_model"), "ImportError" | "ModuleNotFoundError" | "OSError" | "FileNotFoundError" | "PermissionError") => format!(
                "Some of the model's files are missing or could not be opened. Antivirus software may have removed or blocked them: add the folder {} to its exclusions, then use Repair in Seekgram.",
                data_folder()
            ),
            (Some("loading_model"), _) => format!(
                "The model could not be loaded onto the graphics card. Its NVIDIA driver may be too old for CUDA 12.8, or the card may be short of free memory: close other programs that use it, {DRIVER}."
            ),
            (Some("video"), "RequestError") => blocked("running ffmpeg on a test video"),
            (Some(stage), _) => {
                let part = match stage {
                    "text" => "a text search",
                    "image" => "a picture",
                    "audio" => "a sound",
                    _ => "a video clip",
                };
                format!(
                    "The model loaded but failed while testing {part}. The NVIDIA driver may be too old for CUDA 12.8, or the card may be short of free memory: close other programs that use it, {DRIVER}."
                )
            }
        }
    }
}

/// Loads the model and embeds one text, picture, sound and video clip, exactly as the
/// indexer will. Only an environment that passes this is called ready.
async fn self_test(shared: &Shared) -> Result<(), String> {
    enter(shared, SetupStep::SelfTest, "Starting the model for a test");
    let mut command = Command::new(python());
    command.current_dir(paths::worker_dir()).arg(worker_script()).arg("--self-test").arg("--ffmpeg").arg(ffmpeg()).envs(worker_env());
    let mut seen = SelfTest::default();
    let outcome = run_child(&mut command, None, SELF_TEST_TIMEOUT, |line| {
        if let Some(detail) = seen.line(line) {
            setup(shared, |setup| setup.detail = detail.to_owned());
        }
    })
    .await;
    match outcome {
        Ok(status) if seen.passed(status) => Ok(()),
        Ok(status) => {
            eprintln!(
                "Setup: the self-test failed ({status}, stage {}, error {}, on the {})",
                seen.stage.as_deref().unwrap_or("none"),
                seen.error.as_deref().unwrap_or("none"),
                if seen.on_cpu { "processor" } else { "graphics card" }
            );
            Err(seen.failure())
        }
        Err(Stop::Timeout) => Err(format!(
            "The model test did not finish within {} minutes. The PC may be very busy or the graphics card stuck: restart the PC, then use Repair in Seekgram.",
            SELF_TEST_TIMEOUT.as_secs() / 60
        )),
        Err(Stop::Launch(error)) => {
            eprintln!("Setup: cannot run the self-test: {error}");
            Err(blocked("starting the model worker"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker() -> Marker {
        Marker::of(["worker", "project", "lock"], "1.0", "2.0", "model@revision")
    }

    #[test]
    fn marker_hash_is_stable() {
        // A changed algorithm would silently make every existing install set up again.
        assert_eq!(marker().worker_files, "cac21f9ef5f8ec328bf5b2e115d1d9499ce3893c319c1b7a6c00c0d754757b19");
        assert_eq!(marker(), marker());
        assert_eq!(Marker::current(), Marker::current());
    }

    #[test]
    fn marker_matches_only_what_it_was_written_for() {
        let saved = serde_json::to_string_pretty(&marker()).unwrap();
        assert!(marker().matches(&saved));
        for changed in [
            Marker::of(["worker2", "project", "lock"], "1.0", "2.0", "model@revision"),
            Marker::of(["worker", "project2", "lock"], "1.0", "2.0", "model@revision"),
            Marker::of(["worker", "project", "lock2"], "1.0", "2.0", "model@revision"),
            Marker::of(["worker", "project", "lock"], "1.1", "2.0", "model@revision"),
            Marker::of(["worker", "project", "lock"], "1.0", "2.1", "model@revision"),
            Marker::of(["worker", "project", "lock"], "1.0", "2.0", "model@other"),
        ] {
            assert!(!changed.matches(&saved), "{changed:?}");
        }
        assert!(!marker().matches(""));
        assert!(!marker().matches("{}"));
        assert!(!marker().matches(r#"{"worker_files":"x"}"#));
    }

    #[test]
    fn hash_sees_text_moving_between_files() {
        assert_ne!(digest(&["a", "bc"]), digest(&["ab", "c"]));
        assert_eq!(digest(&["a", "bc"]), "9a8acca1b6c6c0befd3fbc756aed625da998c998f7252e738c4ef061906b9b21");
    }

    #[test]
    fn pins_agree_with_each_other_and_with_the_worker() {
        assert!(UV.url.contains(&format!("/{UV_VERSION}/")));
        assert_eq!(UV_BANNER, format!("uv {UV_VERSION} "));
        assert!(FFMPEG.url.ends_with(&format!("ffmpeg-{FFMPEG_VERSION}.zip")));
        assert_eq!(FFMPEG_BANNER, format!("ffmpeg version {FFMPEG_VERSION}-"));
        assert_eq!(FFPROBE_BANNER, format!("ffprobe version {FFMPEG_VERSION}-"));
        assert!(FFMPEG.members.iter().all(|(member, _)| member.starts_with(&format!("ffmpeg-{FFMPEG_VERSION}/bin/"))));
        assert!(WORKER_PY.contains(&format!("MODEL = \"{MODEL}\"")));
        assert!(WORKER_PY.contains(&format!("REVISION = \"{MODEL_REVISION}\"")));
        assert!(UV_LOCK.contains("torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl"));
    }

    #[test]
    fn reads_nvidia_smi_output() {
        let gpus = parse_gpus("NVIDIA GeForce RTX 4070, 12282\r\n");
        assert_eq!(gpus.len(), 1);
        assert_eq!((gpus[0].name.as_str(), gpus[0].vram_mb), ("NVIDIA GeForce RTX 4070", 12282));

        let gpus = parse_gpus("NVIDIA GeForce GTX 1650, 4096\nNVIDIA RTX A6000, 49140\n\n");
        assert_eq!(gpus.iter().map(|gpu| gpu.vram_mb).collect::<Vec<_>>(), [4096, 49140]);
        assert_eq!(gpus.into_iter().max_by_key(|gpu| gpu.vram_mb).unwrap().name, "NVIDIA RTX A6000");
    }

    #[test]
    fn ignores_nvidia_smi_lines_that_are_not_cards() {
        assert!(parse_gpus("").is_empty());
        assert!(parse_gpus("No devices were found\n").is_empty());
        assert!(parse_gpus("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.").is_empty());
        assert!(parse_gpus("Some Card, [N/A]\n, 4096\n").is_empty());
    }

    #[test]
    fn summarises_uv_output() {
        let mut progress = UvProgress::default();
        let mut shown = |line: &str| progress.line(line);
        assert_eq!(shown("Downloading cpython-3.12.16-windows-x86_64-none (download) (21.3MiB)").unwrap(), "Downloading Python 3.12.16 (22 MB)");
        assert_eq!(shown(" Downloaded cpython-3.12.16-windows-x86_64-none (download)").unwrap(), "Preparing the remaining packages");
        assert_eq!(shown("Using CPython 3.12.16").unwrap(), "Using Python 3.12.16");
        assert_eq!(shown("Creating virtual environment at: .venv").unwrap(), "Creating the Python environment");
        assert_eq!(shown("Downloading numpy (14.8MiB)").unwrap(), "Downloading numpy (16 MB)");
        assert_eq!(shown("Downloading torch (2.7GiB)").unwrap(), "Downloading torch (2.9 GB) and 1 more package");
        assert_eq!(shown("Downloading scipy (34.9MiB)").unwrap(), "Downloading torch (2.9 GB) and 2 more packages");
        assert_eq!(shown(" Downloaded numpy").unwrap(), "Downloading torch (2.9 GB) and 1 more package");
        assert_eq!(shown(" Downloaded scipy").unwrap(), "Downloading torch (2.9 GB)");
        assert_eq!(shown(" Downloaded torch").unwrap(), "Preparing the remaining packages");
        assert_eq!(shown("Prepared 51 packages in 12m 03s").unwrap(), "Installing 51 packages");
        assert_eq!(shown("Installed 51 packages in 48.21s").unwrap(), "Installed 51 packages");
    }

    #[test]
    fn keeps_uv_noise_out_of_the_status() {
        let mut progress = UvProgress::default();
        for line in ["", " + torch==2.9.1+cu128", "Resolved 52 packages in 2.50s", "warning: something about C:\\Users\\Name", "Installed Python 3.12.16 in 3.1s"] {
            assert_eq!(progress.line(line), None, "{line}");
        }
        assert_eq!(progress.line("Downloading some-package").unwrap(), "Downloading some-package");
        assert_eq!(progress.line("Downloading odd (sizeless)").unwrap(), "Downloading odd (sizeless) and 1 more package");
    }

    #[test]
    fn reads_uv_sizes() {
        assert_eq!(uv_size("512B"), Some(512));
        assert_eq!(uv_size("1.5KiB"), Some(1536));
        assert_eq!(uv_size("21.3MiB"), Some(22_334_668));
        assert_eq!(uv_size("2.7GiB"), Some(2_899_102_924));
        assert_eq!(uv_size("download"), None);
        assert_eq!(uv_size("3 parsecs"), None);
        assert_eq!(size_text(1_521_121_614), "1.5 GB");
        assert_eq!(size_text(106_259_850), "106 MB");
    }

    #[test]
    fn explains_uv_failures() {
        assert!(uv_failure("error: Failed to fetch: `https://download-r2.pytorch.org/whl/cu128/torch.whl`\n  Caused by: error sending request").contains("internet connection"));
        assert!(uv_failure("error: failed to write\n  Caused by: Недостаточно места на диске. (os error 112)").contains("ran out of space"));
        assert!(uv_failure("error: failed to remove file\n  Caused by: Отказано в доступе. (os error 5)").contains("Antivirus"));
        assert!(uv_failure("error: something new").contains("service.log"));
    }

    #[test]
    fn reads_the_model_file_list() {
        let info = serde_json::json!({
            "sha": MODEL_REVISION,
            "siblings": [
                {"rfilename": "config.json", "size": 4455},
                {"rfilename": "tokenizer.model", "size": 4689013},
                {"rfilename": "1_Pooling/config.json", "size": 90},
                {"rfilename": "model.safetensors", "size": 1488915288u64},
            ],
        });
        let files = model_files(&info).unwrap();
        assert_eq!(files.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>(), ["config.json", "1_Pooling/config.json", "model.safetensors"]);
        assert_eq!(files.iter().map(|(_, size)| size).sum::<u64>(), 1_488_919_833);

        assert!(model_files(&serde_json::json!({"sha": "another revision", "siblings": []})).is_none());
        assert!(model_files(&serde_json::json!({"sha": MODEL_REVISION})).is_none());
        assert!(model_files(&serde_json::json!({"sha": MODEL_REVISION, "siblings": [{"rfilename": "config.json"}]})).is_none());
    }

    #[test]
    fn explains_self_test_failures() {
        let run = |lines: &[&str]| {
            let mut seen = SelfTest::default();
            let shown: Vec<_> = lines.iter().filter_map(|line| seen.line(line)).collect();
            (seen, shown)
        };

        let (seen, shown) = run(&["stage=loading_model", "device=cuda", "stage=text", "stage=image", "stage=audio", "stage=video", "self-test passed"]);
        assert_eq!(shown.len(), 5);
        assert!(!seen.on_cpu);

        let (seen, _) = run(&["stage=loading_model", "device=cpu", "stage=text", "self-test passed"]);
        assert!(seen.on_cpu);
        assert!(seen.failure().contains("driver is too old"));

        let (seen, _) = run(&["stage=loading_model", "worker initialization/self-test failed: OutOfMemoryError"]);
        assert!(seen.failure().contains("ran out of memory"));

        let (seen, _) = run(&["stage=loading_model", "worker initialization/self-test failed: OSError"]);
        assert!(seen.failure().contains("missing or could not be opened"));

        let (seen, _) = run(&["stage=loading_model", "worker initialization/self-test failed: RuntimeError"]);
        assert!(seen.failure().contains("CUDA 12.8"));

        let (seen, _) = run(&["stage=loading_model", "device=cuda", "stage=text", "stage=image", "stage=audio", "stage=video", "worker initialization/self-test failed: RequestError"]);
        assert!(seen.failure().contains("ffmpeg"));

        let (seen, _) = run(&["stage=loading_model", "device=cuda", "stage=text", "stage=image", "worker initialization/self-test failed: AcceleratorError"]);
        assert!(seen.failure().contains("testing a picture"));

        let (seen, _) = run(&[]);
        assert!(seen.failure().contains("Antivirus"));
    }
}
