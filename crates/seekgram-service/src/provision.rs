//! Gets the embedding model running on a PC that has nothing installed: no Python, no
//! ffmpeg, no graphics driver beyond what Windows ships. Everything is downloaded into the
//! data folder at pinned versions and checked against pinned checksums, so every install
//! runs the same software the index format was validated with.
//!
//! The model runs on DirectML (any DirectX 12 graphics card) in half precision when it can.
//! When that does not pass the self-test, setup steps down to DirectML in full precision,
//! then to the processor, and says which one is in use. A marker file records what was
//! prepared and which engine passed; while it matches this build, later starts are ready
//! at once.

use std::{
    fs, io,
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
    sync::Notify,
};

use crate::{
    config,
    embed::{self, Engine, gpu::Adapter},
    paths,
};

use super::{
    hidden,
    state::{Gpu, Setup, SetupState, SetupStep, Shared},
};

/// A release archive pinned by checksum, and the programs taken out of it.
struct Archive {
    name: &'static str,
    url: &'static str,
    sha256: &'static str,
    /// Path inside the zip, and the file name it gets in the tools folder.
    members: &'static [(&'static str, &'static str)],
}

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

/// One file of a NuGet package (a zip), fetched alone with an HTTP range request: the two
/// DLLs are 15 MB of 215 MB of packages.
struct Member {
    file: &'static str,
    url: &'static str,
    /// Where the member's deflated bytes start in the package, and how many there are.
    offset: u64,
    packed: u64,
    packed_sha256: &'static str,
    sha256: &'static str,
}

/// ONNX Runtime 1.24.4 (its last DirectML build) and DirectML 1.15.4, x64.
const RUNTIME_VERSION: &str = "onnxruntime-directml-1.24.4+directml-1.15.4";
static RUNTIME: [Member; 2] = [
    Member {
        file: "onnxruntime.dll",
        url: "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.24.4",
        offset: 6_342_542,
        packed: 6_039_111,
        packed_sha256: "fa778315b8039b717b0c5bee5fa71d9eb1ef892b761f2c61df0a351f56d5e07a",
        sha256: "e7eedec6a6f26dc39dc948276a75ef6d2bee3fff944d874ceed0bbd3b97bff40",
    },
    Member {
        file: "DirectML.dll",
        url: "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4",
        offset: 142_873_434,
        packed: 9_332_741,
        packed_sha256: "d0c0798ed5a3cfb114b590c99ca3c1062a45da1517b2bf5aab1dff9d60691bdd",
        sha256: "9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1",
    },
];

/// The ONNX export of the model the index was built with (`google/embeddinggemma-2`).
const MODEL_REPO: &str = "onnx-community/embeddinggemma-2-ONNX";
const MODEL_REVISION: &str = "daa72c51243991dfcaf9f9137d2c573d8f7790c0";

/// A file of the model repository, pinned by size and checksum. Stored in the model folder
/// under its own name, without the repository's `onnx/` folder.
struct ModelFile {
    path: &'static str,
    size: u64,
    sha256: &'static str,
}

impl ModelFile {
    fn name(&self) -> &'static str {
        self.path.rsplit('/').next().unwrap_or(self.path)
    }
}

const TOKENIZER: ModelFile =
    ModelFile { path: "tokenizer.json", size: 32_170_510, sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4" };
const WEIGHTS_FP16: [ModelFile; 3] = [
    ModelFile { path: "onnx/model_fp16.onnx_data", size: 542_085_120, sha256: "b9dbe09415d77c8686ba6e216a244086f4ada1608238c13f2bfaa587ba627207" },
    ModelFile { path: "onnx/vision_encoder_fp16.onnx_data", size: 335_513_088, sha256: "9f374070808b8b96252f7d04c9c4eae34df60283473fde5a3fbf4d9205495e19" },
    ModelFile { path: "onnx/audio_encoder_fp16.onnx_data", size: 586_694_912, sha256: "3a9417444f3cb7bb868525b66859cc1d4b7a00128ea0b111f7aee2843739cd3b" },
];
const WEIGHTS_FP32: [ModelFile; 3] = [
    ModelFile { path: "onnx/model.onnx_data", size: 1_084_170_240, sha256: "9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2" },
    ModelFile { path: "onnx/vision_encoder.onnx_data", size: 671_026_176, sha256: "ca8d28c99feb8129618f141978a27adae25090e0dadb68a30c235e191c01e37e" },
    ModelFile { path: "onnx/audio_encoder.onnx_data", size: 1_172_750_848, sha256: "6943b910d1d1e9b5dffd4ccad63f61aa308ad10b70ed128bb61886b119da1e3a" },
];

/// The graphs, rewritten so DirectML can run them (`Reshape` with `allowzero=0`; the audio
/// encoder's rank-5 `MatMul`s as rank 3, see docs/ARCHITECTURE.md). Both rewrites are exact;
/// the weights they point to are the repository's own.
const GRAPH_FILES: [(&str, &[u8]); 6] = [
    ("model_fp16.onnx", include_bytes!("../../../assets/model/model_fp16.onnx")),
    ("vision_encoder_fp16.onnx", include_bytes!("../../../assets/model/vision_encoder_fp16.onnx")),
    ("audio_encoder_fp16.onnx", include_bytes!("../../../assets/model/audio_encoder_fp16.onnx")),
    ("model.onnx", include_bytes!("../../../assets/model/model.onnx")),
    ("vision_encoder.onnx", include_bytes!("../../../assets/model/vision_encoder.onnx")),
    ("audio_encoder.onnx", include_bytes!("../../../assets/model/audio_encoder.onnx")),
];

const GIB: u64 = 1 << 30;
const TOOLS_SPACE: u64 = GIB / 2;
/// Kept free beyond what the downloads take, for the index and the media cache.
const SPACE_MARGIN: u64 = GIB / 2;

const DOWNLOAD_ATTEMPTS: u32 = 3;
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);
const TOOL_TIMEOUT: Duration = Duration::from_secs(30);
/// Loading the model and embedding four short inputs; minutes only on a slow processor.
const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(10 * 60);

static REPAIR: Notify = Notify::const_new();
/// The engine that passed the self-test; `None` until setup is ready.
static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);

/// A failed setup is tried again by itself: the usual cause at logon is a network that
/// is not up yet, and a user who never opens the panel would otherwise stay stuck.
const RETRY_AFTER: Duration = Duration::from_secs(5 * 60);

/// Runs until shutdown: prepares the model, then stays idle.
/// Re-runs when `repair()` is called, and periodically after a failure. Owns `Status.setup`.
pub async fn run(shared: Arc<Shared>) {
    loop {
        set_engine(None);
        setup(&shared, |setup| {
            setup.state = SetupState::Running;
            setup.error = None;
        });
        shared.notify();
        // Dropping the unfinished preparation kills whatever child process it is waiting
        // on, so shutdown never waits for a download or a test.
        let outcome = tokio::select! {
            _ = shared.stopped() => return,
            outcome = prepare(&shared) => outcome,
        };
        let failed = outcome.is_err();
        setup(&shared, |setup| {
            setup.bytes_done = None;
            setup.bytes_total = None;
            match outcome {
                Ok(ready) => {
                    eprintln!("Setup: {ready}");
                    setup.state = SetupState::Ready;
                    setup.detail = ready;
                }
                Err(error) => {
                    eprintln!("Setup failed: {error}");
                    setup.state = SetupState::Failed;
                    setup.error = Some(error);
                }
            }
        });
        shared.notify();
        if !failed {
            tokio::task::spawn_blocking(remove_leftovers);
        }
        tokio::select! {
            _ = shared.stopped() => return,
            _ = REPAIR.notified() => {
                let _ = fs::remove_file(marker_file());
            }
            _ = tokio::time::sleep(RETRY_AFTER), if failed => {}
        }
    }
}

/// Asks `run` to verify and repair everything again (control API: POST /api/setup/repair).
pub fn repair() {
    REPAIR.notify_one();
}

/// The engine model workers run on. Only `Some` while `Status.setup.state == Ready`.
pub fn engine() -> Option<Engine> {
    *ENGINE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn set_engine(engine: Option<Engine>) {
    *ENGINE.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = engine;
}

pub use paths::{ffmpeg, ffprobe};

fn marker_file() -> PathBuf {
    paths::data_dir().join("ready.json")
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

fn detail(shared: &Shared, detail: String) {
    eprintln!("Setup: {detail}");
    setup(shared, |setup| setup.detail = detail);
}

/// What a finished setup consisted of. Any difference from the running build (a new tool,
/// runtime or model pin, changed graphs) makes the setup run again.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Marker {
    ffmpeg: String,
    runtime: String,
    model: String,
    /// The engine that passed the self-test.
    engine: Engine,
}

impl Marker {
    fn of(ffmpeg: &str, runtime: &str, model: &str, engine: Engine) -> Self {
        Self { ffmpeg: ffmpeg.to_owned(), runtime: runtime.to_owned(), model: model.to_owned(), engine }
    }

    fn current(engine: Engine) -> Self {
        let graphs: Vec<&[u8]> = GRAPH_FILES.iter().map(|(_, bytes)| *bytes).collect();
        Self::of(FFMPEG_VERSION, RUNTIME_VERSION, &format!("{MODEL_REPO}@{MODEL_REVISION}+{}", digest(&graphs)), engine)
    }

    /// The engine of a saved marker that this build would have written itself.
    fn engine(saved: &str) -> Option<Engine> {
        let saved = serde_json::from_str::<Marker>(saved).ok()?;
        (saved == Self::current(saved.engine)).then_some(saved.engine)
    }
}

/// SHA-256 over the parts, each prefixed with its length so that bytes moving from one
/// part to the next still change the result.
fn digest(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    format!("{:x}", hasher.finalize())
}

fn weights(engine: Engine) -> &'static [ModelFile; 3] {
    if engine.fp32() { &WEIGHTS_FP32 } else { &WEIGHTS_FP16 }
}

/// Whether everything `engine` needs is in place at its full size.
fn installed(engine: Engine) -> bool {
    let sized = |path: PathBuf, size: u64| fs::metadata(path).is_ok_and(|meta| meta.len() == size);
    [ffmpeg(), ffprobe()].iter().all(|file| file.is_file())
        && RUNTIME.iter().all(|member| paths::runtime_dir().join(member.file).is_file())
        && std::iter::once(&TOKENIZER).chain(weights(engine)).all(|file| sized(paths::model_dir().join(file.name()), file.size))
        && GRAPH_FILES.iter().all(|(name, bytes)| sized(paths::model_dir().join(name), bytes.len() as u64))
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

/// The device an engine runs on, as the panel names it.
fn device(engine: Engine, adapter: Option<&Adapter>) -> String {
    match adapter {
        Some(adapter) if engine.directml() && !adapter.name.is_empty() => adapter.name.clone(),
        Some(_) if engine.directml() => "the graphics card".to_owned(),
        _ => "this PC's processor".to_owned(),
    }
}

fn ready_line(engine: Engine, adapter: Option<&Adapter>) -> String {
    format!("Model ready on {} ({})", device(engine, adapter), engine.label())
}

/// Prepares everything and returns the line the panel shows once ready.
async fn prepare(shared: &Shared) -> Result<String, String> {
    enter(shared, SetupStep::Checking, "Looking for the graphics card");
    let adapter = tokio::task::spawn_blocking(embed::gpu::adapter).await.ok().flatten();
    match &adapter {
        Some(adapter) => eprintln!("Setup: graphics card {} ({} MB, adapter {})", adapter.name, adapter.vram_mb, adapter.index),
        None => eprintln!("Setup: no hardware graphics card found"),
    }
    setup(shared, |setup| setup.gpu = adapter.as_ref().map(|adapter| Gpu { name: adapter.name.clone(), vram_mb: adapter.vram_mb }));

    let saved = fs::read_to_string(marker_file()).ok().and_then(|saved| Marker::engine(&saved));
    if let Some(engine) = saved.filter(|engine| installed(*engine) && (adapter.is_some() || !engine.directml())) {
        set_engine(Some(engine));
        return Ok(ready_line(engine, adapter.as_ref()));
    }

    for folder in [paths::tools_dir(), paths::runtime_dir(), paths::model_dir()] {
        fs::create_dir_all(&folder).map_err(|error| write_problem("creating its folders", &error))?;
    }
    let client = reqwest::Client::builder()
        .user_agent(concat!("seekgram/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(|error| {
            eprintln!("Setup: cannot create the download client: {error}");
            offline("the setup files")
        })?;

    install_ffmpeg(shared, &client).await?;
    install_runtime(shared, &client).await?;
    let mut fp32 = false;
    download_model(shared, &client, false).await?;

    let mut last = None;
    for engine in Engine::LADDER {
        if engine.directml() && adapter.is_none() {
            continue;
        }
        if engine.fp32() && !fp32 {
            enter(shared, SetupStep::Model, &match &last {
                Some(_) => "The graphics card could not run the smaller model; getting the full-precision one".to_owned(),
                None => "No DirectX 12 graphics card was found, so the model will run on the processor; getting its full-precision version".to_owned(),
            });
            download_model(shared, &client, true).await?;
            fp32 = true;
        }
        match self_test(shared, engine, adapter.as_ref()).await {
            Ok(()) => {
                let marker = serde_json::to_vec_pretty(&Marker::current(engine)).map_err(|error| error.to_string());
                marker.and_then(|bytes| config::write_atomic(&marker_file(), &bytes)).map_err(|error| {
                    eprintln!("Setup: cannot save the ready marker: {error}");
                    blocked("finishing the setup")
                })?;
                set_engine(Some(engine));
                return Ok(ready_line(engine, adapter.as_ref()));
            }
            Err(Verdict::Fatal(message)) => return Err(message),
            Err(Verdict::Engine(reason)) => {
                eprintln!("Setup: {} did not pass the self-test: {reason}", engine.name());
                last = Some(reason);
            }
        }
    }
    Err(format!(
        "The search model could not run on this PC, neither on the graphics card nor on the processor. Use Repair in Seekgram to try again; if it keeps failing, the reason is at the end of {}.",
        paths::logs_dir().join("service.log").display()
    ))
}

/// Fails before anything large is downloaded when it cannot fit.
fn check_space(needed: u64) -> Result<(), String> {
    if needed == 0 {
        return Ok(());
    }
    let needed = needed + SPACE_MARGIN;
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

async fn install_ffmpeg(shared: &Shared, client: &reqwest::Client) -> Result<(), String> {
    enter(shared, SetupStep::Ffmpeg, "Checking ffmpeg");
    if ffmpeg_ready().await {
        return Ok(());
    }
    check_space(TOOLS_SPACE)?;
    detail(shared, "Downloading ffmpeg, which reads videos and sounds".to_owned());
    let package = paths::tools_dir().join("ffmpeg.zip");
    let pinned = Pinned { what: "ffmpeg", url: FFMPEG.url, sha256: FFMPEG.sha256, size: None, window: None };
    download(shared, client, &pinned, &package, Progress::default()).await?;
    setup(shared, |setup| {
        setup.detail = "Unpacking ffmpeg".to_owned();
        setup.bytes_done = None;
        setup.bytes_total = None;
    });
    let source = package.clone();
    let unpacked = tokio::task::spawn_blocking(move || extract(&source, FFMPEG.members)).await.unwrap_or_else(|error| Err(io::Error::other(error)));
    let _ = fs::remove_file(&package);
    unpacked.map_err(|error| write_problem(&format!("unpacking {}", FFMPEG.name), &error))?;
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

/// ONNX Runtime and DirectML: two DLLs taken out of their NuGet packages.
async fn install_runtime(shared: &Shared, client: &reqwest::Client) -> Result<(), String> {
    enter(shared, SetupStep::Runtime, "Checking the model runtime");
    let mut missing = Vec::new();
    for member in &RUNTIME {
        if file_sha256(&paths::runtime_dir().join(member.file)).await.ok().as_deref() != Some(member.sha256) {
            missing.push(member);
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    let total: u64 = missing.iter().map(|member| member.packed).sum();
    check_space(total * 3)?;
    detail(shared, format!("Downloading ONNX Runtime and DirectML ({})", size_text(total)));
    let mut base = 0;
    for member in missing {
        let target = paths::runtime_dir().join(member.file);
        let packed = with_suffix(&target, ".deflate");
        let pinned = Pinned {
            what: "the model runtime",
            url: member.url,
            sha256: member.packed_sha256,
            size: Some(member.packed),
            window: Some((member.offset, member.packed)),
        };
        download(shared, client, &pinned, &packed, Progress { base, total: Some(total) }).await?;
        base += member.packed;
        let (source, partial) = (packed.clone(), with_suffix(&target, ".part"));
        let inflated = tokio::task::spawn_blocking(move || -> io::Result<()> {
            let mut file = fs::File::create(&partial)?;
            io::copy(&mut flate2::read::DeflateDecoder::new(io::BufReader::new(fs::File::open(&source)?)), &mut file)?;
            file.sync_all()
        });
        let doing = format!("unpacking {}", member.file);
        inflated.await.unwrap_or_else(|error| Err(io::Error::other(error))).map_err(|error| write_problem(&doing, &error))?;
        let _ = fs::remove_file(&packed);
        let partial = with_suffix(&target, ".part");
        match file_sha256(&partial).await {
            Ok(found) if found == member.sha256 => fs::rename(&partial, &target).map_err(|error| write_problem(&doing, &error))?,
            Ok(found) => {
                let _ = fs::remove_file(&partial);
                eprintln!("Setup: {} unpacked to checksum {found}, not the pinned one", member.file);
                return Err(offline("the model runtime"));
            }
            Err(error) => return Err(write_problem(&doing, &error)),
        }
    }
    Ok(())
}

/// The tokenizer and the weights of one precision, plus the graphs from this binary.
async fn download_model(shared: &Shared, client: &reqwest::Client, fp32: bool) -> Result<(), String> {
    if !fp32 {
        enter(shared, SetupStep::Model, "Checking the embedding model");
    }
    for (name, bytes) in GRAPH_FILES {
        let path = paths::model_dir().join(name);
        if fs::read(&path).is_ok_and(|existing| existing == bytes) {
            continue;
        }
        config::write_atomic(&path, bytes).map_err(|error| {
            eprintln!("Setup: cannot write the model graphs: {error}");
            blocked("writing the model files")
        })?;
    }
    let files: Vec<&ModelFile> = if fp32 { WEIGHTS_FP32.iter().collect() } else { std::iter::once(&TOKENIZER).chain(&WEIGHTS_FP16).collect() };
    let mut missing = Vec::new();
    for file in files {
        if file_sha256(&paths::model_dir().join(file.name())).await.ok().as_deref() != Some(file.sha256) {
            missing.push(file);
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    let total: u64 = missing.iter().map(|file| file.size).sum();
    let partial: u64 = missing.iter().map(|file| fs::metadata(with_suffix(&paths::model_dir().join(file.name()), ".part")).map(|meta| meta.len()).unwrap_or(0)).sum();
    check_space(total.saturating_sub(partial))?;
    let what = if fp32 { "the full-precision model" } else { "the embedding model" };
    detail(shared, format!("Downloading {what} ({})", size_text(total)));
    let mut base = 0;
    for file in missing {
        let url = format!("https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{}", file.path);
        let pinned = Pinned { what: "the embedding model", url: &url, sha256: file.sha256, size: Some(file.size), window: None };
        download(shared, client, &pinned, &paths::model_dir().join(file.name()), Progress { base, total: Some(total) }).await?;
        base += file.size;
    }
    Ok(())
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// A download pinned by checksum.
struct Pinned<'a> {
    /// What it is, for the user: "ffmpeg", "the embedding model".
    what: &'a str,
    url: &'a str,
    sha256: &'a str,
    size: Option<u64>,
    /// Only this part of the remote file: where it starts and how long it is.
    window: Option<(u64, u64)>,
}

/// Where a download stands within the bar the panel shows: bytes already counted before it,
/// and the bar's total when it spans several files.
#[derive(Clone, Copy, Default)]
struct Progress {
    base: u64,
    total: Option<u64>,
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

/// Downloads `file` to `target`, continuing a partial file where the server allows it. The
/// file only gets its final name once its checksum is the pinned one.
async fn download(shared: &Shared, client: &reqwest::Client, file: &Pinned<'_>, target: &Path, progress: Progress) -> Result<(), String> {
    if file_sha256(target).await.is_ok_and(|found| found == file.sha256) {
        return Ok(());
    }
    let doing = format!("downloading {}", file.what);
    let partial = with_suffix(target, ".part");
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        pause_before(attempt).await;
        let have = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);
        if file.size.is_some_and(|size| have > size) {
            let _ = fs::remove_file(&partial);
        }
        // A partial file that is already whole needs no request: the server would refuse the range.
        let fetched = if file.size == Some(have) { Ok(()) } else { fetch(shared, client, file, &partial, progress).await };
        let short = |size: u64| fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0) < size;
        let problem = match fetched {
            // The connection closed early without an error; what arrived is kept for the next attempt.
            Ok(()) if file.size.is_some_and(short) => "the download ended early".to_owned(),
            Ok(()) => match file_sha256(&partial).await {
                Ok(found) if found == file.sha256 => {
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
    Err(offline(file.what))
}

async fn pause_before(attempt: u32) {
    if attempt > 1 {
        tokio::time::sleep(Duration::from_secs(5 * u64::from(attempt - 1))).await;
    }
}

async fn fetch(shared: &Shared, client: &reqwest::Client, file: &Pinned<'_>, partial: &Path, progress: Progress) -> Result<(), FetchError> {
    let have = tokio::fs::metadata(partial).await.map(|meta| meta.len()).unwrap_or(0);
    let (start, limit) = match file.window {
        Some((start, length)) => (start, Some(length)),
        None => (0, None),
    };
    let mut request = client.get(file.url);
    if have > 0 || start > 0 {
        let from = start + have;
        let range = match limit {
            Some(length) => format!("bytes={from}-{}", start + length - 1),
            None => format!("bytes={from}-"),
        };
        request = request.header(reqwest::header::RANGE, range);
    }
    let mut response = request.send().await.map_err(network)?;
    let status = response.status();
    // `skip`: bytes of the response before the wanted part, when the server sent the whole file.
    let (mut output, mut done, mut skip) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        (tokio::fs::OpenOptions::new().create(true).append(true).open(partial).await.map_err(FetchError::Disk)?, have, 0)
    } else if status.is_success() {
        (tokio::fs::File::create(partial).await.map_err(FetchError::Disk)?, 0, start)
    } else {
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            // The partial file is not a prefix the server can continue.
            let _ = tokio::fs::remove_file(partial).await;
        }
        return Err(FetchError::Network(format!("the server answered {status}")));
    };
    let total = progress.total.or(limit).or_else(|| response.content_length().map(|rest| done + rest - skip));
    let show = |done: u64| {
        setup(shared, |setup| {
            setup.bytes_done = Some(progress.base + done);
            setup.bytes_total = total;
        });
    };
    show(done);
    let mut shown = Instant::now();
    while let Some(chunk) = response.chunk().await.map_err(network)? {
        let mut bytes = &chunk[..];
        let skipped = bytes.len().min(usize::try_from(skip).unwrap_or(usize::MAX));
        bytes = &bytes[skipped..];
        skip -= skipped as u64;
        if let Some(limit) = limit {
            bytes = &bytes[..bytes.len().min(usize::try_from(limit - done).unwrap_or(usize::MAX))];
        }
        output.write_all(bytes).await.map_err(FetchError::Disk)?;
        done += bytes.len() as u64;
        if limit == Some(done) {
            break;
        }
        if shown.elapsed() >= PROGRESS_INTERVAL {
            shown = Instant::now();
            show(done);
        }
    }
    output.flush().await.map_err(FetchError::Disk)?;
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

/// How a self-test ended when it did not pass.
#[derive(Debug, PartialEq)]
enum Verdict {
    /// The engine cannot run the model correctly here; the next one may.
    Engine(String),
    /// Something no other engine would fix; the sentence for the user.
    Fatal(String),
}

/// What the worker's self-test said on stderr: `stage=...` before each part, then
/// `passed` or `failure=<cause>`.
#[derive(Default)]
struct SelfTest {
    stage: Option<String>,
    failure: Option<String>,
    passed: bool,
}

impl SelfTest {
    /// The line to show when the test moves to its next part.
    fn line(&mut self, line: &str) -> Option<&'static str> {
        if let Some(stage) = line.strip_prefix("stage=") {
            self.stage = Some(stage.to_owned());
            return match stage {
                "text" => Some("Testing a text search"),
                "image" => Some("Testing a picture"),
                "audio" => Some("Testing a sound"),
                "video" => Some("Testing a video clip"),
                _ => None,
            };
        }
        if let Some(cause) = line.strip_prefix("failure=") {
            self.failure = Some(cause.to_owned());
        }
        if line == "passed" {
            self.passed = true;
        }
        None
    }

    fn verdict(&self, status: ExitStatus) -> Result<(), Verdict> {
        if status.success() && self.passed {
            return Ok(());
        }
        let at = self.stage.as_deref().unwrap_or("start");
        match self.failure.as_deref() {
            Some("files") => Err(Verdict::Fatal(format!(
                "Some of the model's files are missing. Antivirus software may have removed them: add the folder {} to its exclusions, then use Repair in Seekgram.",
                data_folder()
            ))),
            Some("runtime") => Err(Verdict::Fatal(format!(
                "The model runtime (ONNX Runtime) could not be started. Antivirus software may be blocking it: add the folder {} to its exclusions. If that does not help, install the Microsoft Visual C++ Redistributable from https://aka.ms/vs/17/release/vc_redist.x64.exe. Then use Repair in Seekgram.",
                data_folder()
            ))),
            Some("write") => Err(Verdict::Fatal(blocked("writing the test files"))),
            Some("media") => Err(Verdict::Fatal(blocked("reading the test files with ffmpeg"))),
            Some("load") => Err(Verdict::Engine("the model could not be loaded".to_owned())),
            Some("run") => Err(Verdict::Engine(format!("the model failed on the {at} test"))),
            Some("mismatch") => Err(Verdict::Engine(format!("the {at} result differs from the reference"))),
            Some(other) => Err(Verdict::Engine(format!("failure {other} at {at}"))),
            // The process ended without a word: a crash in the graphics driver, typically.
            None if self.stage.is_some() => Err(Verdict::Engine(format!("the test process ended ({status}) at {at}"))),
            None => Err(Verdict::Fatal(blocked("starting the model"))),
        }
    }
}

/// Runs `seekgram embed-worker --self-test` on one engine.
async fn self_test(shared: &Shared, engine: Engine, adapter: Option<&Adapter>) -> Result<(), Verdict> {
    enter(shared, SetupStep::SelfTest, &format!("Testing the model on {} ({})", device(engine, adapter), engine.label()));
    let exe = std::env::current_exe().map_err(|error| {
        eprintln!("Setup: cannot find the Seekgram program file: {error}");
        Verdict::Fatal(blocked("starting the model"))
    })?;
    let mut command = Command::new(exe);
    command.args(engine.arguments()).arg("--self-test").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped()).kill_on_drop(true);
    let mut child = hidden(&mut command).spawn().map_err(|error| {
        eprintln!("Setup: cannot run the self-test: {error}");
        Verdict::Fatal(blocked("starting the model"))
    })?;
    let mut seen = SelfTest::default();
    let stderr = child.stderr.take();
    let work = async {
        if let Some(stderr) = stderr {
            let mut reader = BufReader::new(stderr);
            let mut line = Vec::new();
            // Read as bytes: one line that is not valid UTF-8 must not stop the reading, or
            // the child would block on a full pipe.
            while reader.read_until(b'\n', &mut line).await? != 0 {
                let text = String::from_utf8_lossy(&line);
                let text = text.trim_end();
                eprintln!("Setup: self-test: {text}");
                if let Some(detail) = seen.line(text) {
                    setup(shared, |setup| setup.detail = detail.to_owned());
                }
                line.clear();
            }
        }
        child.wait().await
    };
    match tokio::time::timeout(SELF_TEST_TIMEOUT, work).await {
        Ok(Ok(status)) => seen.verdict(status),
        Ok(Err(error)) => {
            eprintln!("Setup: cannot watch the self-test: {error}");
            Err(Verdict::Fatal(blocked("starting the model")))
        }
        Err(_) => Err(Verdict::Engine(format!("no result within {} minutes", SELF_TEST_TIMEOUT.as_secs() / 60))),
    }
}

/// Bytes of real files under `folder`. Links are skipped.
fn tree_size(folder: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(folder) else { return 0 };
    let size = |entry: fs::DirEntry| match entry.file_type() {
        Ok(kind) if kind.is_dir() => tree_size(&entry.path()),
        Ok(kind) if kind.is_file() => entry.metadata().map(|meta| meta.len()).unwrap_or(0),
        _ => 0,
    };
    entries.flatten().map(size).sum()
}

/// What earlier versions installed to run the model in Python: the worker and its
/// environment, uv and its cache, uv's Python, and the Hugging Face model cache.
fn leftovers() -> [PathBuf; 5] {
    [
        paths::data_dir().join("worker"),
        paths::data_dir().join("uv-cache"),
        paths::tools_dir().join("python"),
        paths::tools_dir().join("uv.exe"),
        paths::data_dir().join("models"),
    ]
}

/// Deletes the leftovers of the Python setup once the new one is ready. What cannot be
/// deleted now (a file held open) is tried again after the next setup.
fn remove_leftovers() {
    let mut freed = 0;
    for path in leftovers() {
        let Ok(meta) = fs::symlink_metadata(&path) else { continue };
        let size = if meta.is_dir() { tree_size(&path) } else { meta.len() };
        let removed = if meta.is_dir() { fs::remove_dir_all(&path) } else { fs::remove_file(&path) };
        match removed {
            Ok(()) => freed += size,
            Err(error) => {
                let left = if meta.is_dir() { tree_size(&path) } else { size };
                freed += size.saturating_sub(left);
                eprintln!("Setup: cannot remove the old {}: {error}", path.file_name().unwrap_or_default().to_string_lossy());
            }
        }
    }
    if freed > 0 {
        eprintln!("Setup: removed the old Python model environment, {} freed", size_text(freed));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_hash_is_stable() {
        // A changed algorithm would silently make every existing install set up again.
        assert_eq!(digest(&[b"a", b"bc"]), "9a8acca1b6c6c0befd3fbc756aed625da998c998f7252e738c4ef061906b9b21");
        assert_ne!(digest(&[b"a", b"bc"]), digest(&[b"ab", b"c"]));
        assert_eq!(Marker::current(Engine::DirectmlFp16), Marker::current(Engine::DirectmlFp16));
    }

    #[test]
    fn marker_gives_the_engine_only_for_this_build() {
        for engine in Engine::LADDER {
            let saved = serde_json::to_string_pretty(&Marker::current(engine)).unwrap();
            assert_eq!(Marker::engine(&saved), Some(engine));
        }
        let current = Marker::current(Engine::CpuFp32);
        for changed in [
            Marker::of("7.0", &current.runtime, &current.model, Engine::CpuFp32),
            Marker::of(&current.ffmpeg, "onnxruntime-1.0", &current.model, Engine::CpuFp32),
            Marker::of(&current.ffmpeg, &current.runtime, "model@other", Engine::CpuFp32),
        ] {
            assert_eq!(Marker::engine(&serde_json::to_string(&changed).unwrap()), None, "{changed:?}");
        }
        assert_eq!(Marker::engine(""), None);
        assert_eq!(Marker::engine("{}"), None);
        // The marker of the Python setup.
        assert_eq!(Marker::engine(r#"{"worker_files":"x","uv":"0.13.8","ffmpeg":"8.0.1-essentials_build","model":"m"}"#), None);
    }

    #[test]
    fn pins_agree_with_each_other_and_with_the_worker() {
        assert!(FFMPEG.url.ends_with(&format!("ffmpeg-{FFMPEG_VERSION}.zip")));
        assert_eq!(FFMPEG_BANNER, format!("ffmpeg version {FFMPEG_VERSION}-"));
        assert_eq!(FFPROBE_BANNER, format!("ffprobe version {FFMPEG_VERSION}-"));
        assert!(FFMPEG.members.iter().all(|(member, _)| member.starts_with(&format!("ffmpeg-{FFMPEG_VERSION}/bin/"))));
        assert!(RUNTIME[0].url.ends_with("/Microsoft.ML.OnnxRuntime.DirectML/1.24.4") && RUNTIME_VERSION.contains("1.24.4"));
        assert!(RUNTIME[1].url.ends_with("/Microsoft.AI.DirectML/1.15.4") && RUNTIME_VERSION.contains("directml-1.15.4"));
        // Every graph the worker opens is shipped, and every weight file it needs is pinned.
        for engine in Engine::LADDER {
            for graph in embed::GRAPHS {
                let name = format!("{graph}{}.onnx", engine.suffix());
                assert!(GRAPH_FILES.iter().any(|(file, _)| *file == name), "{name}");
                assert!(weights(engine).iter().any(|file| file.name() == format!("{name}_data")), "{name}_data");
            }
        }
        assert_eq!(WEIGHTS_FP16.iter().chain([&TOKENIZER]).map(|file| file.size).sum::<u64>(), 1_496_463_630);
    }

    #[test]
    fn graphs_are_onnx_protos_naming_their_weights() {
        for (name, bytes) in GRAPH_FILES {
            let data = format!("{name}_data");
            assert!(bytes.windows(data.len()).any(|window| window == data.as_bytes()), "{name} does not reference {data}");
        }
    }

    #[test]
    fn ready_line_names_device_and_engine() {
        let card = Adapter { index: 1, name: "NVIDIA GeForce RTX 5060 Ti".to_owned(), vram_mb: 16_000 };
        assert_eq!(ready_line(Engine::DirectmlFp16, Some(&card)), "Model ready on NVIDIA GeForce RTX 5060 Ti (DirectML)");
        assert_eq!(ready_line(Engine::DirectmlFp32, Some(&card)), "Model ready on NVIDIA GeForce RTX 5060 Ti (DirectML, full precision)");
        assert_eq!(ready_line(Engine::CpuFp32, Some(&card)), "Model ready on this PC's processor (CPU, slow)");
        assert_eq!(ready_line(Engine::CpuFp32, None), "Model ready on this PC's processor (CPU, slow)");
    }

    #[test]
    fn reads_the_self_test() {
        #[cfg(windows)]
        fn exit(code: u32) -> ExitStatus {
            std::os::windows::process::ExitStatusExt::from_raw(code)
        }
        #[cfg(not(windows))]
        fn exit(code: u32) -> ExitStatus {
            std::os::unix::process::ExitStatusExt::from_raw((code as i32) << 8)
        }
        let run = |lines: &[&str], code: u32| {
            let mut seen = SelfTest::default();
            let shown: Vec<_> = lines.iter().filter_map(|line| seen.line(line)).collect();
            (seen.verdict(exit(code)), shown)
        };

        let (verdict, shown) = run(&["stage=loading_model", "device=X", "stage=text", "cosine=text 0.99996", "stage=image", "stage=audio", "stage=video", "passed"], 0);
        assert_eq!(verdict, Ok(()));
        assert_eq!(shown.len(), 4);

        let (verdict, _) = run(&["stage=loading_model", "device=X", "stage=image", "cosine=image 0.5", "failure=mismatch"], 1);
        assert!(matches!(verdict, Err(Verdict::Engine(reason)) if reason.contains("image")));
        assert!(matches!(run(&["stage=loading_model", "failure=load"], 1).0, Err(Verdict::Engine(_))));
        assert!(matches!(run(&["stage=loading_model", "stage=video", "failure=run"], 1).0, Err(Verdict::Engine(_))));
        // A crash mid-test (a driver fault) is the engine's.
        assert!(matches!(run(&["stage=loading_model", "stage=text"], 0xC000_0005).0, Err(Verdict::Engine(_))));
        // Nothing passed without saying so.
        assert!(matches!(run(&["stage=loading_model"], 0).0, Err(Verdict::Engine(_))));

        assert!(matches!(run(&["stage=loading_model", "failure=files"], 1).0, Err(Verdict::Fatal(message)) if message.contains("Antivirus")));
        assert!(matches!(run(&["stage=loading_model", "failure=runtime"], 1).0, Err(Verdict::Fatal(message)) if message.contains("Visual C++")));
        assert!(matches!(run(&["stage=loading_model", "stage=audio", "failure=media"], 1).0, Err(Verdict::Fatal(message)) if message.contains("ffmpeg")));
        assert!(matches!(run(&[], 1).0, Err(Verdict::Fatal(_))));
    }

    #[test]
    fn sizes_read_well() {
        assert_eq!(size_text(1_496_463_630), "1.5 GB");
        assert_eq!(size_text(15_371_852), "15 MB");
    }
}
