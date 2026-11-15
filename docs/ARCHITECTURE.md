# Seekgram architecture

Seekgram indexes the photos, videos and audio of a user's Telegram account with a
local multimodal embedding model and lets them search it: from a Telegram chat of
their own (`?pink dragon`), and through an MCP endpoint for AI agents. Windows only;
the model runs on any DirectX 12 GPU through DirectML, or on the CPU. Modelled on
[Vitreus](https://github.com/Nominy/vitreus): one Rust binary, one-line installer,
GitHub Pages site.

## One binary, three roles

`seekgram.exe` (built with `windows_subsystem = "windows"`):

| Invocation | Role |
|--|--|
| `seekgram` | The control panel (GPUI window). Starts the service if it is not running. |
| `seekgram service` | The background service. Started at Windows logon (HKCU `Run` key). No window. |
| `seekgram start` / `stop` / `status` | Start the service detached / ask it to shut down / print status JSON. |
| `seekgram embed-worker [--engine E] [--self-test]` | A model worker, started by the service (below). |
| `seekgram --version`, `--help` | |

The service does everything; the panel is only a view. They talk over loopback HTTP.

## Data directory

Everything lives in `%LOCALAPPDATA%\Seekgram` (`SEEKGRAM_DATA_DIR` overrides it, for a
second isolated instance during development). See `src/paths.rs`; use those functions,
never build paths by hand and never write next to the executable.

## Modules and owners

| Path | What |
|--|--|
| `src/main.rs`, `src/cli.rs` | Mode dispatch, console attach, single-instance. |
| `src/paths.rs`, `src/config.rs` | Paths, `Settings` (JSON), `Secrets` (DPAPI). |
| `src/service/state.rs` | `Shared`, `Status` and all its sections. **The contract.** |
| `src/service/mod.rs` | Service lifecycle: lock, log, spawn tasks, shut down. |
| `src/service/api.rs` | Control API (below) and mounting of `/mcp`. |
| `src/service/account.rs` | Login state machine; owns `Status.account`. |
| `src/service/telegram.rs`, `mcp.rs`, `search.rs`, `chat_search.rs`, `cache.rs` | Telegram client, MCP protocol, index search, chat search and the media cache. |
| `src/service/provision.rs` | Gets ffmpeg, ONNX Runtime with DirectML and the model, picks the engine by self-test; owns `Status.setup`. |
| `src/service/workers.rs` | Pool of model worker processes (`seekgram embed-worker`). |
| `src/service/indexer.rs` | The indexer; owns `Status.indexer` and `Status.index`. |
| `src/embed/` | The model worker: JSONL protocol, preprocessing, ONNX Runtime sessions, self-test. |
| `src/app/` | The GPUI control panel. |
| `assets/model/` | The six model graphs (fp16 and fp32), embedded and written next to the downloaded weights. |
| `assets/selftest/` | Self-test picture, sound and clip, and the reference vectors (`vectors.f32`). |
| `install.ps1`, `site/`, `.github/`, `README.md`, `assets/` | Distribution. |

## Shared state (`src/service/state.rs`)

Read that file; it is short and is the contract. Rules:

- A subsystem writes only its own `Status` section, through `shared.update(|s| ...)`.
- User-facing strings in `Status` (`error`, `reason`, `detail`) are plain sentences a
  non-programmer can act on. No Rust error dumps, no internal identifiers.
- Long tasks check `shared.stopping()` / `select!` on `shared.stopped()` and return
  promptly (within ~5 s) after finishing or abandoning the current unit of work.
- Call `shared.notify()` after changing anything other tasks wait on.
- Never log message text, captions, file names, phone numbers, API hashes or tokens.
  Logging is `eprintln!`; the service redirects stderr to `logs/service.log`.

## Internal contracts between modules

```rust
// src/service/provision.rs
/// Runs until shutdown: prepares the model, then stays idle.
/// Re-runs when `repair()` is called. Owns `Status.setup`.
pub async fn run(shared: Arc<Shared>);
/// Asks `run` to verify and repair everything again (control API: POST /api/setup/repair).
pub fn repair();
/// The engine workers run on (`directml-fp16`, `directml-fp32` or `cpu-fp32`);
/// `Some` only while `Status.setup.state == Ready`.
pub fn engine() -> Option<Engine>;
/// Paths of the prepared tools. Only valid once `Status.setup.state == Ready`.
pub fn ffmpeg() -> PathBuf;
pub fn ffprobe() -> PathBuf;

// src/service/workers.rs
#[derive(Clone, Copy, PartialEq, Eq)] pub enum Kind { Text, Image, Audio, Video }
pub enum EmbedError {
    /// This input cannot be embedded (corrupt file, no frames). Record it and move on.
    Media(String),
    /// No worker could do it (model failed to load, process died twice). Stop and report.
    Worker(String),
}
/// One embedding request. `seconds`: length of an audio chunk file, at most 30.
pub struct Embed { pub kind: Kind, pub input: String, pub seconds: Option<f64> }
impl Workers {
    pub fn new(shared: Arc<Shared>) -> Arc<Self>;
    /// Embeds one item and returns a normalized 768-float vector. Workers start lazily,
    /// up to the configured count. `urgent` requests (searches) are served before
    /// queued indexing work.
    pub async fn embed(&self, request: Embed, urgent: bool) -> Result<Vec<f32>, EmbedError>;
    /// How many workers are loaded right now.
    pub fn loaded(&self) -> u32;
    /// Stops all worker processes (shutdown, or to free memory when idle).
    pub async fn stop(&self);
}

// src/service/indexer.rs
/// Runs until shutdown. Indexes while settings.indexing && account ready && setup ready,
/// otherwise sets Status.indexer.state = Off with a `reason` and waits on shared.changed.
pub async fn run(shared: Arc<Shared>, workers: Arc<Workers>);
```

`Arc<Workers>` is created once in `service/mod.rs` and also given to `search.rs` for
query embeddings (`urgent = true`).

## The model

EmbeddingGemma 2 (`onnx-community/embeddinggemma-2-ONNX`, pinned revision) runs in
ONNX Runtime 1.24.4 (its last DirectML build, loaded at run time from
`paths::runtime_dir()`) inside `seekgram embed-worker` processes, so a driver fault
cannot take the service down and stopping a worker frees its memory. Workers speak
JSONL: `{"id","kind","input","start_seconds"?,"end_seconds"?}` in, `{"id","vector"}` or
`{"id","error":{"type","message","location"}}` out. Preprocessing (tokenizer, libjpeg
decoding, torch's antialiased bicubic resize, log-mel features, ffmpeg's FFV1 round trip
for video frames) reproduces the Python worker the index was built with bit for bit, so
old and new vectors share the index under the same identity string.

Setup downloads, each pinned by SHA-256 and resumable: ffmpeg (GitHub), `onnxruntime.dll`
and `DirectML.dll` (only those zip members of their NuGet packages, by HTTP range), and
`tokenizer.json` plus the fp16 weights (1.5 GB) into `paths::model_dir()`. The graphs
come from `assets/model/`: the published ones with two exact rewrites DirectML needs in
the vision and audio encoders (`Reshape` with `allowzero=1` becomes `allowzero=0`, the same
as long as no target dimension is 0; the audio encoder's 24 rank-5 `MatMul`s are reshaped
to rank 3 and back).

An engine is ready only after `embed-worker --self-test` embeds the text, picture, sound
and clip in `assets/selftest/` and matches the Python worker's vectors (cosine at least
0.999, 0.995 for audio). Setup tries `directml-fp16`, then `directml-fp32`, then `cpu-fp32`
(the fp32 weights, 2.9 GB, are downloaded only then) and shows the one in use in
`Status.setup.detail`. `ready.json` in the data folder records the pins and the engine.
DirectML uses the hardware adapter with the most dedicated memory. Once ready, what the
Python setup left (`worker/`, `uv-cache/`, `tools/python`, `tools/uv.exe`, `models/`) is
deleted.

## Control API

Loopback only (`127.0.0.1:<settings.port>`), JSON, every request needs
`Authorization: Bearer <secrets.token>`. Errors are `4xx`/`5xx` with
`{"error": "sentence for the user"}`. Successful POSTs return the new `Status`.

| Method and path | Body | Effect |
|--|--|--|
| `GET /api/status` | | The `Status` document. The panel polls this once a second. |
| `POST /api/login/api_key` | `{"api_id": 123, "api_hash": "..."}` | Store the key, connect. |
| `POST /api/login/phone` | `{"phone": "+123..."}` | Request a login code. |
| `POST /api/login/code` | `{"code": "12345"}` | Sign in; may move to `password`. |
| `POST /api/login/password` | `{"password": "..."}` | Two-step verification. |
| `POST /api/logout` | | Log out of Telegram; keeps the index. |
| `POST /api/settings` | any subset of `Settings` fields | Change and persist settings. |
| `POST /api/search_chat/create` | | Create a private group "Search" and answer searches there. |
| `POST /api/search_chat/saved` | | Answer searches in Saved Messages again. |
| `POST /api/setup/repair` | | Re-check and repair the model setup, and test the engines again. |
| `POST /api/shutdown` | | Stop the service gracefully. |
| `POST /mcp` | JSON-RPC | MCP Streamable HTTP for agents (same token). Answers 404 unless `settings.mcp` is on; it is off by default because most users are not programmers. |

Login actions return as soon as Telegram answers; `Status.account.step` says what to
ask for next and `Status.account.error` why the last action failed.

## Index format

`paths::index_dir()/index.sqlite3`, unchanged from the prototype so an existing index
can be moved in: tables `metadata`, `items`, `checkpoints`, `failures`, `sources`
(schema in `src/service/indexer.rs`, `open_database`). Vectors are 768 little-endian
`f32`, L2-normalized. Model identity strings must stay byte-identical.

## Build and test

Do not run `cargo` while several people are editing; the integrator builds once.
GPUI: `gpui` is the crates.io package `gpui-pre` 0.3.10, with `gpui-component` 0.8.1.
Sources for both are in `~/.cargo/registry/src/index.crates.io-*/` (read them rather
than guessing an API); the same goes for `grammers-client-0.7.0`.
