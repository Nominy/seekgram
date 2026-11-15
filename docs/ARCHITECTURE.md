# Seekgram architecture

Seekgram indexes the photos, videos and audio of a user's Telegram account with a
local multimodal embedding model and lets them search it: from a Telegram chat of
their own (`?pink dragon`), and through an MCP endpoint for AI agents. Windows with
an NVIDIA GPU only, for now. Modelled on [Vitreus](https://github.com/Nominy/vitreus):
one Rust binary, one-line installer, GitHub Pages site.

## One binary, two roles

`seekgram.exe` (built with `windows_subsystem = "windows"`):

| Invocation | Role |
|--|--|
| `seekgram` | The control panel (GPUI window). Starts the service if it is not running. |
| `seekgram service` | The background service. Started at Windows logon (HKCU `Run` key). No window. |
| `seekgram start` / `stop` / `status` | Start the service detached / ask it to shut down / print status JSON. |
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
| `src/service/provision.rs` | Gets uv, ffmpeg, the Python environment and the model; owns `Status.setup`. |
| `src/service/workers.rs` | Pool of Python model worker processes. |
| `src/service/indexer.rs` | The indexer; owns `Status.indexer` and `Status.index`. |
| `src/app/` | The GPUI control panel. |
| `worker/` | `worker.py`, `pyproject.toml`, `uv.lock`: embedded in the binary, written to `paths::worker_dir()`. |
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
/// Runs until shutdown: prepares the worker environment, then stays idle.
/// Re-runs when `repair()` is called. Owns `Status.setup`.
pub async fn run(shared: Arc<Shared>);
/// Asks `run` to verify and repair everything again (control API: POST /api/setup/repair).
pub fn repair();
/// Paths of the prepared tools. Only valid once `Status.setup.state == Ready`.
pub fn python() -> PathBuf;        // worker_dir()/.venv/Scripts/python.exe
pub fn worker_script() -> PathBuf; // worker_dir()/worker.py
pub fn ffmpeg() -> PathBuf;
pub fn ffprobe() -> PathBuf;
/// Environment for worker processes (HF_HOME = paths::models_dir(), HF_HUB_OFFLINE=1, ...).
pub fn worker_env() -> Vec<(String, String)>;

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
    /// Stops all worker processes (shutdown, or to free VRAM when idle).
    pub async fn stop(&self);
}

// src/service/indexer.rs
/// Runs until shutdown. Indexes while settings.indexing && account ready && setup ready,
/// otherwise sets Status.indexer.state = Off with a `reason` and waits on shared.changed.
pub async fn run(shared: Arc<Shared>, workers: Arc<Workers>);
```

`Arc<Workers>` is created once in `service/mod.rs` and also given to `search.rs` for
query embeddings (`urgent = true`).

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
| `POST /api/setup/repair` | | Re-check and repair the model worker environment. |
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
