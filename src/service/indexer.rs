//! The indexer: walks every chat from its oldest message to its newest, downloads each
//! photo, video and audio message, embeds it through the model workers and stores the
//! vectors.
//!
//! Ported from the prototype's `omni_index.py --watch`. The database format and every
//! decision that shapes its contents (what counts as media, source signatures, 30 s
//! chunks, the ffmpeg arguments) are unchanged, so an index made by the prototype keeps
//! growing here.

use std::{
    collections::{HashSet, VecDeque},
    ffi::OsString,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex, MutexGuard, mpsc},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use grammers_client::{
    ChatMap, Client, InvocationError, grammers_tl_types as tl,
    types::{Downloadable, Media, Message, PackedChat, Photo, media::Document, photo_sizes::PhotoSize},
};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde_json::{Value, json};
use tokio::{
    io::AsyncWriteExt,
    process::Command,
    sync::oneshot,
    task::JoinSet,
    time::{error::Elapsed, timeout},
};
use uuid::Uuid;

use crate::{config, paths};

use super::{
    hidden, provision,
    state::{AccountStep, CurrentChat, IndexCounts, IndexerState, SetupState, Shared, unix_now},
    telegram::marked_id,
    workers::{Embed, EmbedError, Kind, Workers},
};

const MODEL: &str = "google/embeddinggemma-2";
const REVISION: &str = "f9972ac22e696a105c482202646979e33e4b5a1e";
const VERSION: &str = "omni-v1";
/// Everything that shapes a vector. Search refuses an index whose stored copy differs.
const IDENTITY: &str = "google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e|omni-v1|query=task: search result | query: |media=no-prefix|audio=16khz-mono|video=1fps|dim=768";
const DATABASE: &str = "index.sqlite3";
const SCHEMA: &str = "
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS items(chat_id INTEGER,message_id INTEGER,segment INTEGER,
        modality TEXT,start_seconds REAL,end_seconds REAL,vector BLOB,
        PRIMARY KEY(chat_id,message_id,modality,segment));
    CREATE TABLE IF NOT EXISTS checkpoints(chat_id INTEGER PRIMARY KEY,message_id INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS failures(chat_id INTEGER,message_id INTEGER,error_type TEXT NOT NULL,
        attempts INTEGER NOT NULL,updated_at TEXT NOT NULL,PRIMARY KEY(chat_id,message_id));
    CREATE TABLE IF NOT EXISTS sources(chat_id INTEGER,message_id INTEGER,signature TEXT NOT NULL,
        PRIMARY KEY(chat_id,message_id));
";

/// Video and audio are embedded in pieces of this length; it is part of the index identity.
const CHUNK_SECONDS: i64 = 30;
const MAX_CHUNKS: i64 = 1000;
const MAX_MEDIA_BYTES: u64 = 1024 * 1024 * 1024;
/// Media that failed this often (unsupported codec, deleted file) stays recorded but is
/// no longer downloaded again on every pass.
const MAX_ATTEMPTS: i64 = 3;
const NETWORK_TIMEOUT: Duration = Duration::from_secs(180);
const TOOL_TIMEOUT: Duration = Duration::from_secs(600);
/// Between two passes over all chats, and before trying again after a failure.
const PASS_INTERVAL: Duration = Duration::from_secs(60);
const RETRY_INTERVAL: Duration = Duration::from_secs(60);
const PAGE: i32 = 100;
/// Messages kept ahead of the checkpoint while an older one is still being indexed.
const WINDOW: usize = 500;
const PUBLISH_INTERVAL: Duration = Duration::from_millis(300);
const RATE_WINDOW: Duration = Duration::from_secs(180);
/// Shortest span the rate is averaged over, so the first item does not read as hundreds per minute.
const RATE_FLOOR: Duration = Duration::from_secs(30);

const SELF_DESTRUCTING: &str = "self-destructing media is not indexed";
const TELEGRAM_SILENT: &str = "Telegram did not answer. Check the internet connection; indexing will try again in a minute.";
const INTERNAL: &str = "The indexer ran into an unexpected problem. It will start again in a minute.";
const TOOLS_MISSING: &str = "The video tools (ffmpeg) could not be started. Open Seekgram and run Repair in Setup.";
const SCRATCH_FAILED: &str = "Seekgram could not save a downloaded file. Check that the disk is not full; indexing will try again in a minute.";
const PAUSE_UNREADABLE: &str = "The saved Telegram pause could not be read and was discarded. Indexing will start again in a minute.";

/// Runs until shutdown. Indexes while settings.indexing && account ready && setup ready,
/// otherwise sets Status.indexer.state = Off with a `reason` and waits on shared.changed.
pub async fn run(shared: Arc<Shared>, workers: Arc<Workers>) {
    let progress = Arc::new(Progress::new(shared.clone()));
    let housekeeping = tokio::spawn(housekeeping(shared.clone(), progress.clone()));
    while !shared.stopping() {
        // Subscribed before the gate is read, so a change in between is not slept through.
        let mut changed = shared.changed.subscribe();
        if let Err(reason) = gate(&shared) {
            set_state(&shared, IndexerState::Off, Some(reason.to_owned()));
            tokio::select! {
                _ = changed.changed() => {}
                _ = shared.stopped() => {}
            }
            continue;
        }
        set_state(&shared, IndexerState::Starting, None);
        shared.update(|status| status.indexer.started_at = Some(unix_now()));
        let halt = index(&shared, &workers, &progress).await;
        progress.set(|tally| tally.chat = None);
        if let Halt::Failed(reason) = halt {
            set_state(&shared, IndexerState::Failed, Some(reason));
            let _ = timeout(RETRY_INTERVAL, interrupted(&shared)).await;
        }
    }
    housekeeping.abort();
    progress.flush();
    set_state(&shared, IndexerState::Off, None);
}

/// Why the indexer left what it was doing.
#[derive(Debug)]
enum Halt {
    Shutdown,
    /// Indexing was switched off, or login or setup is no longer ready.
    Gate,
    /// Nothing more can be done until something changes; the sentence is shown to the user.
    Failed(String),
}

/// How work on one message or chat ended short of success.
#[derive(Debug)]
enum Fault {
    Halt(Halt),
    /// Telegram asked for a pause of this many seconds.
    Flood(u64),
    /// A short code for `failures.error_type`. Never message content.
    Code(String),
}

impl From<Halt> for Fault {
    fn from(halt: Halt) -> Self {
        Fault::Halt(halt)
    }
}

fn code(text: &str) -> Fault {
    Fault::Code(text.to_owned())
}

/// Whether indexing may run, or what the user is told it is waiting for.
fn gate(shared: &Shared) -> Result<(), &'static str> {
    shared.read(|status| {
        if !status.settings.indexing {
            Err("Indexing is switched off")
        } else if status.account.step != AccountStep::Ready {
            Err("Waiting for you to log in to Telegram")
        } else {
            match status.setup.state {
                SetupState::Ready => Ok(()),
                SetupState::Failed => Err("The model could not be prepared (see Setup)"),
                _ => Err("Preparing the model (see Setup)"),
            }
        }
    })
}

fn halted(shared: &Shared) -> Option<Halt> {
    if shared.stopping() {
        Some(Halt::Shutdown)
    } else if gate(shared).is_err() {
        Some(Halt::Gate)
    } else {
        None
    }
}

/// Resolves once the service is stopping or the gate has closed.
async fn interrupted(shared: &Shared) -> Halt {
    // Subscribed before the first look, so a change in between wakes the wait below.
    let mut changed = shared.changed.subscribe();
    let mut shutdown = shared.shutdown.subscribe();
    loop {
        if let Some(halt) = halted(shared) {
            return halt;
        }
        let closed = tokio::select! {
            result = changed.changed() => result.is_err(),
            result = shutdown.changed() => result.is_err(),
        };
        if closed {
            return Halt::Shutdown;
        }
    }
}

/// Runs `work` unless and until the indexer has to stop; this is what makes switching
/// indexing off take effect within seconds, wherever the indexer happens to be waiting.
async fn guard<T>(shared: &Shared, work: impl Future<Output = T>) -> Result<T, Halt> {
    tokio::select! {
        biased;
        halt = interrupted(shared) => Err(halt),
        value = work => Ok(value),
    }
}

fn set_state(shared: &Shared, state: IndexerState, reason: Option<String>) {
    shared.update(|status| {
        status.indexer.state = state;
        status.indexer.reason = reason;
        status.indexer.flood_wait_seconds = None;
    });
}

fn unix_time() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

/// Items added per minute over the last few minutes.
#[derive(Default)]
struct Rate {
    added: VecDeque<Instant>,
}

impl Rate {
    fn record(&mut self, now: Instant) {
        self.added.push_back(now);
    }

    /// Averaged over the time since the oldest item still in the window, so the figure
    /// is right from the first minute and falls to zero once nothing is added.
    fn per_minute(&mut self, now: Instant) -> f64 {
        while self.added.front().is_some_and(|&at| now.duration_since(at) > RATE_WINDOW) {
            self.added.pop_front();
        }
        let Some(&oldest) = self.added.front() else { return 0.0 };
        let span = now.duration_since(oldest).clamp(RATE_FLOOR, RATE_WINDOW);
        self.added.len() as f64 * 60.0 / span.as_secs_f64()
    }
}

/// The indexer's counters. They live as long as the service, across indexing sessions.
struct Tally {
    chats_done: u32,
    chats_total: u32,
    chat: Option<CurrentChat>,
    messages_scanned: u64,
    items_added: u64,
    errors: u64,
    bytes_downloaded: u64,
    in_flight: u32,
    rate: Rate,
    last_activity_at: Option<u64>,
    published: Instant,
}

/// Collects progress from every task and copies it into `Status.indexer` a few times a
/// second at most; a download alone would otherwise rewrite the status per chunk.
struct Progress {
    shared: Arc<Shared>,
    tally: Mutex<Tally>,
}

impl Progress {
    fn new(shared: Arc<Shared>) -> Self {
        let tally = Tally {
            chats_done: 0,
            chats_total: 0,
            chat: None,
            messages_scanned: 0,
            items_added: 0,
            errors: 0,
            bytes_downloaded: 0,
            in_flight: 0,
            rate: Rate::default(),
            last_activity_at: None,
            published: Instant::now(),
        };
        Self { shared, tally: Mutex::new(tally) }
    }

    fn tally(&self) -> MutexGuard<'_, Tally> {
        self.tally.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Changes where the indexer stands. The status follows now, or at the next `flush`.
    fn set(&self, change: impl FnOnce(&mut Tally)) {
        let mut tally = self.tally();
        change(&mut tally);
        if tally.published.elapsed() >= PUBLISH_INTERVAL {
            self.publish(&mut tally);
        }
    }

    /// Like `set`, for things that count as the indexer doing something.
    fn note(&self, change: impl FnOnce(&mut Tally)) {
        self.set(|tally| {
            change(tally);
            tally.last_activity_at = Some(unix_now());
        });
    }

    fn flush(&self) {
        self.publish(&mut self.tally());
    }

    fn publish(&self, tally: &mut Tally) {
        let now = Instant::now();
        let items_per_minute = (tally.rate.per_minute(now) * 10.0).round() / 10.0;
        self.shared.update(|status| {
            let indexer = &mut status.indexer;
            indexer.chats_done = tally.chats_done;
            indexer.chats_total = tally.chats_total;
            indexer.chat = tally.chat.clone();
            indexer.messages_scanned = tally.messages_scanned;
            indexer.items_added = tally.items_added;
            indexer.errors = tally.errors;
            indexer.bytes_downloaded = tally.bytes_downloaded;
            indexer.in_flight = tally.in_flight;
            indexer.items_per_minute = items_per_minute;
            indexer.last_activity_at = tally.last_activity_at;
        });
        tally.published = now;
    }

    fn in_flight(&self) -> InFlight<'_> {
        self.set(|tally| tally.in_flight += 1);
        InFlight(self)
    }
}

/// Counts a media message as in flight until its work ends, also when it is cancelled.
struct InFlight<'a>(&'a Progress);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.set(|tally| tally.in_flight = tally.in_flight.saturating_sub(1));
    }
}

/// Keeps the status fresh between events, and `Status.index` in step with the database.
async fn housekeeping(shared: Arc<Shared>, progress: Arc<Progress>) {
    let mut second = 0u64;
    loop {
        progress.flush();
        if second % 5 == 0 {
            if let Ok(Some(counts)) = tokio::task::spawn_blocking(count_index).await {
                shared.update(|status| status.index = counts);
            }
        }
        second += 1;
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// Counts what the index holds through a read-only connection of its own, opened per
/// reading so the file is not held open while the indexer is off.
fn count_index() -> Option<IndexCounts> {
    let path = paths::index_dir().join(DATABASE);
    if !path.exists() {
        return Some(IndexCounts::default());
    }
    let counted = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX).and_then(|db| {
        db.busy_timeout(Duration::from_secs(2))?;
        read_counts(&db)
    });
    match counted {
        Ok(mut counts) => {
            let wal = path.with_extension("sqlite3-wal");
            counts.size_bytes = [path, wal].iter().filter_map(|file| std::fs::metadata(file).ok()).map(|file| file.len()).sum();
            Some(counts)
        }
        Err(error) => {
            eprintln!("indexer: the index could not be counted: {error}");
            None
        }
    }
}

fn read_counts(db: &Connection) -> rusqlite::Result<IndexCounts> {
    let mut counts = IndexCounts::default();
    let mut statement = db.prepare("SELECT modality,COUNT(*) FROM items GROUP BY modality")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let count = row.get::<_, i64>(1)? as u64;
        match row.get::<_, Option<String>>(0)?.as_deref() {
            Some("image") => counts.images = count,
            Some("video") => counts.video_segments = count,
            Some("audio") => counts.audio_segments = count,
            _ => {}
        }
    }
    counts.chats = db.query_row("SELECT COUNT(DISTINCT chat_id) FROM items", [], |row| row.get::<_, i64>(0))? as u64;
    counts.failures = db.query_row("SELECT COUNT(*) FROM failures", [], |row| row.get::<_, i64>(0))? as u64;
    Ok(counts)
}

/// Why the index could not be opened for writing.
#[derive(Debug)]
enum OpenError {
    Folder(std::io::Error),
    Database(rusqlite::Error),
    /// The index was made for another account or model; the metadata key that differs.
    Mismatch(&'static str),
    /// Vectors without the metadata that says what made them.
    Unidentified,
}

impl From<rusqlite::Error> for OpenError {
    fn from(error: rusqlite::Error) -> Self {
        OpenError::Database(error)
    }
}

impl From<OpenError> for Halt {
    fn from(error: OpenError) -> Self {
        let folder = paths::index_dir();
        let folder = folder.display();
        let way_out = format!("stop Seekgram and move the folder {folder} somewhere else to start a new index");
        Halt::Failed(match error {
            OpenError::Folder(error) => {
                eprintln!("indexer: the index folder could not be created: {error}");
                format!("Seekgram cannot create its index folder ({folder}). Check that the disk is not full or read-only.")
            }
            OpenError::Database(error) => return database_failure(error),
            OpenError::Mismatch("account_id") => {
                format!("The index on this PC belongs to a different Telegram account, so nothing is added to it. Log in with that account, or {way_out}.")
            }
            OpenError::Mismatch(_) => {
                format!("The index on this PC was made with a different version of the search model and cannot be continued. To index again, {way_out}.")
            }
            OpenError::Unidentified => {
                format!("The index on this PC does not say which account and model made it, so nothing is added to it. To index again, {way_out}.")
            }
        })
    }
}

fn database_failure(error: rusqlite::Error) -> Halt {
    use rusqlite::ErrorCode;
    eprintln!("indexer: database error: {error}");
    let reason = match error.sqlite_error_code() {
        Some(ErrorCode::DiskFull) => "The disk that holds the index is full. Free some space; indexing will try again in a minute.",
        Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked) => "The index is being used by another program. Close it; indexing will try again in a minute.",
        Some(ErrorCode::DatabaseCorrupt | ErrorCode::NotADatabase) => "The index file is damaged. Stop Seekgram and move the index folder somewhere else to start a new index.",
        Some(ErrorCode::ReadOnly | ErrorCode::CannotOpen | ErrorCode::PermissionDenied) => "Seekgram cannot write to its index folder. Check that the folder is not read-only.",
        _ => "The index could not be updated because of an unexpected database error. Indexing will try again in a minute.",
    };
    Halt::Failed(reason.to_owned())
}

fn open_database(account_id: i64) -> Result<Connection, OpenError> {
    std::fs::create_dir_all(paths::index_dir()).map_err(OpenError::Folder)?;
    let mut db = Connection::open(paths::index_dir().join(DATABASE))?;
    db.busy_timeout(Duration::from_secs(10))?;
    prepare(&mut db, account_id)?;
    Ok(db)
}

/// Creates the tables and makes sure the index is this account's and this model's.
/// A fresh index is stamped; a foreign or unidentified one is refused, never adopted.
fn prepare(db: &mut Connection, account_id: i64) -> Result<(), OpenError> {
    db.execute_batch(SCHEMA)?;
    let expected = [
        ("model", IDENTITY.to_owned()),
        ("model_repo", MODEL.to_owned()),
        ("model_revision", REVISION.to_owned()),
        ("processing_version", VERSION.to_owned()),
        ("dimension", "768".to_owned()),
        ("vector_format", "float32-le-normalized".to_owned()),
        ("account_id", account_id.to_string()),
        ("audio_chunk_seconds", CHUNK_SECONDS.to_string()),
        ("video_chunk_seconds", CHUNK_SECONDS.to_string()),
    ];
    let transaction = db.transaction()?;
    for (key, value) in expected {
        let stored: Option<String> = transaction.query_row("SELECT value FROM metadata WHERE key=?1", [key], |row| row.get(0)).optional()?;
        match stored {
            Some(stored) if stored != value => return Err(OpenError::Mismatch(key)),
            Some(_) => {}
            None => {
                if transaction.query_row("SELECT 1 FROM items LIMIT 1", [], |_| Ok(())).optional()?.is_some() {
                    return Err(OpenError::Unidentified);
                }
                transaction.execute("INSERT INTO metadata VALUES (?1,?2)", params![key, value])?;
            }
        }
    }
    transaction.commit()?;
    Ok(())
}

fn checkpoint(db: &Connection, chat_id: i64) -> rusqlite::Result<i32> {
    let stored = db.query_row("SELECT message_id FROM checkpoints WHERE chat_id=?1", [chat_id], |row| row.get(0)).optional()?;
    Ok(stored.unwrap_or(0))
}

fn retryable_failures(db: &Connection, chat_id: i64) -> rusqlite::Result<Vec<i32>> {
    let mut statement = db.prepare("SELECT message_id FROM failures WHERE chat_id=?1 AND attempts < ?2 ORDER BY message_id")?;
    statement.query_map(params![chat_id, MAX_ATTEMPTS], |row| row.get(0))?.collect()
}

fn record_failure(db: &Connection, chat_id: i64, message_id: i32, code: &str) -> rusqlite::Result<()> {
    db.execute(
        "INSERT INTO failures VALUES(?1,?2,?3,1,strftime('%Y-%m-%dT%H:%M:%f+00:00','now')) ON CONFLICT(chat_id,message_id)
         DO UPDATE SET error_type=excluded.error_type,attempts=attempts+1,updated_at=excluded.updated_at",
        params![chat_id, message_id, code],
    )?;
    Ok(())
}

fn clear_failure(db: &Connection, chat_id: i64, message_id: i32) -> rusqlite::Result<()> {
    db.execute("DELETE FROM failures WHERE chat_id=?1 AND message_id=?2", params![chat_id, message_id])?;
    Ok(())
}

/// Notes which file a message carries. When the message was edited to carry another
/// one, the vectors of the old file are dropped before the new one is indexed.
fn adopt_source(db: &mut Connection, chat_id: i64, message_id: i32, signature: &str) -> rusqlite::Result<()> {
    let transaction = db.transaction()?;
    let known: Option<String> = transaction
        .query_row("SELECT signature FROM sources WHERE chat_id=?1 AND message_id=?2", params![chat_id, message_id], |row| row.get(0))
        .optional()?;
    if known.is_some_and(|known| known != signature) {
        transaction.execute("DELETE FROM items WHERE chat_id=?1 AND message_id=?2", params![chat_id, message_id])?;
    }
    transaction.execute("INSERT OR REPLACE INTO sources VALUES (?1,?2,?3)", params![chat_id, message_id, signature])?;
    transaction.commit()
}

fn has_item(db: &Connection, chat_id: i64, message_id: i32, modality: &str, segment: i64) -> rusqlite::Result<bool> {
    let found = db
        .query_row(
            "SELECT 1 FROM items WHERE chat_id=?1 AND message_id=?2 AND modality=?3 AND segment=?4",
            params![chat_id, message_id, modality, segment],
            |_| Ok(()),
        )
        .optional()?;
    Ok(found.is_some())
}

/// One vector on its way into `items`. `span` is the chunk's start and end in seconds;
/// images have none.
struct Item {
    chat_id: i64,
    message_id: i32,
    modality: &'static str,
    segment: i64,
    span: Option<(f64, f64)>,
    vector: Vec<u8>,
}

fn insert_item(db: &Connection, item: &Item) -> rusqlite::Result<()> {
    db.execute(
        "INSERT INTO items VALUES (?1,?2,?3,?4,?5,?6,?7)",
        params![item.chat_id, item.message_id, item.segment, item.modality, item.span.map(|span| span.0), item.span.map(|span| span.1), item.vector],
    )?;
    Ok(())
}

/// Moves a chat's checkpoint over messages that are done, in one commit together with
/// the outcome of those that needed no download.
fn advance_checkpoint(db: &mut Connection, chat_id: i64, advance: &Advance) -> rusqlite::Result<()> {
    let transaction = db.transaction()?;
    for &(message_id, failure) in &advance.settled {
        match failure {
            Some(code) => record_failure(&transaction, chat_id, message_id, code)?,
            None => clear_failure(&transaction, chat_id, message_id)?,
        }
    }
    transaction.execute("INSERT OR REPLACE INTO checkpoints VALUES (?1,?2)", params![chat_id, advance.checkpoint])?;
    transaction.commit()
}

type DbJob = Box<dyn FnOnce(&mut Connection) + Send>;

/// The read-write connection, on a thread of its own so that commits (an fsync each)
/// never stall the async runtime. Jobs run one at a time, in the order they were sent.
struct Db {
    jobs: mpsc::Sender<DbJob>,
}

impl Db {
    async fn open(account_id: i64) -> Result<Self, Halt> {
        let (jobs, queue) = mpsc::channel::<DbJob>();
        let (opened, outcome) = oneshot::channel();
        let thread = std::thread::Builder::new().name("index-db".to_owned()).spawn(move || {
            let mut db = match open_database(account_id) {
                Ok(db) => db,
                Err(error) => return drop(opened.send(Err(error))),
            };
            if opened.send(Ok(())).is_ok() {
                for job in queue {
                    job(&mut db);
                }
            }
        });
        if let Err(error) = thread {
            eprintln!("indexer: the database thread could not be started: {error}");
            return Err(Halt::Failed(INTERNAL.to_owned()));
        }
        match outcome.await {
            Ok(Ok(())) => Ok(Self { jobs }),
            Ok(Err(error)) => Err(error.into()),
            Err(_) => Err(Halt::Failed(INTERNAL.to_owned())),
        }
    }

    /// Runs `work` on the database thread. It still runs to completion when the caller
    /// is cancelled meanwhile, so a transaction is never left half done.
    async fn call<T: Send + 'static>(&self, work: impl FnOnce(&mut Connection) -> rusqlite::Result<T> + Send + 'static) -> Result<T, Halt> {
        let (reply, answer) = oneshot::channel();
        let job: DbJob = Box::new(move |db| drop(reply.send(work(db))));
        if self.jobs.send(job).is_err() {
            return Err(Halt::Failed(INTERNAL.to_owned()));
        }
        answer.await.map_err(|_| Halt::Failed(INTERNAL.to_owned()))?.map_err(database_failure)
    }

    /// Waits until everything already queued has been written.
    async fn drain(&self) {
        let _ = self.call(|_| Ok(())).await;
    }
}

/// The pause Telegram asked for, shared by every task and kept on disk so that a
/// restart does not walk straight back into it.
struct Flood {
    shared: Arc<Shared>,
    /// Unix time before which nothing new is asked of Telegram.
    until: Mutex<f64>,
}

impl Flood {
    fn file() -> PathBuf {
        paths::index_dir().join("flood-wait.json")
    }

    fn restore(shared: Arc<Shared>) -> Result<Self, Halt> {
        let until = match std::fs::read(Self::file()) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0.0,
            read => {
                let saved = read.ok().and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
                match saved.and_then(|saved| saved["until_unix"].as_f64()).filter(|until| until.is_finite()) {
                    Some(until) => until,
                    None => {
                        let _ = std::fs::remove_file(Self::file());
                        return Err(Halt::Failed(PAUSE_UNREADABLE.to_owned()));
                    }
                }
            }
        };
        Ok(Self { shared, until: Mutex::new(until) })
    }

    fn until(&self) -> MutexGuard<'_, f64> {
        self.until.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn remaining(&self) -> f64 {
        *self.until() - unix_time()
    }

    fn report(&self, seconds: u64) {
        let mut until = self.until();
        // Several messages can be told to wait at once; all of them sit out the longest.
        *until = until.max(unix_time() + seconds.max(1) as f64);
        let saved = json!({"until_unix": *until}).to_string();
        if let Err(error) = config::write_atomic(&Self::file(), saved.as_bytes()) {
            eprintln!("indexer: the Telegram pause could not be saved: {error}");
        }
    }

    /// Sits out the pause, counting down in the status.
    async fn wait(&self) {
        loop {
            let remaining = self.remaining();
            if remaining <= 0.0 {
                break;
            }
            self.shared.update(|status| {
                status.indexer.state = IndexerState::FloodWait;
                status.indexer.reason = None;
                status.indexer.flood_wait_seconds = Some(remaining.ceil() as u64);
            });
            tokio::time::sleep(Duration::from_secs_f64(remaining.min(1.0))).await;
        }
        self.forget();
        self.shared.update(|status| {
            if status.indexer.state == IndexerState::FloodWait {
                status.indexer.state = IndexerState::Running;
                status.indexer.flood_wait_seconds = None;
            }
        });
    }

    /// Removes the saved pause once it is over. Checked under the lock, so a longer
    /// pause reported at this very moment keeps its file.
    fn forget(&self) {
        let until = self.until();
        if *until <= unix_time() {
            let _ = std::fs::remove_file(Self::file());
        }
    }

    /// Holds a new download back while a pause is on. The task that was told to wait
    /// shows the countdown; this one only stays out of Telegram's way.
    async fn clear(&self) {
        loop {
            let remaining = self.remaining();
            if remaining <= 0.0 {
                return;
            }
            tokio::time::sleep(Duration::from_secs_f64(remaining.min(1.0))).await;
        }
    }

    /// Sorts out one answer from Telegram: the value, `None` after sitting out a pause
    /// it asked for (ask again), or the code of what went wrong.
    async fn absorb<T>(&self, outcome: Result<Result<T, InvocationError>, Elapsed>) -> Result<Option<T>, Fault> {
        match outcome {
            Ok(Ok(value)) => Ok(Some(value)),
            Ok(Err(error)) => match telegram_fault(&error) {
                Fault::Flood(seconds) => {
                    self.report(seconds);
                    guard(&self.shared, self.wait()).await?;
                    Ok(None)
                }
                fault => Err(fault),
            },
            Err(_) => Err(code("timeout:telegram")),
        }
    }
}

fn telegram_fault(error: &InvocationError) -> Fault {
    match error {
        InvocationError::Rpc(rpc) => match rpc.value {
            Some(seconds) if matches!(rpc.name.as_str(), "FLOOD_WAIT" | "FLOOD_PREMIUM_WAIT") => Fault::Flood(u64::from(seconds)),
            _ => Fault::Code(format!("telegram:{}", rpc.name.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_').collect::<String>())),
        },
        _ => code("telegram:connection"),
    }
}

/// A Telegram failure with no message or chat to record it against: the session ends
/// and is tried again later.
fn telegram_failure(fault: Fault) -> Halt {
    match fault {
        Fault::Halt(halt) => halt,
        fault => {
            eprintln!("indexer: a Telegram request failed: {fault:?}");
            Halt::Failed(TELEGRAM_SILENT.to_owned())
        }
    }
}

/// What a message offers for indexing.
struct Target {
    kind: Kind,
    /// Identifies the file; a different one on a later pass means the message was edited.
    signature: String,
    /// Size of the document, when the message has one.
    size: Option<i64>,
    /// `None` when Telegram only has a placeholder for the file.
    download: Option<Downloadable>,
}

/// The photo and the document a message exposes, by Telethon's rules: its own media,
/// a web page preview's, or the photo of a "chat photo changed" service message.
fn attachments<'a>(
    media: Option<&'a tl::enums::MessageMedia>,
    action: Option<&'a tl::enums::MessageAction>,
) -> (Option<&'a tl::enums::Photo>, Option<&'a tl::types::Document>) {
    use tl::enums::{Document as D, MessageAction as A, MessageMedia as M, Photo as P, WebPage as W};
    let real_photo = |photo: &'a Option<tl::enums::Photo>| photo.as_ref().filter(|photo| matches!(photo, P::Photo(_)));
    let real_document = |document: &'a Option<tl::enums::Document>| match document {
        Some(D::Document(document)) => Some(document),
        _ => None,
    };
    match media {
        None | Some(M::Empty) => match action {
            Some(A::ChatEditPhoto(edit)) => (Some(&edit.photo), None),
            _ => (None, None),
        },
        Some(M::Photo(media)) => (real_photo(&media.photo), None),
        Some(M::Document(media)) => (None, real_document(&media.document)),
        Some(M::WebPage(media)) => match &media.webpage {
            W::Page(page) => (real_photo(&page.photo), real_document(&page.document)),
            _ => (None, None),
        },
        Some(_) => (None, None),
    }
}

/// `image`, `video` or `audio` for a document, by MIME type first and then by the
/// attributes Telegram attached, exactly as the prototype decided it.
fn media_kind(mime: &str, video_attribute: bool, audio_attribute: bool) -> Option<Kind> {
    let mime = mime.to_lowercase();
    if mime.starts_with("image/") {
        Some(Kind::Image)
    } else if mime.starts_with("video/") || video_attribute {
        Some(Kind::Video)
    } else if mime.starts_with("audio/") || audio_attribute {
        Some(Kind::Audio)
    } else {
        None
    }
}

/// Byte-identical to the prototype's `json.dumps([kind, id, size], separators=(',', ':'))`.
fn signature(kind: Kind, id: i64, size: Option<i64>) -> String {
    match size {
        Some(size) => format!("[\"{}\",{id},{size}]", kind.as_str()),
        None => format!("[\"{}\",{id},null]", kind.as_str()),
    }
}

fn classify(message: &Message) -> Option<Target> {
    use tl::enums::DocumentAttribute as Attribute;
    let (photo, document) = attachments(message.raw.media.as_ref(), message.raw_action.as_ref());
    let kind = match (photo, document) {
        (Some(_), _) => Kind::Image,
        (None, Some(document)) => media_kind(
            &document.mime_type,
            document.attributes.iter().any(|attribute| matches!(attribute, Attribute::Video(_))),
            document.attributes.iter().any(|attribute| matches!(attribute, Attribute::Audio(_))),
        )?,
        (None, None) => return None,
    };
    // A preview with both a photo and a document is indexed as the photo but signed
    // with the document; existing indexes hold such rows, so it stays that way.
    let id = document.map(|document| document.id).or_else(|| photo.map(tl::enums::Photo::id))?;
    let size = document.map(|document| document.size);
    let download = match (kind, photo) {
        (Kind::Image, Some(photo)) => photo_download(photo),
        _ => document.map(document_download),
    };
    Some(Target { kind, signature: signature(kind, id, size), size, download })
}

/// The last size Telegram lists for the photo, which is the one Telethon downloads;
/// picking the same file keeps new vectors equal to the prototype's for the same photo.
fn photo_download(photo: &tl::enums::Photo) -> Option<Downloadable> {
    let last = Photo::from_raw(photo.clone()).thumbs().pop()?;
    matches!(last, PhotoSize::Size(_) | PhotoSize::Progressive(_) | PhotoSize::Cached(_)).then(|| Downloadable::PhotoSize(last))
}

fn document_download(document: &tl::types::Document) -> Downloadable {
    Downloadable::Media(Media::Document(Document::from_raw_media(tl::types::MessageMediaDocument {
        nopremium: false,
        spoiler: false,
        video: false,
        round: false,
        voice: false,
        document: Some(tl::enums::Document::Document(document.clone())),
        alt_document: None,
        ttl_seconds: None,
    })))
}

/// Media that Telegram deletes on a timer: view-once photos and videos, and anything
/// in a chat with auto-delete. Its vectors would outlive it, so it is never indexed.
fn self_destructing(message: &tl::types::Message) -> bool {
    use tl::enums::MessageMedia as M;
    match &message.media {
        None | Some(M::Empty) => false,
        Some(M::Photo(media)) => media.ttl_seconds.is_some() || message.ttl_period.is_some(),
        Some(M::Document(media)) => media.ttl_seconds.is_some() || message.ttl_period.is_some(),
        Some(_) => message.ttl_period.is_some(),
    }
}

/// What ffprobe found in a downloaded file.
#[derive(Debug, PartialEq)]
struct Probe {
    duration: f64,
    video: bool,
    audio: bool,
    /// Codec of the first video stream.
    video_codec: Option<String>,
}

impl Probe {
    fn parse(output: &[u8]) -> Result<Self, Fault> {
        let unreadable = || code("invalid probe output");
        let info: Value = serde_json::from_slice(output).map_err(|_| unreadable())?;
        let streams = info["streams"].as_array().map(Vec::as_slice).unwrap_or_default();
        let mut duration = 0.0_f64;
        for stated in std::iter::once(&info["format"]["duration"]).chain(streams.iter().map(|stream| &stream["duration"])) {
            let seconds = match stated {
                Value::Null => continue,
                Value::String(text) if text == "N/A" => continue,
                Value::String(text) => text.trim().parse::<f64>().map_err(|_| unreadable())?,
                other => other.as_f64().ok_or_else(unreadable)?,
            };
            duration = duration.max(seconds);
        }
        if !duration.is_finite() || duration <= 0.0 {
            return Err(code("media duration unavailable"));
        }
        let of_type = |wanted: &'static str| streams.iter().filter(move |stream| stream["codec_type"] == wanted);
        Ok(Self {
            duration,
            video: of_type("video").next().is_some(),
            audio: of_type("audio").next().is_some(),
            video_codec: of_type("video").next().and_then(|stream| stream["codec_name"].as_str()).map(str::to_owned),
        })
    }

    /// What to embed from a file indexed as `kind`: a video's picture and its sound, or
    /// an audio file's sound.
    fn modalities(&self, kind: Kind) -> Result<Vec<Kind>, Fault> {
        let mut modalities = Vec::new();
        if kind == Kind::Video {
            if !self.video {
                return Err(code("video stream unavailable"));
            }
            modalities.push(Kind::Video);
        }
        if self.audio {
            modalities.push(Kind::Audio);
        } else if kind == Kind::Audio {
            return Err(code("audio stream unavailable"));
        }
        Ok(modalities)
    }
}

fn chunk_count(duration: f64) -> Result<i64, Fault> {
    let count = (duration / CHUNK_SECONDS as f64).ceil();
    if count > MAX_CHUNKS as f64 {
        return Err(code("media exceeds configured chunk limit"));
    }
    Ok(count as i64)
}

/// Start and end of one chunk, in seconds from the start of the file.
fn chunk_span(duration: f64, segment: i64) -> (f64, f64) {
    let start = (segment * CHUNK_SECONDS) as f64;
    (start, duration.min(start + CHUNK_SECONDS as f64))
}

/// The ffmpeg arguments that cut one chunk out of `source`: mono 16 kHz WAV for sound,
/// H.264 for picture. `repair_vp9` overrides colour metadata the decoder rejects.
fn extraction(source: &Path, chunk: &Path, modality: Kind, start: f64, length: f64, repair_vp9: bool) -> Vec<OsString> {
    let mut arguments: Vec<OsString> = ["-nostdin", "-v", "error", "-y", "-ss"].map(OsString::from).into();
    arguments.push(start.to_string().into());
    if repair_vp9 {
        arguments.extend(["-bsf:v", "vp9_metadata=color_space=bt709"].map(OsString::from));
    }
    arguments.push("-i".into());
    arguments.push(source.into());
    arguments.push("-t".into());
    arguments.push(length.to_string().into());
    let encoding: &[&str] = if modality == Kind::Audio {
        &["-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]
    } else {
        &["-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    };
    arguments.extend(encoding.iter().map(OsString::from));
    arguments.push(chunk.into());
    arguments
}

enum ToolFailure {
    /// The program is missing or broken; no file is to blame.
    NotStarted,
    /// It ran and rejected the file.
    Refused,
    TimedOut,
}

impl ToolFailure {
    fn fault(self, phase: &str) -> Fault {
        match self {
            ToolFailure::NotStarted => Halt::Failed(TOOLS_MISSING.to_owned()).into(),
            ToolFailure::Refused => Fault::Code(format!("{phase} failed")),
            ToolFailure::TimedOut => Fault::Code(format!("timeout:{phase}")),
        }
    }
}

/// Runs ffmpeg or ffprobe and returns what it printed. The process dies with this
/// future, so cancelled work leaves nothing running.
async fn tool(program: PathBuf, arguments: Vec<OsString>) -> Result<Vec<u8>, ToolFailure> {
    let mut command = Command::new(program);
    command.args(arguments).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).kill_on_drop(true);
    let child = hidden(&mut command).spawn().map_err(|error| {
        eprintln!("indexer: a media tool did not start: {error}");
        ToolFailure::NotStarted
    })?;
    match timeout(TOOL_TIMEOUT, child.wait_with_output()).await {
        Ok(Ok(output)) if output.status.success() => Ok(output.stdout),
        Ok(_) => Err(ToolFailure::Refused),
        Err(_) => Err(ToolFailure::TimedOut),
    }
}

fn scratch_failure(error: std::io::Error) -> Halt {
    eprintln!("indexer: temporary media could not be written: {error}");
    Halt::Failed(SCRATCH_FAILED.to_owned())
}

/// A fresh directory for one message's download and chunks, removed when dropped:
/// on success, on failure and when the work is cancelled.
struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn create() -> Result<Self, Halt> {
        // Absolute, because the worker refuses relative media paths.
        let path = std::path::absolute(paths::temp_dir()).map_err(scratch_failure)?.join(Uuid::new_v4().simple().to_string());
        std::fs::create_dir_all(&path).map_err(scratch_failure)?;
        Ok(Self { path })
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if std::fs::remove_dir_all(&self.path).is_ok() {
            return;
        }
        // A killed ffmpeg or an unfinished write can hold a file open a moment longer.
        let path = std::mem::take(&mut self.path);
        std::thread::spawn(move || {
            for _ in 0..10 {
                std::thread::sleep(Duration::from_millis(500));
                if std::fs::remove_dir_all(&path).is_ok() || !path.exists() {
                    return;
                }
            }
        });
    }
}

/// One message ahead of the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Slot {
    /// Media whose task is still downloading or embedding.
    Running,
    /// Media whose task has ended and recorded its own outcome.
    Finished,
    /// Needed no download: nothing to index, or `Some` failure to record.
    Settled(Option<&'static str>),
}

/// Messages are indexed concurrently, but the checkpoint may only pass a message once
/// it and every older one are done, so a crash never skips one still in flight.
#[derive(Default)]
struct Window {
    slots: VecDeque<(i32, Slot)>,
}

/// The oldest messages of a window that are all done.
#[derive(Debug, PartialEq)]
struct Advance {
    checkpoint: i32,
    messages: u64,
    settled: Vec<(i32, Option<&'static str>)>,
}

impl Window {
    fn len(&self) -> usize {
        self.slots.len()
    }

    fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Media messages not yet behind the checkpoint, finished or not. Bounding this,
    /// rather than only the running ones, bounds what is downloaded again after a crash.
    fn media(&self) -> usize {
        self.slots.iter().filter(|(_, slot)| matches!(slot, Slot::Running | Slot::Finished)).count()
    }

    fn push(&mut self, id: i32, slot: Slot) {
        self.slots.push_back((id, slot));
    }

    fn finish(&mut self, id: i32) {
        if let Some((_, slot)) = self.slots.iter_mut().find(|(slot_id, _)| *slot_id == id) {
            *slot = Slot::Finished;
        }
    }

    /// Takes the done messages off the old end, if the oldest is done.
    fn advance(&mut self) -> Option<Advance> {
        let done = self.slots.iter().take_while(|(_, slot)| *slot != Slot::Running).count();
        let checkpoint = self.slots.get(done.checked_sub(1)?)?.0;
        let settled = self
            .slots
            .drain(..done)
            .filter_map(|(id, slot)| match slot {
                Slot::Settled(failure) => Some((id, failure)),
                _ => None,
            })
            .collect();
        Some(Advance { checkpoint, messages: done as u64, settled })
    }
}

/// A chat's history read oldest to newest. grammers only iterates newest first, so
/// this pages upward the way Telethon's `reverse=True` does: ask for the page that
/// ends just above the last message seen, by a negative `add_offset`.
struct History {
    request: tl::functions::messages::GetHistory,
    /// Highest message ID handed out; a page never goes back below it.
    last: i32,
    finished: bool,
}

impl History {
    fn new(peer: tl::enums::InputPeer, after: i32) -> Self {
        let request = tl::functions::messages::GetHistory {
            peer,
            offset_id: if after > 0 { after + 1 } else { 1 },
            offset_date: 0,
            add_offset: -PAGE,
            limit: PAGE,
            max_id: 0,
            min_id: 0,
            hash: 0,
        };
        Self { request, last: after, finished: false }
    }

    /// Takes one response and returns its messages, oldest first.
    fn accept(&mut self, client: &Client, response: tl::enums::messages::Messages) -> Vec<Message> {
        use tl::enums::messages::Messages;
        // A plain `Messages` is the whole remaining history rather than a slice of it.
        let (messages, users, chats, whole) = match response {
            Messages::Messages(page) => (page.messages, page.users, page.chats, true),
            Messages::Slice(page) => (page.messages, page.users, page.chats, false),
            Messages::ChannelMessages(page) => (page.messages, page.users, page.chats, false),
            Messages::NotModified(_) => (Vec::new(), Vec::new(), Vec::new(), true),
        };
        let chats = ChatMap::new(users, chats);
        let mut page = Vec::with_capacity(messages.len());
        for raw in messages.into_iter().rev() {
            let Some(message) = Message::from_raw(client, raw, &chats) else { continue };
            if message.id() > self.last {
                self.last = message.id();
                page.push(message);
            }
        }
        match page.last() {
            Some(newest) if !whole => {
                self.request.offset_id = newest.id() + 1;
                self.request.offset_date = newest.raw.date;
            }
            // Telegram returns at least one message while there is more to fetch.
            _ => self.finished = true,
        }
        page
    }
}

/// A chat to index.
struct ChatRef {
    /// Marked ID, as stored in the index.
    id: i64,
    title: String,
    peer: PackedChat,
    /// ID of the chat's newest message when the chats were listed.
    top: i32,
}

/// One stretch of indexing, from the gate opening until it closes or something fails.
struct Session {
    shared: Arc<Shared>,
    workers: Arc<Workers>,
    progress: Arc<Progress>,
    client: Client,
    flood: Flood,
    db: Db,
}

/// Indexes until it has to stop, and says why it stopped.
async fn index(shared: &Arc<Shared>, workers: &Arc<Workers>, progress: &Arc<Progress>) -> Halt {
    let session = match Session::start(shared, workers, progress).await {
        Ok(session) => Arc::new(session),
        Err(halt) => return halt,
    };
    set_state(shared, IndexerState::Running, None);
    let halt = session.passes().await;
    // Writes queued by work that was just cancelled land before anything opens the index again.
    session.db.drain().await;
    halt
}

impl Session {
    async fn start(shared: &Arc<Shared>, workers: &Arc<Workers>, progress: &Arc<Progress>) -> Result<Self, Halt> {
        let client = shared.telegram().await.map_err(Halt::Failed)?.client.clone();
        let flood = Flood::restore(shared.clone())?;
        guard(shared, flood.wait()).await?;
        let account_id = loop {
            let outcome = guard(shared, timeout(NETWORK_TIMEOUT, client.get_me())).await?;
            if let Some(me) = flood.absorb(outcome).await.map_err(telegram_failure)? {
                break me.id();
            }
        };
        let db = guard(shared, Db::open(account_id)).await??;
        Ok(Self { shared: shared.clone(), workers: workers.clone(), progress: progress.clone(), client, flood, db })
    }

    async fn guard<T>(&self, work: impl Future<Output = T>) -> Result<T, Halt> {
        guard(&self.shared, work).await
    }

    /// Passes over all chats, again and again, until something stops it.
    async fn passes(self: &Arc<Self>) -> Halt {
        loop {
            if let Err(halt) = self.pass().await {
                return halt;
            }
            set_state(&self.shared, IndexerState::Idle, None);
            let mut changed = self.shared.changed.subscribe();
            tokio::select! {
                _ = tokio::time::sleep(PASS_INTERVAL) => {}
                _ = changed.changed() => {}
                _ = self.shared.stopped() => {}
            }
            if let Some(halt) = halted(&self.shared) {
                return halt;
            }
            set_state(&self.shared, IndexerState::Running, None);
        }
    }

    async fn pass(self: &Arc<Self>) -> Result<(), Halt> {
        let chats = self.chats().await.map_err(telegram_failure)?;
        self.progress.set(|tally| {
            tally.chats_done = 0;
            tally.chats_total = chats.len() as u32;
        });
        for chat in &chats {
            loop {
                match self.index_chat(chat).await {
                    Ok(()) => break,
                    Err(Fault::Halt(halt)) => return Err(halt),
                    Err(Fault::Flood(seconds)) => {
                        self.flood.report(seconds);
                        self.guard(self.flood.wait()).await?;
                    }
                    // The rest of this chat waits for the next pass; its checkpoint only
                    // covers what was finished.
                    Err(Fault::Code(code)) => {
                        eprintln!("indexer: a chat could not be read ({code})");
                        self.progress.note(|tally| tally.errors += 1);
                        break;
                    }
                }
            }
            self.progress.set(|tally| tally.chats_done += 1);
        }
        self.progress.set(|tally| tally.chat = None);
        Ok(())
    }

    /// Every dialog of the account, once each, except the private search group: it only
    /// holds copies of results, and indexing those would return them as results again.
    async fn chats(&self) -> Result<Vec<ChatRef>, Fault> {
        let mut dialogs = self.client.iter_dialogs();
        let mut seen = HashSet::new();
        let mut chats = Vec::new();
        loop {
            let outcome = self.guard(timeout(NETWORK_TIMEOUT, dialogs.next())).await?;
            let dialog = match self.flood.absorb(outcome).await? {
                Some(Some(dialog)) => dialog,
                Some(None) => return Ok(chats),
                None => continue,
            };
            let id = marked_id(dialog.chat());
            if self.shared.settings().search_chat != Some(id) && seen.insert(id) {
                chats.push(ChatRef { id, title: dialog.chat().name().to_owned(), peer: dialog.chat().pack(), top: dialog.raw.top_message() });
            }
        }
    }

    async fn index_chat(self: &Arc<Self>, chat: &ChatRef) -> Result<(), Fault> {
        let chat_id = chat.id;
        let scanned = self.db.call(move |db| checkpoint(db, chat_id)).await?;
        let current = CurrentChat { id: chat_id, title: chat.title.clone(), message_id: scanned, top_message_id: chat.top };
        self.progress.set(|tally| tally.chat = Some(current));
        self.retry_failures(chat).await?;
        // Nothing newer than the checkpoint: no need to ask Telegram for an empty page
        // per chat on every pass.
        if chat.top > 0 && chat.top <= scanned {
            return Ok(());
        }
        let mut tasks = JoinSet::new();
        let outcome = self.scan(chat, scanned, &mut tasks).await;
        if matches!(outcome, Err(Fault::Halt(Halt::Shutdown | Halt::Gate))) {
            set_state(&self.shared, IndexerState::Stopping, None);
        }
        // Whatever is still in flight is cancelled; the checkpoint never passed it.
        tasks.shutdown().await;
        outcome
    }

    /// Tries the chat's recorded failures again, one at a time.
    async fn retry_failures(&self, chat: &ChatRef) -> Result<(), Fault> {
        let chat_id = chat.id;
        for id in self.db.call(move |db| retryable_failures(db, chat_id)).await? {
            let fetched = loop {
                let outcome = self.guard(timeout(NETWORK_TIMEOUT, self.client.get_messages_by_id(chat.peer, &[id]))).await?;
                match self.flood.absorb(outcome).await {
                    Ok(Some(found)) => break Ok(found.into_iter().next().flatten()),
                    Ok(None) => {}
                    Err(Fault::Code(code)) => break Err(code),
                    Err(fault) => return Err(fault),
                }
            };
            let target = match fetched {
                Err(code) => Err(code),
                Ok(None) => Err("message unavailable".to_owned()),
                Ok(Some(message)) => match classify(&message) {
                    None => Err("previously failed media unavailable".to_owned()),
                    Some(_) if self_destructing(&message.raw) => Err(SELF_DESTRUCTING.to_owned()),
                    Some(target) => Ok(target),
                },
            };
            match target {
                Ok(target) => self.guard(self.index_media(chat_id, id, target)).await??,
                Err(code) => self.fail(chat_id, id, code).await?,
            }
            self.progress.note(|tally| tally.messages_scanned += 1);
        }
        Ok(())
    }

    /// Indexes the chat's messages above its checkpoint, oldest first, several media
    /// messages at a time. Their tasks are left in `tasks` for the caller to end.
    async fn scan(self: &Arc<Self>, chat: &ChatRef, after: i32, tasks: &mut JoinSet<(i32, Result<(), Halt>)>) -> Result<(), Fault> {
        let mut history = History::new(chat.peer.to_input_peer(), after);
        let mut fetched = VecDeque::new();
        let mut window = Window::default();
        loop {
            let concurrency = usize::from(self.shared.read(|status| status.settings.concurrency).clamp(1, 16));
            let room = |window: &Window| window.len() < WINDOW && window.media() < concurrency;
            while room(&window) {
                let Some(message) = fetched.pop_front() else { break };
                self.admit(chat.id, message, &mut window, tasks);
            }
            if let Some(advance) = window.advance() {
                self.commit(chat.id, advance).await?;
            } else if fetched.is_empty() && !history.finished && room(&window) {
                fetched.extend(self.page(&mut history).await?);
            } else if window.is_empty() {
                return Ok(());
            } else {
                match self.guard(tasks.join_next()).await? {
                    Some(Ok((id, Ok(())))) => window.finish(id),
                    Some(Ok((_, Err(halt)))) => return Err(halt.into()),
                    Some(Err(error)) => {
                        eprintln!("indexer: a message task ended abnormally: {error}");
                        return Err(Halt::Failed(INTERNAL.to_owned()).into());
                    }
                    None => return Err(Halt::Failed(INTERNAL.to_owned()).into()),
                }
            }
        }
    }

    async fn page(&self, history: &mut History) -> Result<Vec<Message>, Fault> {
        loop {
            let outcome = self.guard(timeout(NETWORK_TIMEOUT, self.client.invoke(&history.request))).await?;
            if let Some(response) = self.flood.absorb(outcome).await? {
                return Ok(history.accept(&self.client, response));
            }
        }
    }

    /// Puts a message in the window, starting a task for it when it carries media.
    fn admit(self: &Arc<Self>, chat_id: i64, message: Message, window: &mut Window, tasks: &mut JoinSet<(i32, Result<(), Halt>)>) {
        let id = message.id();
        if self_destructing(&message.raw) {
            return window.push(id, Slot::Settled(Some(SELF_DESTRUCTING)));
        }
        let Some(target) = classify(&message) else { return window.push(id, Slot::Settled(None)) };
        window.push(id, Slot::Running);
        let session = self.clone();
        tasks.spawn(async move { (id, session.index_media(chat_id, id, target).await) });
    }

    async fn commit(&self, chat_id: i64, advance: Advance) -> Result<(), Halt> {
        let (checkpoint, messages) = (advance.checkpoint, advance.messages);
        let failures = advance.settled.iter().filter(|(_, failure)| failure.is_some()).count() as u64;
        self.db.call(move |db| advance_checkpoint(db, chat_id, &advance)).await?;
        self.progress.note(|tally| {
            tally.messages_scanned += messages;
            tally.errors += failures;
            if let Some(chat) = &mut tally.chat {
                chat.message_id = checkpoint;
            }
        });
        Ok(())
    }

    async fn fail(&self, chat_id: i64, id: i32, code: String) -> Result<(), Halt> {
        self.db.call(move |db| record_failure(db, chat_id, id, &code)).await?;
        self.progress.note(|tally| tally.errors += 1);
        Ok(())
    }

    /// Indexes one media message to the end: success, or a failure recorded against it.
    /// Only what stops the whole indexer comes back as an error.
    async fn index_media(&self, chat_id: i64, id: i32, target: Target) -> Result<(), Halt> {
        let _in_flight = self.progress.in_flight();
        loop {
            match self.index_media_once(chat_id, id, &target).await {
                Ok(()) => return self.db.call(move |db| clear_failure(db, chat_id, id)).await,
                // The download starts over after the pause; chunks already stored are kept.
                Err(Fault::Flood(seconds)) => {
                    self.flood.report(seconds);
                    self.flood.wait().await;
                }
                Err(Fault::Halt(halt)) => return Err(halt),
                Err(Fault::Code(code)) => return self.fail(chat_id, id, code).await,
            }
        }
    }

    async fn index_media_once(&self, chat_id: i64, id: i32, target: &Target) -> Result<(), Fault> {
        let signature = target.signature.clone();
        self.db.call(move |db| adopt_source(db, chat_id, id, &signature)).await?;
        let scratch = Scratch::create()?;
        let source = scratch.path.join("source");
        self.download(target, &source).await?;
        if target.kind == Kind::Image {
            if !self.stored(chat_id, id, Kind::Image, 0).await? {
                self.embed(chat_id, id, Kind::Image, 0, &source, None).await?;
            }
            return Ok(());
        }
        let probe = match tool(provision::ffprobe(), probe_arguments(&source)).await {
            Ok(output) => Probe::parse(&output)?,
            Err(failure) => return Err(failure.fault("probing")),
        };
        let modalities = probe.modalities(target.kind)?;
        let count = chunk_count(probe.duration)?;
        for modality in modalities {
            let chunk = scratch.path.join(if modality == Kind::Audio { "chunk.wav" } else { "chunk.mp4" });
            for segment in 0..count {
                if self.stored(chat_id, id, modality, segment).await? {
                    continue;
                }
                let span = chunk_span(probe.duration, segment);
                extract(&source, &chunk, modality, span, probe.video_codec.as_deref()).await?;
                self.embed(chat_id, id, modality, segment, &chunk, Some(span)).await?;
                let _ = tokio::fs::remove_file(&chunk).await;
            }
        }
        Ok(())
    }

    async fn download(&self, target: &Target, destination: &Path) -> Result<(), Fault> {
        let too_large = || code("media exceeds configured byte limit");
        if target.size.is_some_and(|size| size > MAX_MEDIA_BYTES as i64) {
            return Err(too_large());
        }
        self.flood.clear().await;
        let Some(downloadable) = &target.download else { return Err(code("empty or unavailable media")) };
        let mut file = tokio::fs::File::create(destination).await.map_err(scratch_failure)?;
        let mut chunks = self.client.iter_download(downloadable);
        let mut received = 0u64;
        loop {
            let chunk = match timeout(NETWORK_TIMEOUT, chunks.next()).await {
                Ok(Ok(Some(chunk))) => chunk,
                Ok(Ok(None)) => break,
                Ok(Err(error)) => return Err(telegram_fault(&error)),
                Err(_) => return Err(code("timeout:downloading")),
            };
            received += chunk.len() as u64;
            if received > MAX_MEDIA_BYTES {
                return Err(too_large());
            }
            file.write_all(&chunk).await.map_err(scratch_failure)?;
            self.progress.note(|tally| tally.bytes_downloaded += chunk.len() as u64);
        }
        file.flush().await.map_err(scratch_failure)?;
        if received == 0 {
            return Err(code("empty or unavailable media"));
        }
        Ok(())
    }

    async fn stored(&self, chat_id: i64, id: i32, modality: Kind, segment: i64) -> Result<bool, Halt> {
        self.db.call(move |db| has_item(db, chat_id, id, modality.as_str(), segment)).await
    }

    /// Embeds one image or chunk and stores its vector.
    async fn embed(&self, chat_id: i64, id: i32, modality: Kind, segment: i64, file: &Path, span: Option<(f64, f64)>) -> Result<(), Fault> {
        let request = Embed { kind: modality, input: file.to_string_lossy().into_owned(), seconds: span.map(|(start, end)| end - start) };
        let vector = self.workers.embed(request, false).await.map_err(|error| match error {
            EmbedError::Media(code) => Fault::Code(code),
            EmbedError::Worker(reason) => Fault::Halt(Halt::Failed(reason)),
        })?;
        let item = Item {
            chat_id,
            message_id: id,
            modality: modality.as_str(),
            segment,
            span,
            vector: vector.iter().flat_map(|value| value.to_le_bytes()).collect(),
        };
        self.db.call(move |db| insert_item(db, &item)).await?;
        self.progress.note(|tally| {
            tally.items_added += 1;
            tally.rate.record(Instant::now());
        });
        Ok(())
    }
}

fn probe_arguments(source: &Path) -> Vec<OsString> {
    let mut arguments: Vec<OsString> =
        ["-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,duration", "-of", "json"].map(OsString::from).into();
    arguments.push(source.into());
    arguments
}

/// Cuts one chunk out of the downloaded file.
async fn extract(source: &Path, chunk: &Path, modality: Kind, (start, end): (f64, f64), video_codec: Option<&str>) -> Result<(), Fault> {
    let phase = if modality == Kind::Audio { "extracting_audio" } else { "extracting_video" };
    let run = |repair_vp9| tool(provision::ffmpeg(), extraction(source, chunk, modality, start, end - start, repair_vp9));
    let outcome = match run(false).await {
        // Some Telegram VP9 stickers carry reserved colour metadata that the decoder
        // rejects. It is overridden only after the plain extraction has failed, so
        // files that already work keep producing the same frames.
        Err(ToolFailure::Refused) if modality == Kind::Video && video_codec == Some("vp9") => run(true).await,
        outcome => outcome,
    };
    outcome.map(drop).map_err(|failure| failure.fault(phase))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> Connection {
        let mut db = Connection::open_in_memory().unwrap();
        prepare(&mut db, 42).unwrap();
        db
    }

    fn count(db: &Connection, sql: &str) -> i64 {
        db.query_row(sql, [], |row| row.get(0)).unwrap()
    }

    fn item(message_id: i32, modality: &'static str, segment: i64) -> Item {
        Item { chat_id: -100, message_id, modality, segment, span: None, vector: vec![0; 3072] }
    }

    #[test]
    fn signatures_match_python_json() {
        assert_eq!(signature(Kind::Image, 5_240_123_456_789_012_345, None), "[\"image\",5240123456789012345,null]");
        assert_eq!(signature(Kind::Image, 77, Some(1024)), "[\"image\",77,1024]");
        assert_eq!(signature(Kind::Video, -9, Some(123_456_789_012)), "[\"video\",-9,123456789012]");
        assert_eq!(signature(Kind::Audio, 1, Some(0)), "[\"audio\",1,0]");
    }

    #[test]
    fn documents_are_classified_by_mime_then_attributes() {
        assert_eq!(media_kind("image/jpeg", false, false), Some(Kind::Image));
        assert_eq!(media_kind("image/webp", false, false), Some(Kind::Image));
        assert_eq!(media_kind("IMAGE/GIF", true, false), Some(Kind::Image));
        assert_eq!(media_kind("video/mp4", false, false), Some(Kind::Video));
        assert_eq!(media_kind("video/webm", true, false), Some(Kind::Video));
        assert_eq!(media_kind("audio/ogg", false, true), Some(Kind::Audio));
        assert_eq!(media_kind("audio/mpeg", false, false), Some(Kind::Audio));
        assert_eq!(media_kind("application/octet-stream", true, true), Some(Kind::Video));
        assert_eq!(media_kind("application/octet-stream", false, true), Some(Kind::Audio));
        assert_eq!(media_kind("application/x-tgsticker", false, false), None);
        assert_eq!(media_kind("application/pdf", false, false), None);
        assert_eq!(media_kind("", false, false), None);
    }

    #[test]
    fn chunks_cover_the_duration_in_thirty_second_steps() {
        assert_eq!(chunk_count(0.4).unwrap(), 1);
        assert_eq!(chunk_count(30.0).unwrap(), 1);
        assert_eq!(chunk_count(30.01).unwrap(), 2);
        assert_eq!(chunk_count(60.0).unwrap(), 2);
        assert_eq!(chunk_count(61.5).unwrap(), 3);
        assert_eq!(chunk_span(61.5, 0), (0.0, 30.0));
        assert_eq!(chunk_span(61.5, 1), (30.0, 60.0));
        assert_eq!(chunk_span(61.5, 2), (60.0, 61.5));
        assert_eq!(chunk_span(0.4, 0), (0.0, 0.4));
        assert_eq!(chunk_count(30_000.0).unwrap(), 1000);
        assert!(matches!(chunk_count(30_000.5), Err(Fault::Code(code)) if code == "media exceeds configured chunk limit"));
    }

    #[test]
    fn rate_is_items_per_minute_over_a_sliding_window() {
        let start = Instant::now();
        let at = |seconds: u64| start + Duration::from_secs(seconds);
        let mut rate = Rate::default();
        assert_eq!(rate.per_minute(at(0)), 0.0);
        rate.record(at(0));
        assert_eq!(rate.per_minute(at(1)), 2.0);
        for seconds in [10, 20, 30, 40, 50] {
            rate.record(at(seconds));
        }
        assert_eq!(rate.per_minute(at(60)), 6.0);
        assert_eq!(rate.per_minute(at(180)), 2.0);
        // The first item has left the window; five remain since second 10.
        assert!((rate.per_minute(at(181)) - 5.0 * 60.0 / 171.0).abs() < 1e-9);
        assert_eq!(rate.per_minute(at(231)), 0.0);
    }

    #[test]
    fn checkpoint_only_passes_a_finished_prefix() {
        let mut window = Window::default();
        window.push(11, Slot::Running);
        window.push(12, Slot::Settled(None));
        window.push(13, Slot::Running);
        window.push(14, Slot::Settled(Some(SELF_DESTRUCTING)));
        window.push(15, Slot::Running);
        assert_eq!((window.len(), window.media()), (5, 3));
        assert_eq!(window.advance(), None);

        // Newer messages finishing first must not move the checkpoint past the oldest.
        window.finish(15);
        window.finish(13);
        assert_eq!(window.advance(), None);
        assert_eq!(window.media(), 3);

        window.finish(11);
        let advance = window.advance().unwrap();
        assert_eq!(advance, Advance { checkpoint: 15, messages: 5, settled: vec![(12, None), (14, Some(SELF_DESTRUCTING))] });
        assert!(window.is_empty());
        assert_eq!(window.advance(), None);
    }

    #[test]
    fn checkpoint_stops_at_the_oldest_running_message() {
        let mut window = Window::default();
        window.push(1, Slot::Settled(None));
        window.push(2, Slot::Running);
        window.push(3, Slot::Running);
        window.push(4, Slot::Settled(None));
        assert_eq!(window.advance(), Some(Advance { checkpoint: 1, messages: 1, settled: vec![(1, None)] }));
        window.finish(3);
        assert_eq!(window.advance(), None);
        window.finish(2);
        assert_eq!(window.advance(), Some(Advance { checkpoint: 4, messages: 3, settled: vec![(4, None)] }));
    }

    #[test]
    fn history_pages_upward_from_the_checkpoint() {
        let first = History::new(tl::enums::InputPeer::Empty, 0).request;
        assert_eq!((first.offset_id, first.add_offset, first.limit, first.offset_date), (1, -100, 100, 0));
        let resumed = History::new(tl::enums::InputPeer::Empty, 4321);
        assert_eq!((resumed.request.offset_id, resumed.request.add_offset, resumed.last), (4322, -100, 4321));
        assert_eq!((resumed.request.min_id, resumed.request.max_id, resumed.request.hash), (0, 0, 0));
    }

    #[test]
    fn probe_reads_duration_and_streams() {
        let video = br#"{"streams":[{"codec_name":"vp9","codec_type":"video","duration":"N/A"},{"codec_name":"opus","codec_type":"audio","duration":"12.480000"}],"format":{"duration":"12.500000"}}"#;
        let probe = Probe::parse(video).unwrap();
        assert_eq!(probe, Probe { duration: 12.5, video: true, audio: true, video_codec: Some("vp9".to_owned()) });
        assert_eq!(probe.modalities(Kind::Video).unwrap(), [Kind::Video, Kind::Audio]);
        assert_eq!(probe.modalities(Kind::Audio).unwrap(), [Kind::Audio]);

        let silent = Probe::parse(br#"{"streams":[{"codec_name":"h264","codec_type":"video","duration":"3.2"}],"format":{}}"#).unwrap();
        assert_eq!(silent.modalities(Kind::Video).unwrap(), [Kind::Video]);
        assert!(matches!(silent.modalities(Kind::Audio), Err(Fault::Code(code)) if code == "audio stream unavailable"));

        let cover = Probe::parse(br#"{"streams":[{"codec_type":"audio","duration":"5"}],"format":{"duration":"5"}}"#).unwrap();
        assert!(matches!(cover.modalities(Kind::Video), Err(Fault::Code(code)) if code == "video stream unavailable"));

        assert!(matches!(Probe::parse(br#"{"streams":[],"format":{"duration":"N/A"}}"#), Err(Fault::Code(code)) if code == "media duration unavailable"));
        assert!(matches!(Probe::parse(b"not json"), Err(Fault::Code(code)) if code == "invalid probe output"));
    }

    #[test]
    fn extraction_arguments_match_the_prototype() {
        let strings = |arguments: Vec<OsString>| arguments.into_iter().map(|argument| argument.into_string().unwrap()).collect::<Vec<_>>();
        let (source, chunk) = (Path::new("source"), Path::new("chunk.mp4"));
        assert_eq!(
            strings(extraction(source, chunk, Kind::Video, 30.0, 30.0, false)),
            ["-nostdin", "-v", "error", "-y", "-ss", "30", "-i", "source", "-t", "30", "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "chunk.mp4"]
        );
        assert_eq!(
            strings(extraction(source, Path::new("chunk.wav"), Kind::Audio, 60.0, 1.5, false)),
            ["-nostdin", "-v", "error", "-y", "-ss", "60", "-i", "source", "-t", "1.5", "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "chunk.wav"]
        );
        let repaired = strings(extraction(source, chunk, Kind::Video, 0.0, 2.25, true));
        assert_eq!(repaired[4..9], ["-ss", "0", "-bsf:v", "vp9_metadata=color_space=bt709", "-i"]);
    }

    #[test]
    fn index_identity_is_stamped_once_and_never_adopted() {
        let mut db = fresh();
        let stored = |db: &Connection, key: &str| db.query_row("SELECT value FROM metadata WHERE key=?1", [key], |row| row.get::<_, String>(0)).unwrap();
        assert_eq!(count(&db, "SELECT COUNT(*) FROM metadata"), 9);
        assert_eq!(stored(&db, "model"), IDENTITY);
        assert_eq!(stored(&db, "account_id"), "42");
        assert_eq!(stored(&db, "audio_chunk_seconds"), "30");
        assert_eq!(stored(&db, "video_chunk_seconds"), "30");
        assert_eq!(stored(&db, "vector_format"), "float32-le-normalized");

        prepare(&mut db, 42).unwrap();
        assert!(matches!(prepare(&mut db, 43), Err(OpenError::Mismatch("account_id"))));
        assert_eq!(stored(&db, "account_id"), "42");

        let mut foreign = Connection::open_in_memory().unwrap();
        foreign.execute_batch(SCHEMA).unwrap();
        insert_item(&foreign, &item(1, "image", 0)).unwrap();
        assert!(matches!(prepare(&mut foreign, 42), Err(OpenError::Unidentified)));
        assert_eq!(count(&foreign, "SELECT COUNT(*) FROM metadata"), 0);
    }

    #[test]
    fn changed_source_drops_the_old_vectors() {
        let mut db = fresh();
        adopt_source(&mut db, -100, 7, "[\"image\",1,null]").unwrap();
        insert_item(&db, &item(7, "image", 0)).unwrap();
        insert_item(&db, &item(8, "image", 0)).unwrap();
        adopt_source(&mut db, -100, 7, "[\"image\",1,null]").unwrap();
        assert!(has_item(&db, -100, 7, "image", 0).unwrap());
        adopt_source(&mut db, -100, 7, "[\"video\",2,500]").unwrap();
        assert!(!has_item(&db, -100, 7, "image", 0).unwrap());
        assert!(has_item(&db, -100, 8, "image", 0).unwrap());
        assert_eq!(count(&db, "SELECT COUNT(*) FROM sources"), 1);
    }

    #[test]
    fn failures_count_attempts_and_stop_being_retried() {
        let mut db = fresh();
        record_failure(&db, -100, 5, "probing failed").unwrap();
        record_failure(&db, -100, 9, "timeout:downloading").unwrap();
        record_failure(&db, -100, 5, "extracting_video failed").unwrap();
        assert_eq!(retryable_failures(&db, -100).unwrap(), [5, 9]);
        record_failure(&db, -100, 5, "extracting_video failed").unwrap();
        assert_eq!(retryable_failures(&db, -100).unwrap(), [9]);
        let (error_type, attempts, updated_at): (String, i64, String) =
            db.query_row("SELECT error_type,attempts,updated_at FROM failures WHERE message_id=5", [], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))).unwrap();
        assert_eq!((error_type.as_str(), attempts), ("extracting_video failed", 3));
        assert!(updated_at.ends_with("+00:00") && updated_at.len() == 29);

        assert_eq!(checkpoint(&db, -100).unwrap(), 0);
        let advance = Advance { checkpoint: 12, messages: 3, settled: vec![(9, None), (12, Some(SELF_DESTRUCTING))] };
        advance_checkpoint(&mut db, -100, &advance).unwrap();
        assert_eq!(checkpoint(&db, -100).unwrap(), 12);
        assert_eq!(retryable_failures(&db, -100).unwrap(), [12]);

        insert_item(&db, &item(1, "image", 0)).unwrap();
        insert_item(&db, &item(2, "video", 0)).unwrap();
        insert_item(&db, &item(2, "video", 1)).unwrap();
        insert_item(&db, &item(2, "audio", 0)).unwrap();
        let counts = read_counts(&db).unwrap();
        assert_eq!((counts.images, counts.video_segments, counts.audio_segments, counts.chats, counts.failures), (1, 2, 1, 1, 2));
    }
}
