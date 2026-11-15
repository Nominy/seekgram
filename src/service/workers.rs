//! The pool of Python model worker processes.
//!
//! Each worker holds the embedding model in VRAM and answers one JSONL request at a
//! time on stdin/stdout. Nothing runs until the first request, and everything stops
//! again after ten quiet minutes, so the graphics card is only occupied while Seekgram
//! is actually indexing or searching.

use std::{
    collections::VecDeque,
    fmt::{self, Write as _},
    pin::Pin,
    process::Stdio,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Notify, oneshot, watch},
    time::{Instant, timeout},
};

use super::{
    hidden, provision,
    state::{SetupState, Shared},
};

const DIMENSION: usize = 768;
/// Loading the model from disk into VRAM, measured by a first throwaway request.
const LOAD_TIMEOUT: Duration = Duration::from_secs(180);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const SEND_TIMEOUT: Duration = Duration::from_secs(30);
/// Workers stop after this long without a request, to give the VRAM back.
const IDLE: Duration = Duration::from_secs(600);
/// A vector is about 16 KB of JSON; anything far beyond that is not an answer.
const LINE_LIMIT: u64 = 1024 * 1024;
/// What one worker needs, and what is left alone for the desktop and other programs.
const VRAM_PER_WORKER_MB: i64 = 3500;
const VRAM_RESERVE_MB: i64 = 1024;

const NOT_READY: &str = "The search model is not ready yet. Open Seekgram and look at Setup.";
const START_FAILED: &str = "The model worker could not be started. Open Seekgram and run Repair in Setup.";
const LOAD_FAILED: &str = "The search model could not be loaded on the graphics card. Close games and other programs that use a lot of video memory; if it keeps failing, run Repair in Setup.";
const DIED: &str = "The model worker stopped unexpectedly, twice in a row. Seekgram will start it again; if this keeps happening, run Repair in Setup.";
const STOPPED: &str = "The model worker was stopped.";
const NO_VRAM_READING: &str = "Seekgram could not ask the NVIDIA driver how much video memory is free. Check that the NVIDIA driver is installed, or choose the number of workers yourself in Settings.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Text,
    Image,
    Audio,
    Video,
}

impl Kind {
    /// The name used both in the worker protocol and in the index's `modality` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Image => "image",
            Kind::Audio => "audio",
            Kind::Video => "video",
        }
    }
}

#[derive(Debug)]
pub enum EmbedError {
    /// This input cannot be embedded (corrupt file, no frames). Record it and move on.
    Media(String),
    /// No worker could do it (model failed to load, process died twice). Stop and report.
    Worker(String),
}

impl fmt::Display for EmbedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedError::Media(code) | EmbedError::Worker(code) => formatter.write_str(code),
        }
    }
}

impl std::error::Error for EmbedError {}

/// One embedding request. `seconds`: length of an audio chunk file, at most 30.
pub struct Embed {
    pub kind: Kind,
    pub input: String,
    pub seconds: Option<f64>,
}

pub struct Workers {
    pool: Arc<Pool>,
}

impl Workers {
    pub fn new(shared: Arc<Shared>) -> Arc<Self> {
        let queue = Queue { urgent: VecDeque::new(), normal: VecDeque::new(), tasks: 0, loaded: 0, last_request: Instant::now() };
        Arc::new(Self {
            pool: Arc::new(Pool {
                shared,
                queue: Mutex::new(queue),
                arrived: Notify::new(),
                retired: Notify::new(),
                generation: watch::channel(0).0,
            }),
        })
    }

    /// Embeds one item and returns a normalized 768-float vector. Workers start lazily,
    /// up to the configured count. `urgent` requests (searches) are served before
    /// queued indexing work.
    pub async fn embed(&self, request: Embed, urgent: bool) -> Result<Vec<f32>, EmbedError> {
        let (reply, answer) = oneshot::channel();
        let launching = {
            let mut queue = self.pool.queue();
            queue.last_request = Instant::now();
            queue.push_back(Job { request, reply, urgent, retried: false });
            // Subscribed under the lock, so a `stop` cannot slip in between unseen.
            (queue.tasks == 0).then(|| {
                queue.tasks = 1;
                self.pool.generation.subscribe()
            })
        };
        if let Some(stop) = launching {
            tokio::spawn(launch(self.pool.clone(), stop));
        }
        self.pool.arrived.notify_one();
        answer.await.unwrap_or_else(|_| Err(EmbedError::Worker(STOPPED.to_owned())))
    }

    /// How many workers are loaded right now.
    pub fn loaded(&self) -> u32 {
        self.pool.queue().loaded
    }

    /// Stops all worker processes (shutdown, or to free VRAM when idle).
    pub async fn stop(&self) {
        loop {
            // Created before the check: a retirement in between still wakes it.
            let retired = self.pool.retired.notified();
            {
                let queue = self.pool.queue();
                if queue.tasks == 0 {
                    return;
                }
                self.pool.generation.send_modify(|generation| *generation += 1);
            }
            retired.await;
        }
    }
}

struct Job {
    request: Embed,
    reply: oneshot::Sender<Result<Vec<f32>, EmbedError>>,
    urgent: bool,
    /// A request gets one more try, on a fresh or different worker, after a worker dies under it.
    retried: bool,
}

struct Queue {
    urgent: VecDeque<Job>,
    normal: VecDeque<Job>,
    /// Worker tasks alive: counting VRAM, loading the model, or serving.
    tasks: u32,
    loaded: u32,
    last_request: Instant,
}

impl Queue {
    fn push_back(&mut self, job: Job) {
        if job.urgent { self.urgent.push_back(job) } else { self.normal.push_back(job) }
    }

    fn push_front(&mut self, job: Job) {
        if job.urgent { self.urgent.push_front(job) } else { self.normal.push_front(job) }
    }

    fn pop(&mut self) -> Option<Job> {
        self.urgent.pop_front().or_else(|| self.normal.pop_front())
    }

    fn is_empty(&self) -> bool {
        self.urgent.is_empty() && self.normal.is_empty()
    }
}

struct Pool {
    shared: Arc<Shared>,
    queue: Mutex<Queue>,
    /// Wakes one waiting worker when a request is queued.
    arrived: Notify,
    /// Signalled when the last worker task has ended and its process is gone.
    retired: Notify,
    /// Bumped by `stop`; every worker task started before then ends.
    generation: watch::Sender<u64>,
}

/// How a worker task ended.
enum Exit {
    /// Nothing was asked for ten minutes.
    Idle,
    /// The reason, for whoever is still waiting if this was the last worker.
    Failed(String),
}

/// What went wrong with one request.
#[derive(Debug, PartialEq)]
enum Failure {
    /// The worker answered that this input cannot be embedded; a short code, never content.
    Media(String),
    /// The process died, hung or answered nonsense. It cannot be trusted with another request.
    Protocol,
}

/// Counts a worker as loaded for as long as it lives, also when its task is cancelled.
struct Loaded<'a>(&'a Pool);

impl<'a> Loaded<'a> {
    fn new(pool: &'a Pool) -> Self {
        pool.count_loaded(1);
        Self(pool)
    }
}

impl Drop for Loaded<'_> {
    fn drop(&mut self) {
        self.0.count_loaded(-1);
    }
}

impl Pool {
    fn queue(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn count_loaded(&self, change: i32) {
        let mut queue = self.queue();
        queue.loaded = queue.loaded.saturating_add_signed(change);
        let loaded = queue.loaded;
        // Published under the lock so two changes cannot reach the status out of order.
        self.shared.update(|status| status.indexer.workers = loaded);
    }

    /// How many workers to run: the setting, or as many as the free VRAM holds.
    async fn count(&self) -> Result<u32, String> {
        let (ready, configured) = self.shared.read(|status| (status.setup.state == SetupState::Ready, status.settings.workers));
        if !ready {
            return Err(NOT_READY.to_owned());
        }
        if configured > 0 {
            return Ok(u32::from(configured));
        }
        let mut command = Command::new("nvidia-smi");
        command
            .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let output = match timeout(Duration::from_secs(20), hidden(&mut command).output()).await {
            Ok(Ok(output)) if output.status.success() => output.stdout,
            _ => return Err(NO_VRAM_READING.to_owned()),
        };
        // PyTorch's default device is the first GPU, so that is the one that matters.
        let free_mb = String::from_utf8_lossy(&output).lines().next().and_then(|line| line.trim().parse::<i64>().ok());
        free_mb.map(workers_for_free_vram).ok_or_else(|| NO_VRAM_READING.to_owned())
    }

    /// Runs one worker process until it is idle for too long or cannot go on.
    /// The process stays in `slot` so the caller can end it whichever way this returns.
    async fn serve(&self, slot: &mut Option<Process>) -> Exit {
        let mut replaced = false;
        loop {
            let process = match Process::spawn() {
                Ok(process) => slot.insert(process),
                Err(reason) => return Exit::Failed(reason.to_owned()),
            };
            let warmup = Embed { kind: Kind::Text, input: "warmup".to_owned(), seconds: None };
            if process.request(&warmup, LOAD_TIMEOUT).await.is_err() {
                return Exit::Failed(LOAD_FAILED.to_owned());
            }
            {
                let _loaded = Loaded::new(self);
                loop {
                    let Some(job) = self.next_job().await else { return Exit::Idle };
                    // Whoever asked has given up; the queue is full of these after indexing is switched off.
                    if job.reply.is_closed() {
                        continue;
                    }
                    let outcome = process.request(&job.request, REQUEST_TIMEOUT).await;
                    self.queue().last_request = Instant::now();
                    match outcome {
                        Ok(vector) => drop(job.reply.send(Ok(vector))),
                        Err(Failure::Media(code)) => drop(job.reply.send(Err(EmbedError::Media(code)))),
                        Err(Failure::Protocol) => {
                            self.abandon(job);
                            break;
                        }
                    }
                }
            }
            if let Some(dead) = slot.take() {
                dead.terminate().await;
            }
            if replaced {
                return Exit::Failed(DIED.to_owned());
            }
            replaced = true;
        }
    }

    /// The next request, urgent ones first; `None` once nothing has been asked for `IDLE`.
    async fn next_job(&self) -> Option<Job> {
        loop {
            let deadline = {
                let mut queue = self.queue();
                if let Some(job) = queue.pop() {
                    // Taking a request counts as activity, so another worker does not go
                    // idle (and shrink the pool) while this one is on a long video.
                    queue.last_request = Instant::now();
                    return Some(job);
                }
                queue.last_request + IDLE
            };
            if Instant::now() >= deadline {
                return None;
            }
            tokio::select! {
                _ = self.arrived.notified() => {}
                _ = tokio::time::sleep_until(deadline) => {}
            }
        }
    }

    /// Hands a request whose worker died back to the queue, once.
    fn abandon(&self, mut job: Job) {
        if job.retried {
            let _ = job.reply.send(Err(EmbedError::Worker(DIED.to_owned())));
            return;
        }
        job.retried = true;
        self.queue().push_front(job);
        self.arrived.notify_one();
    }

    /// Gives up a worker task's place in the pool, after its process is gone.
    /// `failure` is why it could not go on; `None` when it merely went idle.
    fn retire(self: &Arc<Self>, failure: Option<&str>) {
        let mut queue = self.queue();
        queue.tasks -= 1;
        if queue.tasks > 0 {
            return;
        }
        match failure {
            // Asked for while the last idle worker was shutting down: start over for them.
            None if !queue.is_empty() => {
                queue.tasks = 1;
                let stop = self.generation.subscribe();
                drop(queue);
                tokio::spawn(launch(self.clone(), stop));
                return;
            }
            None => drop(queue),
            // Nobody is left to answer; say why instead of leaving them waiting.
            Some(reason) => {
                let queue = &mut *queue;
                let waiting: Vec<Job> = queue.urgent.drain(..).chain(queue.normal.drain(..)).collect();
                for job in waiting {
                    let _ = job.reply.send(Err(EmbedError::Worker(reason.to_owned())));
                }
            }
        }
        self.retired.notify_waiters();
    }
}

/// Starts the pool: decides how many workers fit, then hands its place over to them.
/// Boxed because a retiring worker can start the pool again, which would otherwise
/// make the worker's own future type refer to itself.
fn launch(pool: Arc<Pool>, mut stop: watch::Receiver<u64>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
    Box::pin(async move {
        let count = tokio::select! {
            count = pool.count() => count,
            _ = stop.changed() => Err(STOPPED.to_owned()),
        };
        match count {
            Ok(count) => {
                pool.queue().tasks += count - 1;
                for _ in 0..count {
                    tokio::spawn(worker(pool.clone(), stop.clone()));
                }
            }
            Err(reason) => pool.retire(Some(&reason)),
        }
    })
}

async fn worker(pool: Arc<Pool>, mut stop: watch::Receiver<u64>) {
    let mut process = None;
    let exit = tokio::select! {
        exit = pool.serve(&mut process) => exit,
        _ = stop.changed() => Exit::Failed(STOPPED.to_owned()),
    };
    // The process goes before the place in the pool does, so a restart never loads a
    // second model next to one that has not let go of its VRAM yet.
    if let Some(process) = process {
        process.terminate().await;
    }
    match exit {
        Exit::Idle => pool.retire(None),
        Exit::Failed(reason) => pool.retire(Some(&reason)),
    }
}

struct Process {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    sequence: i64,
}

impl Process {
    fn spawn() -> Result<Self, &'static str> {
        let mut command = Command::new(provision::python());
        command
            .arg(provision::worker_script())
            .arg("--ffmpeg")
            .arg(provision::ffmpeg())
            .envs(provision::worker_env())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let mut child = hidden(&mut command).spawn().map_err(|error| {
            eprintln!("workers: the model worker did not start: {error}");
            START_FAILED
        })?;
        let (stdin, stdout) = child.stdin.take().zip(child.stdout.take()).ok_or(START_FAILED)?;
        Ok(Self { child, stdin, stdout: BufReader::new(stdout), sequence: 0 })
    }

    /// One request and its answer. After `Failure::Protocol` the process must be ended:
    /// its next line could belong to this request.
    async fn request(&mut self, embed: &Embed, patience: Duration) -> Result<Vec<f32>, Failure> {
        self.sequence += 1;
        let line = request_line(self.sequence, embed);
        let sent = async {
            self.stdin.write_all(line.as_bytes()).await?;
            self.stdin.flush().await
        };
        if !matches!(timeout(SEND_TIMEOUT, sent).await, Ok(Ok(()))) {
            return Err(Failure::Protocol);
        }
        timeout(patience, self.answer()).await.unwrap_or(Err(Failure::Protocol))
    }

    async fn answer(&mut self) -> Result<Vec<f32>, Failure> {
        loop {
            let mut line = Vec::new();
            let read = (&mut self.stdout).take(LINE_LIMIT).read_until(b'\n', &mut line).await;
            // Nothing read: the process exited. No newline: cut off, or longer than any answer.
            if read.is_err() || line.last() != Some(&b'\n') {
                return Err(Failure::Protocol);
            }
            let response = serde_json::from_slice::<Value>(&line).map_err(|_| Failure::Protocol)?;
            if let Some(outcome) = decode(&response, self.sequence) {
                return outcome;
            }
        }
    }

    async fn terminate(mut self) {
        let _ = timeout(Duration::from_secs(3), self.child.kill()).await;
    }
}

fn workers_for_free_vram(free_mb: i64) -> u32 {
    ((free_mb - VRAM_RESERVE_MB) / VRAM_PER_WORKER_MB).clamp(1, 2) as u32
}

/// The request as one line of pure-ASCII JSON. The worker reads stdin in the Windows
/// code page, so a raw non-ASCII character (a search in Russian, a user folder with an
/// accent) would arrive garbled; `\uXXXX` escapes survive any code page.
fn request_line(id: i64, embed: &Embed) -> String {
    let mut request = json!({"id": id, "kind": embed.kind.as_str(), "input": embed.input});
    if let (Kind::Audio, Some(seconds)) = (embed.kind, embed.seconds) {
        // The input is already a chunk, so the bounds are relative to that file.
        request["start_seconds"] = json!(0);
        request["end_seconds"] = json!(seconds.min(30.0));
    }
    let json = request.to_string();
    let mut line = String::with_capacity(json.len() + 1);
    for character in json.chars() {
        if character.is_ascii() {
            line.push(character);
        } else {
            for unit in character.encode_utf16(&mut [0; 2]) {
                let _ = write!(line, "\\u{unit:04x}");
            }
        }
    }
    line.push('\n');
    line
}

/// Reads the worker's answer to request `id`. `None` for the answer to an earlier
/// request, which is skipped.
fn decode(response: &Value, id: i64) -> Option<Result<Vec<f32>, Failure>> {
    let Some(response) = response.as_object() else { return Some(Err(Failure::Protocol)) };
    let answered = response.get("id").and_then(Value::as_i64);
    if answered.is_some_and(|answered| answered < id) {
        return None;
    }
    if answered != Some(id) {
        return Some(Err(Failure::Protocol));
    }
    if let Some(error) = response.get("error") {
        return Some(Err(Failure::Media(error_code(error))));
    }
    Some(normalized(response.get("vector")).ok_or(Failure::Protocol))
}

/// `embedding failed:<exception type>@<file:line>`, the form the prototype stored in
/// `failures.error_type`. Only identifier-like text passes; the error message never does.
fn error_code(error: &Value) -> String {
    let kind = error
        .get("type")
        .and_then(Value::as_str)
        .filter(|kind| kind.chars().all(|c| c.is_alphanumeric() || c == '_') && kind.chars().any(char::is_alphanumeric))
        .unwrap_or("Unknown");
    let location = error
        .get("location")
        .and_then(Value::as_str)
        .filter(|location| !location.is_empty() && location.chars().all(|c| c.is_alphanumeric() || "_.:-".contains(c)))
        .unwrap_or("unknown");
    format!("embedding failed:{kind}@{location}")
}

/// Checks the vector and scales it to unit length exactly as the prototype did, so the
/// same image indexed by either gives the same bytes (identical images are grouped by them).
fn normalized(vector: Option<&Value>) -> Option<Vec<f32>> {
    let values = vector?.as_array()?;
    if values.len() != DIMENSION {
        return None;
    }
    let values = values.iter().map(Value::as_f64).collect::<Option<Vec<f64>>>()?;
    // Python 3.12's `sum()` over floats is Neumaier-compensated; a plain sum can differ
    // in the last bit, and with it a few of the stored `f32`s.
    let (mut sum, mut compensation) = (0.0_f64, 0.0_f64);
    for square in values.iter().map(|value| value * value) {
        let total = sum + square;
        compensation += if sum.abs() >= square.abs() { (sum - total) + square } else { (square - total) + sum };
        sum = total;
    }
    if compensation != 0.0 && compensation.is_finite() {
        sum += compensation;
    }
    let norm = sum.sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return None;
    }
    Some(values.iter().map(|value| (value / norm) as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embed(kind: Kind, input: &str, seconds: Option<f64>) -> Embed {
        Embed { kind, input: input.to_owned(), seconds }
    }

    fn unit_vector() -> Vec<Value> {
        let mut values = vec![json!(0.0); DIMENSION];
        values[0] = json!(3.0);
        values[1] = json!(4);
        values
    }

    #[test]
    fn worker_count_follows_free_vram() {
        assert_eq!(workers_for_free_vram(0), 1);
        assert_eq!(workers_for_free_vram(4000), 1);
        assert_eq!(workers_for_free_vram(8023), 1);
        assert_eq!(workers_for_free_vram(8024), 2);
        assert_eq!(workers_for_free_vram(24_000), 2);
    }

    #[test]
    fn only_audio_requests_carry_bounds() {
        let image: Value = serde_json::from_str(&request_line(1, &embed(Kind::Image, "C:\\tmp\\a", Some(12.0)))).unwrap();
        assert_eq!(image, json!({"id": 1, "kind": "image", "input": "C:\\tmp\\a"}));
        let audio: Value = serde_json::from_str(&request_line(7, &embed(Kind::Audio, "a.wav", Some(12.5)))).unwrap();
        assert_eq!(audio, json!({"id": 7, "kind": "audio", "input": "a.wav", "start_seconds": 0, "end_seconds": 12.5}));
        let long: Value = serde_json::from_str(&request_line(8, &embed(Kind::Audio, "a.wav", Some(31.0)))).unwrap();
        assert_eq!(long["end_seconds"], json!(30.0));
        let unbounded: Value = serde_json::from_str(&request_line(9, &embed(Kind::Audio, "a.wav", None))).unwrap();
        assert!(unbounded.get("end_seconds").is_none());
    }

    #[test]
    fn requests_are_ascii_and_round_trip() {
        let line = request_line(2, &embed(Kind::Text, "розовый дракон \u{1F409}", None));
        assert!(line.is_ascii());
        assert!(line.contains("\\u0440") && line.contains("\\ud83d\\udc09"));
        assert_eq!(line.matches('\n').count(), 1);
        let parsed: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["input"], json!("розовый дракон \u{1F409}"));
    }

    #[test]
    fn answers_are_matched_validated_and_normalized() {
        let vector = decode(&json!({"id": 5, "vector": unit_vector()}), 5).unwrap().unwrap();
        assert_eq!(vector.len(), DIMENSION);
        assert_eq!((vector[0], vector[1], vector[2]), (0.6, 0.8, 0.0));
        assert!(decode(&json!({"id": 4, "vector": unit_vector()}), 5).is_none());
        assert_eq!(decode(&json!({"id": 6, "vector": unit_vector()}), 5), Some(Err(Failure::Protocol)));
        assert_eq!(decode(&json!({"id": null, "vector": unit_vector()}), 5), Some(Err(Failure::Protocol)));
        assert_eq!(decode(&json!({"id": 5, "vector": [1.0, 2.0]}), 5), Some(Err(Failure::Protocol)));
        assert_eq!(decode(&json!({"id": 5, "vector": vec![json!(0.0); DIMENSION]}), 5), Some(Err(Failure::Protocol)));
        let mut text = unit_vector();
        text[9] = json!("1");
        assert_eq!(decode(&json!({"id": 5, "vector": text}), 5), Some(Err(Failure::Protocol)));
        assert_eq!(decode(&json!([1, 2]), 5), Some(Err(Failure::Protocol)));
    }

    #[test]
    fn worker_errors_become_codes_without_content() {
        let error = json!({"id": 3, "error": {"type": "RequestError", "message": "secret words", "location": "worker.py:131"}});
        assert_eq!(decode(&error, 3), Some(Err(Failure::Media("embedding failed:RequestError@worker.py:131".to_owned()))));
        let odd = json!({"id": 3, "error": {"type": "bad type!", "location": "C:\\Users\\name\\x.py"}});
        assert_eq!(decode(&odd, 3), Some(Err(Failure::Media("embedding failed:Unknown@unknown".to_owned()))));
        let bare = json!({"id": 3, "error": {"type": "ValueError", "location": null}});
        assert_eq!(decode(&bare, 3), Some(Err(Failure::Media("embedding failed:ValueError@unknown".to_owned()))));
    }
}
