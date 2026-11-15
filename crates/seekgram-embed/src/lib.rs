//! `seekgram embed-worker`: the embedding model (EmbeddingGemma 2 through ONNX Runtime) in a
//! child process of the service, so a driver crash cannot take the service down and stopping
//! the process gives all of its memory back.
//!
//! It speaks JSONL on stdin/stdout, one request at a time:
//! `{"id", "kind", "input", "start_seconds"?, "end_seconds"?}` in, `{"id", "vector"}` or
//! `{"id", "error": {"type", "message", "location"}}` out. Preprocessing reproduces the Python
//! worker the index was built with bit for bit, so old and new vectors can share the index.
//!
//! `--self-test` instead embeds a text, picture, sound and video shipped in this binary and
//! compares the vectors with the ones the Python worker made from them; setup only calls an
//! engine ready that passes. Progress goes to stderr as `stage=...`, `device=...`,
//! `cosine=<kind> <value>`, then `passed` or `failure=<cause>`.

mod audio;
pub mod gpu;
mod media;
mod model;
mod vision;

use std::{
    io::{BufRead, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use seekgram_core::paths;

use media::MediaError;
use model::{Item, Model, Provider};
use vision::{FRAME_TOKENS, IMAGE_TOKENS};

pub use model::GRAPHS;

pub const DIMENSION: usize = 768;

/// Where the model runs and at which precision. Setup tries them in this order and keeps the
/// first that passes the self-test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Engine {
    DirectmlFp16,
    DirectmlFp32,
    CpuFp32,
}

impl Engine {
    pub const LADDER: [Engine; 3] = [Engine::DirectmlFp16, Engine::DirectmlFp32, Engine::CpuFp32];

    pub fn name(self) -> &'static str {
        match self {
            Engine::DirectmlFp16 => "directml-fp16",
            Engine::DirectmlFp32 => "directml-fp32",
            Engine::CpuFp32 => "cpu-fp32",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        Self::LADDER.into_iter().find(|engine| engine.name() == name)
    }

    pub fn directml(self) -> bool {
        self != Engine::CpuFp32
    }

    /// Whether it needs the full-precision weights (about 2.9 GB more).
    pub fn fp32(self) -> bool {
        self != Engine::DirectmlFp16
    }

    /// File-name suffix of the graphs and weights.
    pub fn suffix(self) -> &'static str {
        if self.fp32() { "" } else { "_fp16" }
    }

    /// How the panel names it, in brackets after the device.
    pub fn label(self) -> &'static str {
        match self {
            Engine::DirectmlFp16 => "DirectML",
            Engine::DirectmlFp32 => "DirectML, full precision",
            Engine::CpuFp32 => "CPU, slow",
        }
    }

    /// Arguments that start a worker on this engine.
    pub fn arguments(self) -> [&'static str; 3] {
        ["embed-worker", "--engine", self.name()]
    }
}

/// Why the model could not be loaded.
enum LoadError {
    /// `onnxruntime.dll` or `DirectML.dll` would not load.
    Runtime(anyhow::Error),
    /// A model file is missing.
    Files(String),
    /// The files are there but the engine could not create its sessions.
    Session(anyhow::Error),
}

struct Worker {
    // Declared before the environment, so the sessions are dropped first.
    model: Model,
    _environment: ort::environment::Environment,
    extractor: audio::Extractor,
    ffmpeg: PathBuf,
    /// The graphics card or the processor, in words.
    device: String,
}

/// What failed in a request; becomes the error's `type` and `location`.
enum Stage {
    Decode,
    Model,
}

impl Worker {
    fn load(engine: Engine) -> Result<Self, LoadError> {
        let directory = paths::model_dir();
        let mut files = vec![directory.join("tokenizer.json")];
        for graph in GRAPHS {
            let base = format!("{graph}{}.onnx", engine.suffix());
            files.push(directory.join(format!("{base}_data")));
            files.push(directory.join(base));
        }
        if let Some(missing) = files.iter().find(|file| !file.is_file()) {
            return Err(LoadError::Files(missing.file_name().unwrap_or_default().to_string_lossy().into_owned()));
        }
        let (provider, device) = if engine.directml() {
            let Some(adapter) = gpu::adapter() else {
                return Err(LoadError::Session(anyhow::anyhow!("no hardware graphics adapter")));
            };
            (Provider::DirectMl(adapter.index), adapter.name)
        } else {
            (Provider::Cpu, "this PC's processor".to_owned())
        };
        let environment = model::runtime(&paths::runtime_dir(), engine.directml()).map_err(LoadError::Runtime)?;
        let model = Model::load(&environment, &directory, engine.suffix(), provider).map_err(LoadError::Session)?;
        Ok(Self { model, _environment: environment, extractor: audio::Extractor::new(), ffmpeg: paths::ffmpeg(), device })
    }

    /// Validates a request as the Python worker did, preprocesses and embeds it.
    fn request(&mut self, request: &Value) -> Result<Vec<f32>, (Stage, anyhow::Error)> {
        let item = self.prepare(request).map_err(|error| (Stage::Decode, error))?;
        self.model.embed(item).map_err(|error| (Stage::Model, error))
    }

    fn prepare(&self, request: &Value) -> Result<Item> {
        let kind = request.get("kind").and_then(Value::as_str).unwrap_or("");
        let input = request.get("input").and_then(Value::as_str).filter(|input| !input.is_empty());
        if !matches!(kind, "text" | "image" | "audio" | "video") {
            bail!(MediaError("invalid kind"));
        }
        let Some(input) = input else { bail!(MediaError("input must be a nonempty string")) };
        let bounded = request.get("start_seconds").is_some() || request.get("end_seconds").is_some();
        if kind == "text" {
            if bounded {
                bail!(MediaError("bounds apply only to audio and video"));
            }
            return Ok(Item::Text(input.to_owned()));
        }
        let path = Path::new(input);
        if !path.is_absolute() || !path.is_file() {
            bail!(MediaError("media input must be an existing absolute local file"));
        }
        if kind == "image" {
            if bounded {
                bail!(MediaError("bounds apply only to audio and video"));
            }
            return Ok(Item::Image(vision::patchify(&vision::load_image(path)?, IMAGE_TOKENS)?));
        }
        let (start, end) = bounds(request)?;
        if kind == "audio" {
            return Ok(Item::Audio(self.extractor.extract(&media::audio(&self.ffmpeg, path, start, end)?)));
        }
        let frames = media::video(&self.ffmpeg, path, start, end)?;
        Ok(Item::Video(frames.iter().map(|frame| vision::patchify(frame, FRAME_TOKENS)).collect::<Result<_>>()?))
    }
}

/// `start_seconds` and `end_seconds` as the Python worker checked them.
fn bounds(request: &Value) -> Result<(f64, Option<f64>)> {
    let number = |key: &str| match request.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => Ok(number.as_f64().filter(|value| value.is_finite())),
        Some(_) => Err(()),
    };
    let start = match number("start_seconds") {
        Ok(start) => start.unwrap_or(0.0),
        Err(()) => bail!(MediaError("invalid start_seconds")),
    };
    if start < 0.0 {
        bail!(MediaError("invalid start_seconds"));
    }
    let end = match number("end_seconds") {
        Ok(end) => end,
        Err(()) => bail!(MediaError("invalid end_seconds")),
    };
    if end.is_some_and(|end| end <= start) {
        bail!(MediaError("invalid end_seconds"));
    }
    Ok((start, end))
}

/// The error answer. Only fixed words leave the worker: a decoder's own message could name the
/// file. The full error goes to the log without the request.
fn failure(id: &Value, stage: Stage, error: &anyhow::Error) -> Value {
    let (kind, message, location) = match (error.downcast_ref::<MediaError>(), stage) {
        (Some(media), _) => ("RequestError", media.0, "media"),
        (None, Stage::Decode) => ("DecodeError", "the file could not be decoded", "decode"),
        (None, Stage::Model) => ("ModelError", "the model failed on this input", "model"),
    };
    if !matches!(kind, "RequestError") {
        eprintln!("embed-worker: {kind}: {error:#}");
    }
    json!({"id": id, "error": {"type": kind, "message": message, "location": location}})
}

/// `seekgram embed-worker [--engine directml-fp16|directml-fp32|cpu-fp32] [--self-test]`.
pub fn main() -> i32 {
    let mut engine = Engine::DirectmlFp16;
    let mut self_test = false;
    let mut arguments = std::env::args().skip(2);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--engine" => match arguments.next().as_deref().and_then(Engine::parse) {
                Some(chosen) => engine = chosen,
                None => {
                    eprintln!("embed-worker: --engine takes directml-fp16, directml-fp32 or cpu-fp32");
                    return 2;
                }
            },
            "--self-test" => self_test = true,
            other => {
                eprintln!("embed-worker: unknown argument {other}");
                return 2;
            }
        }
    }
    if self_test {
        return run_self_test(engine);
    }
    match serve(engine) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("embed-worker: {error:#}");
            1
        }
    }
}

fn load(engine: Engine) -> Result<Worker, LoadError> {
    let started = Instant::now();
    let worker = Worker::load(engine)?;
    let directml = if engine.directml() { model::loaded_from("DirectML.dll").map(|path| format!(", {path}")).unwrap_or_default() } else { String::new() };
    eprintln!("embed-worker: {} loaded on {} in {:.1} s{directml}", engine.name(), worker.device, started.elapsed().as_secs_f64());
    Ok(worker)
}

fn serve(engine: Engine) -> Result<()> {
    let mut worker = match load(engine) {
        Ok(worker) => worker,
        Err(LoadError::Runtime(error) | LoadError::Session(error)) => return Err(error),
        Err(LoadError::Files(name)) => bail!("the model file {name} is missing"),
    };
    let mut stdout = std::io::stdout().lock();
    for line in std::io::stdin().lock().lines() {
        let line = line.context("cannot read the request")?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        // A bug in a decoder must cost one item, not the worker and with it the indexing run.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker.request(&request)));
        let response = match outcome {
            Ok(Ok(vector)) => json!({"id": id, "vector": vector}),
            Ok(Err((stage, error))) => failure(&id, stage, &error),
            Err(_) => json!({"id": id, "error": {"type": "Panic", "message": "request processing failed", "location": "embed-worker"}}),
        };
        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}

/// The self-test inputs, the vectors the Python worker made from them, and the least cosine
/// that counts as the same result. Audio is held to a lower bar: the old worker's bfloat16
/// rounding moves audio vectors the most (down to 0.9976 on the spike's test set).
const SELF_TEST_TEXT: &str = "a bear looking at the camera";
const SELF_TEST_IMAGE: &[u8] = include_bytes!("../../../assets/selftest/image.png");
const SELF_TEST_AUDIO: &[u8] = include_bytes!("../../../assets/selftest/audio.ogg");
const SELF_TEST_VIDEO: &[u8] = include_bytes!("../../../assets/selftest/video.mp4");
/// Four 768-float little-endian vectors: text, image, audio, video.
const SELF_TEST_VECTORS: &[u8] = include_bytes!("../../../assets/selftest/vectors.f32");

fn reference(index: usize) -> Vec<f32> {
    SELF_TEST_VECTORS[index * DIMENSION * 4..(index + 1) * DIMENSION * 4]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum::<f64>();
    dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt())
}

fn run_self_test(engine: Engine) -> i32 {
    let fail = |cause: &str| {
        eprintln!("failure={cause}");
        1
    };
    eprintln!("stage=loading_model");
    let mut worker = match load(engine) {
        Ok(worker) => worker,
        Err(LoadError::Runtime(error)) => {
            eprintln!("embed-worker: {error:#}");
            return fail("runtime");
        }
        Err(LoadError::Files(name)) => {
            eprintln!("embed-worker: the model file {name} is missing");
            return fail("files");
        }
        Err(LoadError::Session(error)) => {
            eprintln!("embed-worker: {error:#}");
            return fail("load");
        }
    };
    eprintln!("device={}", worker.device);
    let folder = paths::temp_dir().join(format!("self-test-{}", std::process::id()));
    let written = std::fs::create_dir_all(&folder).and_then(|()| {
        for (name, bytes) in [("image.png", SELF_TEST_IMAGE), ("audio.ogg", SELF_TEST_AUDIO), ("video.mp4", SELF_TEST_VIDEO)] {
            std::fs::write(folder.join(name), bytes)?;
        }
        Ok(())
    });
    if let Err(error) = written {
        eprintln!("embed-worker: cannot write the test files: {error}");
        let _ = std::fs::remove_dir_all(&folder);
        return fail("write");
    }
    let file = |name: &str| folder.join(name).to_string_lossy().into_owned();
    let cases = [
        ("text", SELF_TEST_TEXT.to_owned(), 0.999),
        ("image", file("image.png"), 0.999),
        ("audio", file("audio.ogg"), 0.995),
        ("video", file("video.mp4"), 0.999),
    ];
    let mut outcome = Ok(());
    for (index, (kind, input, least)) in cases.into_iter().enumerate() {
        eprintln!("stage={kind}");
        match worker.request(&json!({"kind": kind, "input": input})) {
            Ok(vector) => {
                let similarity = cosine(&vector, &reference(index));
                eprintln!("cosine={kind} {similarity:.6}");
                if !(similarity >= least) {
                    outcome = Err("mismatch");
                    break;
                }
            }
            Err((stage, error)) => {
                eprintln!("embed-worker: {error:#}");
                // ffmpeg or a decoder failing on known-good files is this PC's problem, not
                // the engine's; the model failing is the engine's.
                outcome = Err(match (error.downcast_ref::<MediaError>(), stage) {
                    (None, Stage::Model) => "run",
                    _ => "media",
                });
                break;
            }
        }
    }
    let _ = std::fs::remove_dir_all(&folder);
    match outcome {
        Ok(()) => {
            eprintln!("passed");
            0
        }
        Err(cause) => fail(cause),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engines_round_trip_by_name() {
        for engine in Engine::LADDER {
            assert_eq!(Engine::parse(engine.name()), Some(engine));
            assert_eq!(serde_json::to_value(engine).unwrap(), json!(engine.name()));
        }
        assert_eq!(Engine::parse("cuda"), None);
        assert_eq!(Engine::DirectmlFp16.suffix(), "_fp16");
        assert_eq!(Engine::CpuFp32.suffix(), "");
        assert!(!Engine::CpuFp32.directml());
    }

    #[test]
    fn reference_vectors_are_unit_length() {
        assert_eq!(SELF_TEST_VECTORS.len(), 4 * DIMENSION * 4);
        for index in 0..4 {
            let vector = reference(index);
            assert!((cosine(&vector, &vector) - 1.0).abs() < 1e-9);
            let norm = vector.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "{index}: {norm}");
        }
        assert!(cosine(&reference(0), &reference(1)) < 0.9);
    }

    #[test]
    fn requests_are_checked_like_the_python_worker() {
        let check = |request: Value| bounds(&request).map_err(|error| error.downcast_ref::<MediaError>().map(|media| media.0));
        assert_eq!(check(json!({})).unwrap(), (0.0, None));
        assert_eq!(check(json!({"start_seconds": 0, "end_seconds": 12.5})).unwrap(), (0.0, Some(12.5)));
        assert_eq!(check(json!({"start_seconds": null})).unwrap(), (0.0, None));
        assert_eq!(check(json!({"start_seconds": -1})).unwrap_err(), Some("invalid start_seconds"));
        assert_eq!(check(json!({"start_seconds": "1"})).unwrap_err(), Some("invalid start_seconds"));
        assert_eq!(check(json!({"start_seconds": 5, "end_seconds": 5})).unwrap_err(), Some("invalid end_seconds"));
    }

    #[test]
    fn errors_carry_fixed_words_only() {
        let media = failure(&json!(3), Stage::Decode, &anyhow::Error::new(MediaError("ffmpeg failed to decode media")));
        assert_eq!(media, json!({"id": 3, "error": {"type": "RequestError", "message": "ffmpeg failed to decode media", "location": "media"}}));
        let decode = failure(&json!(4), Stage::Decode, &anyhow::anyhow!("cannot open C:\\Users\\name\\secret.jpg"));
        assert_eq!(decode["error"], json!({"type": "DecodeError", "message": "the file could not be decoded", "location": "decode"}));
        let model = failure(&json!(5), Stage::Model, &anyhow::anyhow!("vision encoder: out of memory"));
        assert_eq!(model["error"]["type"], json!("ModelError"));
    }
}
