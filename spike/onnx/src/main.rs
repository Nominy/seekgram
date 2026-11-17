//! Feasibility spike: EmbeddingGemma 2 through ONNX Runtime, speaking the Seekgram worker's JSONL
//! protocol (`{"id", "kind", "input", "start_seconds"?, "end_seconds"?}` in,
//! `{"id", "vector"}` or `{"id", "error"}` out) so results can be compared with `worker.py`.
//!
//! Modes:
//! - default: one request at a time, as the worker runs today;
//! - `--bulk`: read every request first, preprocess on `--workers` threads and embed runs of
//!   same-kind requests in batches of `--batch`.
//!
//! Diagnostics (load times, time to first embedding, throughput, GPU memory, loaded DLLs) go to
//! stderr as JSON lines.

mod audio;
mod media;
mod model;
mod vision;

use std::{
    collections::BTreeMap,
    io::{BufRead, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::{
    media::MediaError,
    model::{Item, Model, Options, Provider},
    vision::{FRAME_TOKENS, IMAGE_TOKENS},
};

struct Arguments {
    ort: PathBuf,
    ffmpeg: PathBuf,
    model: Options,
    bulk: bool,
    batch: usize,
    workers: usize,
    timing: bool,
    stats: bool,
    dump: Option<PathBuf>,
}

fn arguments() -> Result<Arguments> {
    let mut ort = None;
    let mut directory = None;
    let mut ffmpeg = PathBuf::from("ffmpeg");
    let mut variant = String::from("fp16");
    let mut provider = Provider::Cpu;
    // Vision batch 4: VRAM peaks grow with frames per encoder run (32 frames: +8.5 GB fp16), speed does not.
    let (mut threads, mut vision_batch, mut batch, mut workers) = (0, 4, 1, 1);
    let (mut bulk, mut timing, mut stats, mut dump) = (false, false, false, None);
    let (mut arena_exact, mut shrink_arena, mut profile) = (false, false, None);
    let mut graphs = String::from("text,vision,audio");
    let mut list = std::env::args().skip(1);
    while let Some(flag) = list.next() {
        let mut value = || list.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--ort" => ort = Some(PathBuf::from(value()?)),
            "--model-dir" => directory = Some(PathBuf::from(value()?)),
            "--ffmpeg" => ffmpeg = PathBuf::from(value()?),
            "--variant" => variant = value()?,
            "--provider" => {
                provider = match value()?.as_str() {
                    "cuda" => Provider::Cuda,
                    "dml" => Provider::DirectMl,
                    "cpu" => Provider::Cpu,
                    other => bail!("unknown provider {other}"),
                }
            }
            "--threads" => threads = value()?.parse()?,
            "--vision-batch" => vision_batch = value()?.parse()?,
            "--batch" => batch = value()?.parse()?,
            "--workers" => workers = value()?.parse()?,
            "--graphs" => graphs = value()?,
            "--dump-dir" => dump = Some(PathBuf::from(value()?)),
            "--bulk" => bulk = true,
            "--arena-exact" => arena_exact = true,
            "--shrink-arena" => shrink_arena = true,
            "--profile" => profile = Some(PathBuf::from(value()?)),
            "--timing" => timing = true,
            "--stats" => stats = true,
            other => bail!("unknown argument {other}"),
        }
    }
    let suffix = match variant.as_str() {
        "fp32" => String::new(),
        "fp16" | "q4" | "q4f16" | "quantized" => format!("_{variant}"),
        other => bail!("unknown variant {other}"),
    };
    Ok(Arguments {
        ort: ort.context("--ort <onnxruntime.dll> is required")?,
        ffmpeg,
        model: Options {
            directory: directory.context("--model-dir is required")?,
            suffix,
            provider,
            threads,
            vision_batch,
            load_vision: graphs.contains("vision"),
            load_audio: graphs.contains("audio"),
            arena_exact,
            shrink_arena,
            profile,
        },
        bulk,
        batch: batch.max(1),
        workers: workers.max(1),
        timing,
        stats,
        dump,
    })
}

struct Prepared {
    item: Item,
    kind: &'static str,
    ms: f64,
}

struct Preprocessor {
    ffmpeg: PathBuf,
    extractor: audio::Extractor,
}

impl Preprocessor {
    /// Validates a request as the worker does and turns it into model input.
    fn prepare(&self, request: &Value) -> Result<Prepared> {
        let started = Instant::now();
        let kind = request.get("kind").and_then(Value::as_str).unwrap_or("");
        let input = request.get("input").and_then(Value::as_str).filter(|input| !input.is_empty());
        let (kind, input) = match (kind, input) {
            ("text", Some(input)) => ("text", input),
            ("image", Some(input)) => ("image", input),
            ("audio", Some(input)) => ("audio", input),
            ("video", Some(input)) => ("video", input),
            (_, None) if matches!(kind, "text" | "image" | "audio" | "video") => bail!(MediaError("input must be a nonempty string")),
            _ => bail!(MediaError("invalid kind")),
        };
        let bounded = request.get("start_seconds").is_some() || request.get("end_seconds").is_some();
        let item = if kind == "text" {
            if bounded {
                bail!(MediaError("bounds apply only to audio and video"));
            }
            Item::Text(input.to_owned())
        } else {
            let path = Path::new(input);
            if !path.is_absolute() || !path.is_file() {
                bail!(MediaError("media input must be an existing absolute local file"));
            }
            if kind == "image" {
                if bounded {
                    bail!(MediaError("bounds apply only to audio and video"));
                }
                Item::Image(vision::patchify(&vision::load_image(path)?, IMAGE_TOKENS)?)
            } else {
                let (start, end) = bounds(request)?;
                if kind == "audio" {
                    Item::Audio(self.extractor.extract(&media::audio(&self.ffmpeg, path, start, end)?))
                } else {
                    let frames = media::video(&self.ffmpeg, path, start, end)?;
                    Item::Video(frames.iter().map(|frame| vision::patchify(frame, FRAME_TOKENS)).collect::<Result<_>>()?)
                }
            }
        };
        Ok(Prepared { item, kind, ms: started.elapsed().as_secs_f64() * 1e3 })
    }
}

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

fn failure(id: &Value, error: &anyhow::Error) -> Value {
    let (kind, message) = match error.downcast_ref::<MediaError>() {
        Some(media) => ("RequestError", media.0.to_owned()),
        None => ("Error", format!("{error:#}")),
    };
    json!({"id": id, "error": {"type": kind, "message": message}})
}

/// Writes the model inputs of one request as raw little-endian files for comparison with the
/// Python trace (`scripts/compare_inputs.py`).
fn dump(directory: &Path, id: &Value, item: &Item, ids: &[i64]) -> Result<()> {
    let name = id.as_str().map_or_else(|| id.to_string(), str::to_owned);
    let write = |suffix: &str, bytes: Vec<u8>| std::fs::write(directory.join(format!("{name}.{suffix}")), bytes);
    let floats = |values: &[f32]| values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<u8>>();
    let longs = |values: &[i64]| values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<u8>>();
    write("input_ids.i64", longs(ids))?;
    match item {
        Item::Text(_) => {}
        Item::Image(patches) => {
            write("pixel_values.f32", floats(&patches.pixels))?;
            write("positions.i64", longs(&patches.positions))?;
        }
        Item::Audio(features) => {
            write("input_features.f32", floats(&features.data))?;
            write("input_features_mask.u8", features.mask.iter().map(|&valid| u8::from(valid)).collect())?;
        }
        Item::Video(frames) => {
            write("pixel_values.f32", frames.iter().flat_map(|frame| floats(&frame.pixels)).collect())?;
            write("positions.i64", frames.iter().flat_map(|frame| longs(&frame.positions)).collect())?;
        }
    }
    Ok(())
}

/// Dedicated GPU memory of this process and the non-system DLLs it has loaded, from Windows'
/// own counters (per-process figures are not available from nvidia-smi under WDDM).
fn process_stats() -> Value {
    let script = format!(
        "$p={pid};$m=0;try{{$m=((Get-Counter \"\\GPU Process Memory(pid_${{p}}_*)\\Dedicated Usage\" -ErrorAction Stop).CounterSamples|Measure-Object CookedValue -Sum).Sum}}catch{{}};\
         $d=(Get-Process -Id $p).Modules|Where-Object{{$_.FileName -notlike \"$env:WINDIR*\"}}|ForEach-Object{{@{{file=$_.FileName;bytes=(Get-Item $_.FileName).Length}}}};\
         @{{gpu_dedicated_bytes=$m;working_set_bytes=(Get-Process -Id $p).WorkingSet64;modules=@($d)}}|ConvertTo-Json -Compress -Depth 4",
        pid = std::process::id()
    );
    std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .ok()
        .and_then(|output| serde_json::from_slice(&output.stdout).ok())
        .unwrap_or(Value::Null)
}

fn log(value: Value) {
    eprintln!("{value}");
}

fn main() -> Result<()> {
    let launched = Instant::now();
    let arguments = arguments()?;
    let environment = model::ort_library(&arguments.ort)?;
    let mut model = Model::load(&environment, &arguments.model)?;
    let preprocessor = Preprocessor { ffmpeg: arguments.ffmpeg.clone(), extractor: audio::Extractor::new() };
    log(json!({
        "event": "loaded",
        "provider": format!("{:?}", arguments.model.provider),
        "variant": arguments.model.suffix,
        "since_launch_ms": launched.elapsed().as_secs_f64() * 1e3,
        "session_ms": model.load_ms.iter().map(|(graph, ms)| json!({"graph": graph, "ms": ms})).collect::<Vec<_>>(),
    }));
    if let Some(directory) = &arguments.dump {
        std::fs::create_dir_all(directory)?;
    }
    let stdout = std::io::stdout();
    let mut first = true;
    let mut respond = |response: Value| -> Result<()> {
        let mut out = stdout.lock();
        serde_json::to_writer(&mut out, &response)?;
        out.write_all(b"\n")?;
        out.flush()?;
        if std::mem::take(&mut first) {
            log(json!({"event": "first_response", "since_launch_ms": launched.elapsed().as_secs_f64() * 1e3}));
        }
        Ok(())
    };

    if !arguments.bulk {
        for line in std::io::stdin().lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let started = Instant::now();
            let request: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
            let id = request.get("id").cloned().unwrap_or(Value::Null);
            if request.get("kind").and_then(Value::as_str) == Some("stats") {
                respond(json!({"id": id, "stats": process_stats()}))?;
                continue;
            }
            let outcome = preprocessor.prepare(&request).and_then(|prepared| {
                if let Some(directory) = &arguments.dump {
                    dump(directory, &id, &prepared.item, &model.tokens(&prepared.item)?)?;
                }
                let (mut vectors, timing) = model.embed(std::slice::from_ref(&prepared.item))?;
                Ok((vectors.remove(0), prepared.ms, timing))
            });
            respond(match outcome {
                Ok((vector, preprocess_ms, timing)) => {
                    let mut response = json!({"id": id, "vector": vector});
                    if arguments.timing {
                        response["ms"] = json!({
                            "preprocess": preprocess_ms, "encoder": timing.encoder_ms, "text": timing.text_ms,
                            "total": started.elapsed().as_secs_f64() * 1e3,
                        });
                    }
                    response
                }
                Err(error) => failure(&id, &error),
            })?;
        }
    } else {
        let requests: Vec<Value> = std::io::stdin()
            .lock()
            .lines()
            .map_while(Result::ok)
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(&line).unwrap_or(Value::Null))
            .collect();
        let started = Instant::now();
        let next = AtomicUsize::new(0);
        // Bounded so preprocessing cannot run far ahead of the GPU (a prepared image is 7.7 MB).
        let (sender, receiver) = mpsc::sync_channel::<(usize, Result<Prepared>)>(arguments.batch * 2);
        let receiver = Mutex::new(receiver);
        let (mut preprocess_ms, mut encoder_ms, mut text_ms, mut done, mut failed, mut rejected) = (0.0, 0.0, 0.0, 0usize, 0usize, 0usize);
        // Time and count when the first batch finished: the steady-state rate excludes the
        // one-off cost of the first run of each graph (CUDA/cuDNN initialisation, 3-19 s).
        let mut warm: Option<(Instant, usize)> = None;
        std::thread::scope(|scope| -> Result<()> {
            for _ in 0..arguments.workers {
                let sender = sender.clone();
                let (next, requests, preprocessor) = (&next, &requests, &preprocessor);
                scope.spawn(move || {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= requests.len() || sender.send((index, preprocessor.prepare(&requests[index]))).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(sender);
            let receiver = receiver.lock().map_err(|_| anyhow!("receiver poisoned"))?;
            let mut pending: BTreeMap<usize, Result<Prepared>> = BTreeMap::new();
            let mut cursor = 0;
            let mut batch: Vec<(usize, Prepared)> = Vec::new();
            let mut flush = |batch: &mut Vec<(usize, Prepared)>, respond: &mut dyn FnMut(Value) -> Result<()>| -> Result<()> {
                if batch.is_empty() {
                    return Ok(());
                }
                let (indices, prepared): (Vec<usize>, Vec<Prepared>) = batch.drain(..).unzip();
                preprocess_ms += prepared.iter().map(|item| item.ms).sum::<f64>();
                let items: Vec<Item> = prepared.into_iter().map(|item| item.item).collect();
                match model.embed(&items) {
                    Ok((vectors, timing)) => {
                        encoder_ms += timing.encoder_ms;
                        text_ms += timing.text_ms;
                        done += vectors.len();
                        warm.get_or_insert_with(|| (Instant::now(), done));
                        for (index, vector) in indices.iter().zip(vectors) {
                            respond(json!({"id": requests[*index].get("id"), "vector": vector}))?;
                        }
                    }
                    Err(error) => {
                        failed += indices.len();
                        for index in indices {
                            respond(failure(requests[index].get("id").unwrap_or(&Value::Null), &error))?;
                        }
                    }
                }
                Ok(())
            };
            while cursor < requests.len() {
                while !pending.contains_key(&cursor) {
                    let (index, outcome) = receiver.recv().context("preprocessing thread died")?;
                    pending.insert(index, outcome);
                }
                match pending.remove(&cursor).context("missing result")? {
                    Ok(prepared) => {
                        if batch.last().is_some_and(|(_, last)| last.kind != prepared.kind) || batch.len() >= arguments.batch {
                            flush(&mut batch, &mut respond)?;
                        }
                        batch.push((cursor, prepared));
                    }
                    Err(error) => {
                        flush(&mut batch, &mut respond)?;
                        rejected += 1;
                        respond(failure(requests[cursor].get("id").unwrap_or(&Value::Null), &error))?;
                    }
                }
                cursor += 1;
            }
            flush(&mut batch, &mut respond)
        })?;
        let seconds = started.elapsed().as_secs_f64();
        let steady = warm.filter(|&(_, count)| done > count).map(|(at, count)| (done - count) as f64 / at.elapsed().as_secs_f64());
        log(json!({
            "event": "bulk", "requests": requests.len(), "embedded": done, "failed": failed + rejected,
            "batch": arguments.batch, "workers": arguments.workers, "seconds": seconds,
            "items_per_second": done as f64 / seconds,
            "steady_items_per_second": steady,
            "preprocess_cpu_ms_per_item": preprocess_ms / done.max(1) as f64,
            "encoder_ms_per_item": encoder_ms / done.max(1) as f64,
            "text_ms_per_item": text_ms / done.max(1) as f64,
        }));
    }
    if arguments.stats {
        log(json!({"event": "stats", "stats": process_stats()}));
    }
    log(json!({"event": "exit", "since_launch_ms": launched.elapsed().as_secs_f64() * 1e3}));
    Ok(())
}
