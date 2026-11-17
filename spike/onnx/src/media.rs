//! Audio and video decoding through an ffmpeg child process, with the worker's arguments and
//! limits (`worker.py`): audio as mono 16 kHz float PCM of at most 30 s, video as 1 frame per
//! second, at most 58 frames.

use std::{
    path::Path,
    process::{Command, Stdio},
};

use anyhow::{Result, bail};

use crate::{audio::MAX_SAMPLES, vision::Rgb};

/// The processor keeps at most this many frames of a clip (`video_processor.max_frames`).
const MAX_FRAMES: usize = 32;
/// The worker rejects clips that sample to more frames than this.
const MAX_SAMPLED_FRAMES: usize = 58;

/// An input the caller can skip (the worker's `RequestError`).
#[derive(Debug)]
pub struct MediaError(pub &'static str);

impl std::fmt::Display for MediaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for MediaError {}

fn command(program: &Path, path: &Path, start: f64, length: Option<f64>, output: &[&str]) -> Command {
    let mut command = Command::new(program);
    command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-protocol_whitelist", "file,pipe", "-ss"])
        .arg(python_float(start))
        .arg("-i")
        .arg(path);
    if let Some(length) = length {
        command.arg("-t").arg(python_float(length));
    }
    command.args(output).stdin(Stdio::null()).stderr(Stdio::null());
    command
}

fn ffmpeg(program: &Path, path: &Path, start: f64, length: Option<f64>, output: &[&str]) -> Result<Vec<u8>> {
    let result = command(program, path, start, length, output).output().map_err(|_| MediaError("ffmpeg could not be started"))?;
    if !result.status.success() {
        bail!(MediaError("ffmpeg failed to decode media"));
    }
    Ok(result.stdout)
}

/// Samples frames like the worker and returns them as RGB.
///
/// The worker encodes the sampled frames to FFV1 and PyAV converts them to RGB. FFV1 negotiation
/// changes pixel formats on the way (full-range `yuvj420p` / `pc` sources are squeezed to
/// limited range), so converting the source to RGB directly differs by up to 3/255 per channel.
/// Piping the same FFV1 stream (in NUT) into a second ffmpeg that writes PPM reproduces PyAV's
/// `to_ndarray(format="rgb24")` bit for bit on the test clips.
fn sampled_frames(program: &Path, path: &Path, start: f64, length: Option<f64>, filter: &str) -> Result<Vec<Rgb>> {
    let started = || MediaError("ffmpeg could not be started");
    let mut encoder = command(program, path, start, length, &["-an", "-vf", filter, "-frames:v", "59", "-c:v", "ffv1", "-f", "nut", "pipe:1"])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|_| started())?;
    let pipe = encoder.stdout.take().ok_or_else(started)?;
    let decoded = Command::new(program)
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-f", "nut", "-i", "pipe:0", "-c:v", "ppm", "-f", "image2pipe", "pipe:1"])
        .stdin(Stdio::from(pipe))
        .stderr(Stdio::null())
        .output()
        .map_err(|_| started())?;
    let encoded = encoder.wait().map_err(|_| started())?;
    if !encoded.success() {
        bail!(MediaError("ffmpeg failed to decode media"));
    }
    if !decoded.status.success() {
        // A NUT stream without frames ("EOF before video frames") is the worker's "frame=0":
        // the caller retries with round=up. Anything else is a decode failure.
        if decoded.stdout.is_empty() {
            return Ok(Vec::new());
        }
        bail!(MediaError("ffmpeg failed to decode media"));
    }
    parse_ppm_stream(&decoded.stdout)
}

/// `str(float)` as Python prints it for the values used here (`0.0`, `12.5`, `30.001`).
fn python_float(value: f64) -> String {
    if value.fract() == 0.0 { format!("{value:.1}") } else { value.to_string() }
}

pub fn audio(program: &Path, path: &Path, start: f64, end: Option<f64>) -> Result<Vec<f32>> {
    if end.is_some_and(|end| end - start > 30.0) {
        bail!(MediaError("audio segment exceeds 30 seconds"));
    }
    let length = end.map_or(30.001, |end| (end - start).min(30.001));
    let bytes = ffmpeg(program, path, start, Some(length), &["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"])?;
    if bytes.is_empty() {
        bail!(MediaError("audio segment contains no samples"));
    }
    if bytes.len() > MAX_SAMPLES * 4 {
        bail!(MediaError("audio segment exceeds 30 seconds"));
    }
    Ok(bytes.chunks_exact(4).map(|sample| f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]])).collect())
}

/// Frames at 1 fps, then the processor's uniform subsampling to at most 32 frames.
pub fn video(program: &Path, path: &Path, start: f64, end: Option<f64>) -> Result<Vec<Rgb>> {
    let length = end.map(|end| end - start);
    let sample = |filter: &str| sampled_frames(program, path, start, length, filter);
    let mut frames = sample("fps=1")?;
    if frames.is_empty() {
        // The default fps filter drops clips shorter than half a second.
        frames = sample("fps=1:round=up")?;
    }
    if frames.is_empty() {
        bail!(MediaError("video segment contains no sampled frames"));
    }
    if frames.len() > MAX_SAMPLED_FRAMES {
        bail!(MediaError("video segment exceeds 58 sampled frames"));
    }
    if frames.len() > MAX_FRAMES {
        // np.linspace(0, n - 1, 32, dtype=int)
        let step = (frames.len() - 1) as f64 / (MAX_FRAMES - 1) as f64;
        let last = frames.len() - 1;
        let mut keep: Vec<usize> = (0..MAX_FRAMES).map(|i| (i as f64 * step) as usize).collect();
        keep[MAX_FRAMES - 1] = last;
        let mut slots: Vec<Option<Rgb>> = frames.into_iter().map(Some).collect();
        frames = keep.into_iter().filter_map(|index| slots[index].take()).collect();
    }
    Ok(frames)
}

/// Splits concatenated binary PPM (`P6`, maxval 255) images.
fn parse_ppm_stream(mut bytes: &[u8]) -> Result<Vec<Rgb>> {
    fn token<'a>(bytes: &mut &'a [u8]) -> Option<&'a [u8]> {
        let start = bytes.iter().position(|byte| !byte.is_ascii_whitespace())?;
        let rest = &bytes[start..];
        let end = rest.iter().position(|byte| byte.is_ascii_whitespace())?;
        *bytes = &rest[end..];
        Some(&rest[..end])
    }
    fn number(bytes: &mut &[u8]) -> Option<usize> {
        std::str::from_utf8(token(bytes)?).ok()?.parse().ok()
    }
    let mut frames = Vec::new();
    while !bytes.is_empty() {
        let header = (token(&mut bytes), number(&mut bytes), number(&mut bytes), number(&mut bytes));
        let (Some(b"P6"), Some(width), Some(height), Some(255)) = header else {
            bail!(MediaError("ffmpeg failed to decode media"));
        };
        // Exactly one whitespace byte separates the header from the pixels.
        let size = width * height * 3;
        if bytes.len() < 1 + size {
            bail!(MediaError("ffmpeg failed to decode media"));
        }
        frames.push(Rgb { width, height, data: bytes[1..1 + size].to_vec() });
        bytes = &bytes[1 + size..];
    }
    Ok(frames)
}
