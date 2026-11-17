//! The three ONNX graphs of `onnx-community/embeddinggemma-2-ONNX` and the glue between them:
//! placeholder token sequences, the encoder runs and the text model that pools and normalizes.

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use ort::{
    environment::Environment,
    ep,
    session::{RunOptions, Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use tokenizers::Tokenizer;

use crate::{
    audio::{Features, MEL_BINS},
    vision::{FRAME_TOKENS, IMAGE_TOKENS, PATCH_DIM, POOL, Patches},
};

/// `config_sentence_transformers.json` prompt `SearchQuery`.
pub const QUERY_PREFIX: &str = "task: search result | query: ";
const HIDDEN: usize = 512;
pub const DIMENSION: usize = 768;
const BOS: i64 = 2;
const EOS: i64 = 1;
const PAD: i64 = 0;
const BOI: i64 = 255_999;
const BOA: i64 = 256_000;
const IMAGE: i64 = 258_880;
const AUDIO: i64 = 258_881;
const EOI: i64 = 258_882;
const EOA: i64 = 258_883;
const VIDEO: i64 = 258_884;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provider {
    Cuda,
    DirectMl,
    Cpu,
}

/// One preprocessed input.
pub enum Item {
    Text(String),
    Image(Patches),
    Audio(Features),
    Video(Vec<Patches>),
}

pub struct Options {
    pub directory: PathBuf,
    /// File-name suffix of the precision variant: `""` (fp32), `"_fp16"`, `"_q4"`, `"_q4f16"`, `"_quantized"`.
    pub suffix: String,
    pub provider: Provider,
    /// CPU intra-op threads; 0 leaves ONNX Runtime's default.
    pub threads: usize,
    /// Most images / video frames per vision-encoder run.
    pub vision_batch: usize,
    pub load_vision: bool,
    pub load_audio: bool,
    /// CUDA: grow the memory arena by exactly what is requested instead of powers of two.
    pub arena_exact: bool,
    /// Give unused arena memory back to the device after every run.
    pub shrink_arena: bool,
    /// Write ONNX Runtime JSON profiles with this path prefix (one file per graph).
    pub profile: Option<PathBuf>,
}

pub struct Model {
    tokenizer: Tokenizer,
    text: Session,
    vision: Option<Session>,
    audio: Option<Session>,
    vision_batch: usize,
    run_options: RunOptions,
    /// Milliseconds spent creating each session, by graph name.
    pub load_ms: Vec<(&'static str, f64)>,
}

/// Where the time of one `embed` call went, in milliseconds.
#[derive(Default, Clone, Copy)]
pub struct Timing {
    pub encoder_ms: f64,
    pub text_ms: f64,
}

fn session(environment: &Environment, options: &Options, graph: &str) -> Result<Session> {
    let path = options.directory.join("onnx").join(format!("{graph}{}.onnx", options.suffix));
    let mut builder = Session::builder(environment).map_err(|e| anyhow!("{e}"))?;
    builder = builder.with_optimization_level(GraphOptimizationLevel::All).map_err(|e| anyhow!("{e}"))?;
    if let Some(prefix) = &options.profile {
        let mut path = prefix.clone().into_os_string();
        path.push(format!("_{graph}"));
        builder = builder.with_profiling(path).map_err(|e| anyhow!("{e}"))?;
    }
    if options.threads > 0 {
        builder = builder.with_intra_threads(options.threads).map_err(|e| anyhow!("{e}"))?;
    }
    builder = match options.provider {
        Provider::Cuda => builder
            .with_execution_providers([{
                let mut cuda = ep::CUDA::default().with_device_id(0);
                if options.arena_exact {
                    cuda = cuda.with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested);
                }
                cuda.build().error_on_failure()
            }])
            .map_err(|e| anyhow!("{e}"))?,
        Provider::DirectMl => builder
            // DirectML requires sequential execution without memory-pattern planning.
            .with_memory_pattern(false)
            .map_err(|e| anyhow!("{e}"))?
            .with_parallel_execution(false)
            .map_err(|e| anyhow!("{e}"))?
            .with_execution_providers([ep::DirectML::default().with_device_id(0).build().error_on_failure()])
            .map_err(|e| anyhow!("{e}"))?,
        Provider::Cpu => builder,
    };
    builder.commit_from_file(&path).map_err(|e| anyhow!("{graph}: {e}")).with_context(|| path.display().to_string())
}

impl Model {
    pub fn load(environment: &Environment, options: &Options) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(options.directory.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let mut load_ms = Vec::new();
        let mut timed = |graph: &'static str| -> Result<Session> {
            let started = Instant::now();
            let session = session(environment, options, graph)?;
            load_ms.push((graph, started.elapsed().as_secs_f64() * 1e3));
            Ok(session)
        };
        let text = timed("model")?;
        let vision = if options.load_vision { Some(timed("vision_encoder")?) } else { None };
        let audio = if options.load_audio { Some(timed("audio_encoder")?) } else { None };
        let mut run_options = RunOptions::new().map_err(|e| anyhow!("{e}"))?;
        if options.shrink_arena {
            let devices = if options.provider == Provider::Cuda { "cpu:0;gpu:0" } else { "cpu:0" };
            run_options.set("memory.enable_memory_arena_shrinkage", devices).map_err(|e| anyhow!("{e}"))?;
        }
        Ok(Self { tokenizer, text, vision, audio, vision_batch: options.vision_batch.max(1), run_options, load_ms })
    }

    /// The token ids of one item; media become runs of placeholder ids.
    pub fn tokens(&self, item: &Item) -> Result<Vec<i64>> {
        let block = |ids: &mut Vec<i64>, begin: i64, placeholder: i64, count: usize, end: i64| {
            ids.push(begin);
            ids.extend(std::iter::repeat_n(placeholder, count));
            ids.push(end);
        };
        Ok(match item {
            Item::Text(text) => {
                // The tokenizer's post-processor adds <bos> and <eos>.
                let encoding = self.tokenizer.encode(format!("{QUERY_PREFIX}{text}"), true).map_err(|e| anyhow!("tokenizer: {e}"))?;
                encoding.get_ids().iter().map(|&id| i64::from(id)).collect()
            }
            Item::Image(patches) => {
                let mut ids = vec![BOS];
                block(&mut ids, BOI, IMAGE, patches.tokens, EOI);
                ids.push(EOS);
                ids
            }
            Item::Audio(features) => {
                let mut ids = vec![BOS];
                block(&mut ids, BOA, AUDIO, features.tokens, EOA);
                ids.push(EOS);
                ids
            }
            Item::Video(frames) => {
                let mut ids = vec![BOS];
                for frame in frames {
                    block(&mut ids, BOI, VIDEO, frame.tokens, EOI);
                }
                ids.push(EOS);
                ids
            }
        })
    }

    /// Runs the vision encoder over `patches` (all padded to `max_tokens * 9` rows) in chunks.
    fn encode_vision(&mut self, patches: &[&Patches], max_tokens: usize) -> Result<Vec<f32>> {
        let session = self.vision.as_mut().context("vision encoder not loaded")?;
        let rows = max_tokens * POOL * POOL;
        let mut features = Vec::new();
        for chunk in patches.chunks(self.vision_batch) {
            let mut pixels = Vec::with_capacity(chunk.len() * rows * PATCH_DIM);
            let mut positions = Vec::with_capacity(chunk.len() * rows * 2);
            for item in chunk {
                pixels.extend_from_slice(&item.pixels);
                positions.extend_from_slice(&item.positions);
            }
            let pixels = Tensor::from_array(([chunk.len(), rows, PATCH_DIM], pixels)).map_err(|e| anyhow!("{e}"))?;
            let positions = Tensor::from_array(([chunk.len(), rows, 2], positions)).map_err(|e| anyhow!("{e}"))?;
            let outputs = session
                .run_with_options(ort::inputs!["pixel_values" => pixels, "pixel_position_ids" => positions], &self.run_options)
                .map_err(|e| anyhow!("vision encoder: {e}"))?;
            let (shape, data) = outputs["image_features"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
            let expected: usize = chunk.iter().map(|item| item.tokens).sum();
            if shape.len() != 2 || shape[0] as usize != expected || shape[1] as usize != HIDDEN {
                bail!("vision encoder returned shape {shape:?}, expected [{expected}, {HIDDEN}]");
            }
            features.extend_from_slice(data);
        }
        Ok(features)
    }

    /// Runs the audio encoder over clips padded to the longest one.
    fn encode_audio(&mut self, clips: &[&Features]) -> Result<Vec<f32>> {
        let session = self.audio.as_mut().context("audio encoder not loaded")?;
        let frames = clips.iter().map(|clip| clip.mask.len()).max().unwrap_or(0);
        let mut data = vec![0f32; clips.len() * frames * MEL_BINS];
        let mut mask = vec![false; clips.len() * frames];
        for (index, clip) in clips.iter().enumerate() {
            data[index * frames * MEL_BINS..][..clip.data.len()].copy_from_slice(&clip.data);
            mask[index * frames..][..clip.mask.len()].copy_from_slice(&clip.mask);
        }
        let data = Tensor::from_array(([clips.len(), frames, MEL_BINS], data)).map_err(|e| anyhow!("{e}"))?;
        let mask = Tensor::from_array(([clips.len(), frames], mask)).map_err(|e| anyhow!("{e}"))?;
        let outputs = session
            .run_with_options(ort::inputs!["input_features" => data, "input_features_mask" => mask], &self.run_options)
            .map_err(|e| anyhow!("audio encoder: {e}"))?;
        let (shape, values) = outputs["audio_features"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let expected: usize = clips.iter().map(|clip| clip.tokens).sum();
        if shape.len() != 2 || shape[0] as usize != expected || shape[1] as usize != HIDDEN {
            bail!("audio encoder returned shape {shape:?}, expected [{expected}, {HIDDEN}]");
        }
        Ok(values.to_vec())
    }

    /// Embeds a batch; returns one L2-normalized 768-float vector per item.
    pub fn embed(&mut self, items: &[Item]) -> Result<(Vec<Vec<f32>>, Timing)> {
        let started = Instant::now();
        let images: Vec<&Patches> = items.iter().filter_map(|item| if let Item::Image(patches) = item { Some(patches) } else { None }).collect();
        let frames: Vec<&Patches> = items.iter().filter_map(|item| if let Item::Video(frames) = item { Some(frames) } else { None }).flatten().collect();
        let clips: Vec<&Features> = items.iter().filter_map(|item| if let Item::Audio(features) = item { Some(features) } else { None }).collect();
        let image_features = if images.is_empty() { Vec::new() } else { self.encode_vision(&images, IMAGE_TOKENS)? };
        let video_features = if frames.is_empty() { Vec::new() } else { self.encode_vision(&frames, FRAME_TOKENS)? };
        let audio_features = if clips.is_empty() { Vec::new() } else { self.encode_audio(&clips)? };
        let encoder_ms = started.elapsed().as_secs_f64() * 1e3;

        let started = Instant::now();
        let sequences: Vec<Vec<i64>> = items.iter().map(|item| self.tokens(item)).collect::<Result<_>>()?;
        let length = sequences.iter().map(Vec::len).max().unwrap_or(0);
        let mut ids = vec![PAD; items.len() * length];
        let mut mask = vec![0i64; items.len() * length];
        for (row, sequence) in sequences.iter().enumerate() {
            ids[row * length..][..sequence.len()].copy_from_slice(sequence);
            mask[row * length..][..sequence.len()].fill(1);
        }
        let tensor = |values: Vec<f32>| Tensor::from_array(([values.len() / HIDDEN, HIDDEN], values)).map_err(|e| anyhow!("{e}"));
        let outputs = self
            .text
            .run_with_options(ort::inputs![
                "input_ids" => Tensor::from_array(([items.len(), length], ids)).map_err(|e| anyhow!("{e}"))?,
                "attention_mask" => Tensor::from_array(([items.len(), length], mask)).map_err(|e| anyhow!("{e}"))?,
                "image_features" => tensor(image_features)?,
                "video_features" => tensor(video_features)?,
                "audio_features" => tensor(audio_features)?,
            ], &self.run_options)
            .map_err(|e| anyhow!("text model: {e}"))?;
        let (shape, data) = outputs["sentence_embedding"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        if shape.len() != 2 || shape[0] as usize != items.len() || shape[1] as usize != DIMENSION {
            bail!("text model returned shape {shape:?}");
        }
        let vectors = data.chunks_exact(DIMENSION).map(normalized).collect::<Result<Vec<_>>>()?;
        Ok((vectors, Timing { encoder_ms, text_ms: started.elapsed().as_secs_f64() * 1e3 }))
    }
}

/// The worker's final check and renormalization.
fn normalized(vector: &[f32]) -> Result<Vec<f32>> {
    let norm = vector.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 0.0 || vector.iter().any(|x| !x.is_finite()) {
        bail!("model returned a zero or invalid embedding");
    }
    Ok(vector.iter().map(|&x| (f64::from(x) / norm) as f32).collect())
}

/// Loads `onnxruntime.dll` from `path` and creates the process-wide environment; keep it alive.
///
/// The CUDA provider's direct imports (cudart, cuBLAS) resolve from the provider DLL's own
/// directory, but it loads cuDNN later with a bare `LoadLibrary("cudnn64_9.dll")`, which only
/// searches the exe directory, System32 and PATH. Adding the runtime directory to the DLL search
/// path makes the bundled copy win without touching PATH.
pub fn ort_library(path: &Path) -> Result<Environment> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        unsafe extern "system" {
            fn SetDllDirectoryW(path: *const u16) -> i32;
        }
        let directory = path.parent().context("--ort has no parent directory")?;
        let wide: Vec<u16> = directory.as_os_str().encode_wide().chain([0]).collect();
        if unsafe { SetDllDirectoryW(wide.as_ptr()) } == 0 {
            bail!("SetDllDirectoryW failed for {}", directory.display());
        }
    }
    ort::init_from(path).map_err(|e| anyhow!("cannot load {}: {e}", path.display()))?.build().map_err(|e| anyhow!("{e}"))
}
