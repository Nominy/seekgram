//! The three ONNX graphs of `onnx-community/embeddinggemma-2-ONNX` and the glue between them:
//! placeholder token sequences, the encoder runs and the text model that pools and normalizes.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use ort::{
    environment::Environment,
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use tokenizers::Tokenizer;

use super::{
    DIMENSION,
    audio::{Features, MEL_BINS},
    vision::{FRAME_TOKENS, IMAGE_TOKENS, PATCH_DIM, POOL, Patches},
};

/// `config_sentence_transformers.json` prompt `SearchQuery`.
const QUERY_PREFIX: &str = "task: search result | query: ";
const HIDDEN: usize = 512;
const BOS: i64 = 2;
const EOS: i64 = 1;
const BOI: i64 = 255_999;
const BOA: i64 = 256_000;
const IMAGE: i64 = 258_880;
const AUDIO: i64 = 258_881;
const EOI: i64 = 258_882;
const EOA: i64 = 258_883;
const VIDEO: i64 = 258_884;
/// Most images or video frames per vision-encoder run. The exported graphs do not fuse
/// attention, so memory grows with the frames per run (32 frames at once: +8.5 GB); speed
/// does not.
const VISION_BATCH: usize = 4;

/// The graph names; each is `<name><suffix>.onnx` with its weights in `<name><suffix>.onnx_data`.
pub const GRAPHS: [&str; 3] = ["model", "vision_encoder", "audio_encoder"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provider {
    /// DirectML on the DXGI adapter with this index.
    DirectMl(u32),
    Cpu,
}

/// One preprocessed input.
pub enum Item {
    Text(String),
    Image(Patches),
    Audio(Features),
    Video(Vec<Patches>),
}

pub struct Model {
    tokenizer: Tokenizer,
    text: Session,
    vision: Session,
    audio: Session,
}

fn session(environment: &Environment, path: &Path, provider: Provider) -> Result<Session> {
    let mut builder = Session::builder(environment).map_err(|e| anyhow!("{e}"))?;
    builder = builder.with_optimization_level(GraphOptimizationLevel::All).map_err(|e| anyhow!("{e}"))?;
    builder = match provider {
        Provider::DirectMl(device) => builder
            // DirectML requires sequential execution without memory-pattern planning.
            .with_memory_pattern(false)
            .map_err(|e| anyhow!("{e}"))?
            .with_parallel_execution(false)
            .map_err(|e| anyhow!("{e}"))?
            .with_execution_providers([ep::DirectML::default().with_device_id(device as i32).build().error_on_failure()])
            .map_err(|e| anyhow!("{e}"))?,
        Provider::Cpu => builder,
    };
    let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    builder.commit_from_file(path).map_err(|e| anyhow!("{name}: {e}"))
}

impl Model {
    /// Loads the tokenizer and the three graphs `<graph><suffix>.onnx` from `directory`.
    pub fn load(environment: &Environment, directory: &Path, suffix: &str, provider: Provider) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let graph = |name: &str| session(environment, &directory.join(format!("{name}{suffix}.onnx")), provider);
        Ok(Self { tokenizer, text: graph(GRAPHS[0])?, vision: graph(GRAPHS[1])?, audio: graph(GRAPHS[2])? })
    }

    /// The token ids of one item; media become runs of placeholder ids.
    fn tokens(&self, item: &Item) -> Result<Vec<i64>> {
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
        let rows = max_tokens * POOL * POOL;
        let mut features = Vec::new();
        for chunk in patches.chunks(VISION_BATCH) {
            let mut pixels = Vec::with_capacity(chunk.len() * rows * PATCH_DIM);
            let mut positions = Vec::with_capacity(chunk.len() * rows * 2);
            for item in chunk {
                pixels.extend_from_slice(&item.pixels);
                positions.extend_from_slice(&item.positions);
            }
            let pixels = Tensor::from_array(([chunk.len(), rows, PATCH_DIM], pixels)).map_err(|e| anyhow!("{e}"))?;
            let positions = Tensor::from_array(([chunk.len(), rows, 2], positions)).map_err(|e| anyhow!("{e}"))?;
            let outputs = self
                .vision
                .run(ort::inputs!["pixel_values" => pixels, "pixel_position_ids" => positions])
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

    /// Runs the audio encoder over one clip.
    fn encode_audio(&mut self, clip: Features) -> Result<Vec<f32>> {
        let (frames, tokens) = (clip.mask.len(), clip.tokens);
        let data = Tensor::from_array(([1, frames, MEL_BINS], clip.data)).map_err(|e| anyhow!("{e}"))?;
        let mask = Tensor::from_array(([1, frames], clip.mask)).map_err(|e| anyhow!("{e}"))?;
        let outputs = self
            .audio
            .run(ort::inputs!["input_features" => data, "input_features_mask" => mask])
            .map_err(|e| anyhow!("audio encoder: {e}"))?;
        let (shape, values) = outputs["audio_features"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        if shape.len() != 2 || shape[0] as usize != tokens || shape[1] as usize != HIDDEN {
            bail!("audio encoder returned shape {shape:?}, expected [{tokens}, {HIDDEN}]");
        }
        Ok(values.to_vec())
    }

    /// Embeds one item; returns its L2-normalized 768-float vector.
    pub fn embed(&mut self, item: Item) -> Result<Vec<f32>> {
        let ids = self.tokens(&item)?;
        let (mut image_features, mut video_features, mut audio_features) = (Vec::new(), Vec::new(), Vec::new());
        match item {
            Item::Text(_) => {}
            Item::Image(patches) => image_features = self.encode_vision(&[&patches], IMAGE_TOKENS)?,
            Item::Video(frames) => video_features = self.encode_vision(&frames.iter().collect::<Vec<_>>(), FRAME_TOKENS)?,
            Item::Audio(clip) => audio_features = self.encode_audio(clip)?,
        }
        let length = ids.len();
        let tensor = |values: Vec<f32>| Tensor::from_array(([values.len() / HIDDEN, HIDDEN], values)).map_err(|e| anyhow!("{e}"));
        let outputs = self
            .text
            .run(ort::inputs![
                "input_ids" => Tensor::from_array(([1, length], ids)).map_err(|e| anyhow!("{e}"))?,
                "attention_mask" => Tensor::from_array(([1, length], vec![1i64; length])).map_err(|e| anyhow!("{e}"))?,
                "image_features" => tensor(image_features)?,
                "video_features" => tensor(video_features)?,
                "audio_features" => tensor(audio_features)?,
            ])
            .map_err(|e| anyhow!("text model: {e}"))?;
        let (shape, data) = outputs["sentence_embedding"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        if shape.len() != 2 || shape[0] != 1 || shape[1] as usize != DIMENSION {
            bail!("text model returned shape {shape:?}");
        }
        normalized(data)
    }
}

/// The Python worker's final check and renormalization.
fn normalized(vector: &[f32]) -> Result<Vec<f32>> {
    let norm = vector.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 0.0 || vector.iter().any(|x| !x.is_finite()) {
        bail!("model returned a zero or invalid embedding");
    }
    Ok(vector.iter().map(|&x| (f64::from(x) / norm) as f32).collect())
}

/// Loads `onnxruntime.dll` from `runtime` and creates the process-wide environment; keep it
/// alive as long as any session.
///
/// Windows ships an older `DirectML.dll` in System32, and ONNX Runtime asks for it by bare
/// name. Loading the downloaded copy by full path first makes every later request by name
/// get that one; the DLL directory covers anything else the runtime loads by name.
pub fn runtime(runtime: &Path, directml: bool) -> Result<Environment> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows::{
            Win32::System::LibraryLoader::{LoadLibraryW, SetDllDirectoryW},
            core::PCWSTR,
        };
        let wide = |path: &Path| path.as_os_str().encode_wide().chain([0]).collect::<Vec<u16>>();
        let directory = wide(runtime);
        // SAFETY: NUL-terminated wide strings that outlive the calls. A loaded module stays
        // loaded for the life of the process, which is intended.
        unsafe {
            SetDllDirectoryW(PCWSTR(directory.as_ptr())).context("cannot set the DLL directory")?;
            if directml {
                LoadLibraryW(PCWSTR(wide(&runtime.join("DirectML.dll")).as_ptr())).context("cannot load DirectML.dll")?;
            }
        }
    }
    let library = runtime.join("onnxruntime.dll");
    ort::init_from(&library).map_err(|e| anyhow!("cannot load onnxruntime.dll: {e}"))?.build().map_err(|e| anyhow!("{e}"))
}

/// The file a loaded DLL came from, to show in the log which `DirectML.dll` is in use.
#[cfg(windows)]
pub fn loaded_from(module: &str) -> Option<String> {
    use windows::{
        Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW},
        core::PCWSTR,
    };
    let name: Vec<u16> = module.encode_utf16().chain([0]).collect();
    let mut buffer = [0u16; 1024];
    // SAFETY: a NUL-terminated name and a buffer whose length is passed along.
    let length = unsafe {
        let handle = GetModuleHandleW(PCWSTR(name.as_ptr())).ok()?;
        GetModuleFileNameW(Some(handle), &mut buffer)
    } as usize;
    (length > 0).then(|| String::from_utf16_lossy(&buffer[..length]))
}

#[cfg(not(windows))]
pub fn loaded_from(_module: &str) -> Option<String> {
    None
}
