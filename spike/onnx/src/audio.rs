//! Audio preprocessing: a port of `Gemma4AudioFeatureExtractor` (transformers 5.23) with the
//! model's `preprocessor_config.json` values (16 kHz, 20 ms frames, 10 ms hop, 512-point FFT,
//! 128 HTK mel filters 0-8000 Hz, log(mel + 0.001), no pre-emphasis, no dither).

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

pub const SAMPLE_RATE: usize = 16_000;
pub const MEL_BINS: usize = 128;
/// `max_length` of the feature extractor: 30 s. Longer input is truncated.
pub const MAX_SAMPLES: usize = 480_000;
const FRAME: usize = 320;
const HOP: usize = 160;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const PAD_MULTIPLE: usize = 128;
const MEL_FLOOR: f64 = 0.001;

pub struct Features {
    /// `frames * MEL_BINS` log-mel values; rows of invalid frames are 0.
    pub data: Vec<f32>,
    /// One flag per frame: its whole analysis window lies in real audio.
    pub mask: Vec<bool>,
    /// Soft tokens the audio encoder returns for this clip.
    pub tokens: usize,
}

pub struct Extractor {
    fft: Arc<dyn Fft<f64>>,
    window: [f32; FRAME],
    /// Per mel filter: first FFT bin with a non-zero weight and the weights from there.
    filters: Vec<(usize, Vec<f64>)>,
}

impl Extractor {
    pub fn new() -> Self {
        let mut window = [0f32; FRAME];
        for (n, value) in window.iter_mut().enumerate() {
            // np.hanning(FRAME + 1)[:-1], i.e. the periodic Hann window, stored as float32.
            *value = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FRAME as f64).cos()) as f32;
        }
        // transformers.audio_utils.mel_filter_bank(257, 128, 0, 8000, 16000, norm=None, mel_scale="htk")
        let to_mel = |hertz: f64| 2595.0 * (1.0 + hertz / 700.0).log10();
        let to_hertz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
        let (mel_min, mel_max) = (to_mel(0.0), to_mel(8000.0));
        let points: Vec<f64> = (0..MEL_BINS + 2)
            .map(|i| to_hertz(mel_min + (mel_max - mel_min) * i as f64 / (MEL_BINS + 1) as f64))
            .collect();
        let bin_hertz = |bin: usize| (SAMPLE_RATE / 2) as f64 * bin as f64 / (BINS - 1) as f64;
        let filters = (0..MEL_BINS)
            .map(|m| {
                let weights: Vec<f64> = (0..BINS)
                    .map(|bin| {
                        let down = (bin_hertz(bin) - points[m]) / (points[m + 1] - points[m]);
                        let up = (points[m + 2] - bin_hertz(bin)) / (points[m + 2] - points[m + 1]);
                        down.min(up).max(0.0)
                    })
                    .collect();
                let first = weights.iter().position(|&w| w != 0.0).unwrap_or(0);
                let last = weights.iter().rposition(|&w| w != 0.0).map_or(first, |i| i + 1);
                (first, weights[first..last].to_vec())
            })
            .collect();
        Self { fft: FftPlanner::new().plan_fft_forward(FFT), window, filters }
    }

    /// `samples`: mono 16 kHz float PCM.
    pub fn extract(&self, samples: &[f32]) -> Features {
        let real = samples.len().min(MAX_SAMPLES);
        let padded = real.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;
        // Semicausal padding: FRAME / 2 zeros in front, so frame 0 is centred on t = 0.
        let left = FRAME / 2;
        let total = left + padded;
        let frames = if total > FRAME { (total - (FRAME + 1)) / HOP + 1 } else { 0 };
        let sample = |index: usize| if index >= left && index - left < real { samples[index - left] } else { 0.0 };
        let mut data = vec![0f32; frames * MEL_BINS];
        let mut mask = vec![false; frames];
        let mut buffer = vec![Complex::new(0f64, 0f64); FFT];
        let mut scratch = vec![Complex::new(0f64, 0f64); self.fft.get_inplace_scratch_len()];
        let mut magnitude = [0f64; BINS];
        for frame in 0..frames {
            // A frame is valid when the last sample of its 321-sample unfold window is real audio.
            let end = frame * HOP + FRAME;
            mask[frame] = end >= left && end - left < real;
            if !mask[frame] {
                continue;
            }
            for (n, slot) in buffer.iter_mut().enumerate() {
                let value = if n < FRAME { sample(frame * HOP + n) * self.window[n] } else { 0.0 };
                *slot = Complex::new(f64::from(value), 0.0);
            }
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            for (bin, value) in magnitude.iter_mut().enumerate() {
                *value = buffer[bin].norm();
            }
            for (m, (first, weights)) in self.filters.iter().enumerate() {
                let mel: f64 = weights.iter().zip(&magnitude[*first..]).map(|(w, x)| w * x).sum();
                data[frame * MEL_BINS + m] = (mel + MEL_FLOOR).ln() as f32;
            }
        }
        let tokens = soft_tokens(&mask);
        Features { data, mask, tokens }
    }
}

/// `EmbeddingGemma2Processor.compute_sscp`: the encoder's two stride-2 convolutions (kernel 3,
/// padding 1) keep every 4th frame; a soft token is valid when that frame is.
pub fn soft_tokens(mask: &[bool]) -> usize {
    let mut current: Vec<bool> = mask.to_vec();
    for _ in 0..2 {
        let keep = if current.is_empty() { 0 } else { (current.len() - 1) / 2 + 1 };
        current = current.iter().step_by(2).take(keep).copied().collect();
    }
    current.iter().filter(|&&valid| valid).count()
}
