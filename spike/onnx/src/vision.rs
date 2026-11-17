//! Image and video-frame preprocessing: a port of `Gemma4ImageProcessor` /
//! `EmbeddingGemma2VideoProcessor` (transformers 5.23) including torch's antialiased bicubic
//! resize on uint8 data (`F.interpolate(..., mode="bicubic", antialias=True)` as torchvision calls
//! it for uint8 CPU tensors).

use std::path::Path;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageDecoder, ImageReader};

pub const PATCH: usize = 16;
pub const POOL: usize = 3;
pub const PATCH_DIM: usize = PATCH * PATCH * 3;
/// Soft-token budget of a still image (`image_processor.max_soft_tokens`).
pub const IMAGE_TOKENS: usize = 280;
/// Soft-token budget of one video frame (`video_processor.max_soft_tokens`).
pub const FRAME_TOKENS: usize = 140;

/// An RGB8 image, row-major, 3 bytes per pixel.
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// One image or frame, patchified and padded to `max_patches` rows.
pub struct Patches {
    /// `max_patches * PATCH_DIM` floats in [0, 1]; padding rows are 0.
    pub pixels: Vec<f32>,
    /// `max_patches * 2` (x, y) patch coordinates; padding rows are (-1, -1).
    pub positions: Vec<i64>,
    /// Soft tokens the vision encoder returns for this item.
    pub tokens: usize,
}

/// Decodes like the worker: `ImageOps.exif_transpose(Image.open(path)).convert("RGB")`.
///
/// JPEG goes through libjpeg (the mozjpeg fork of libjpeg-turbo, ISLOW IDCT, fancy upsampling),
/// the same decoder family Pillow uses; the `image` crate's pure-Rust JPEG decoder differs from
/// Pillow by up to 5/255 per channel. Other formats use the `image` crate.
pub fn load_image(path: &Path) -> Result<Rgb> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut decoder = ImageReader::new(std::io::Cursor::new(&bytes))
        .with_guessed_format()?
        .into_decoder()
        .context("cannot decode image")?;
    let orientation = decoder.orientation()?;
    let mut image = match decode_jpeg(&bytes) {
        Some(rgb) => DynamicImage::ImageRgb8(rgb),
        None => DynamicImage::from_decoder(decoder).context("cannot decode image")?,
    };
    image.apply_orientation(orientation);
    let rgb = image.to_rgb8();
    Ok(Rgb { width: rgb.width() as usize, height: rgb.height() as usize, data: rgb.into_raw() })
}

/// `None` when the data is not a JPEG libjpeg can turn into RGB (CMYK/YCCK included).
fn decode_jpeg(bytes: &[u8]) -> Option<image::RgbImage> {
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    std::panic::catch_unwind(|| -> std::io::Result<Option<image::RgbImage>> {
        let decompress = mozjpeg::Decompress::new_mem(bytes)?;
        if matches!(decompress.color_space(), mozjpeg::ColorSpace::JCS_CMYK | mozjpeg::ColorSpace::JCS_YCCK) {
            return Ok(None);
        }
        let mut started = decompress.rgb()?;
        let (width, height) = (started.width() as u32, started.height() as u32);
        let pixels: Vec<[u8; 3]> = started.read_scanlines()?;
        started.finish()?;
        Ok(image::RgbImage::from_raw(width, height, pixels.into_flattened()))
    })
    .ok()?
    .ok()?
}

/// `get_aspect_ratio_preserving_size`: the largest (height, width), both multiples of 48, whose
/// patch count fits the budget.
pub fn target_size(height: usize, width: usize, max_tokens: usize) -> Result<(usize, usize)> {
    let max_patches = max_tokens * POOL * POOL;
    let target_px = (max_patches * PATCH * PATCH) as f64;
    let factor = (target_px / (height * width) as f64).sqrt();
    let side = POOL * PATCH;
    let mut target_h = (factor * height as f64 / side as f64).floor() as usize * side;
    let mut target_w = (factor * width as f64 / side as f64).floor() as usize * side;
    if target_h == 0 && target_w == 0 {
        bail!("image resizes to 0 x 0");
    }
    let max_side = (max_patches / (POOL * POOL)) * side;
    if target_h == 0 {
        target_h = side;
        target_w = ((width / height) * side).min(max_side);
    } else if target_w == 0 {
        target_w = side;
        target_h = ((height / width) * side).min(max_side);
    }
    if (target_h * target_w) as f64 > target_px {
        bail!("resize target exceeds the patch budget");
    }
    Ok((target_h, target_w))
}

fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per-output-index source window and int16 fixed-point weights, as torch's
/// `_compute_index_ranges_int16_weights` builds them for antialiased bicubic.
struct Taps {
    starts: Vec<usize>,
    sizes: Vec<usize>,
    weights: Vec<i32>,
    stride: usize,
    precision: u32,
}

fn taps(input: usize, output: usize) -> Taps {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let stride = support.ceil() as usize * 2 + 1;
    let mut starts = Vec::with_capacity(output);
    let mut sizes = Vec::with_capacity(output);
    let mut float_weights = vec![0f64; output * stride];
    let mut weight_max = 0f64;
    for i in 0..output {
        let center = scale * (i as f64 + 0.5);
        let start = ((center - support + 0.5) as i64).max(0) as usize;
        let size = (((center + support + 0.5) as i64).min(input as i64) - start as i64).clamp(0, stride as i64) as usize;
        let row = &mut float_weights[i * stride..i * stride + size];
        let mut total = 0f64;
        for (j, weight) in row.iter_mut().enumerate() {
            *weight = cubic((j as f64 + start as f64 - center + 0.5) * invscale);
            total += *weight;
        }
        if total != 0.0 {
            for weight in row.iter_mut() {
                *weight /= total;
                weight_max = weight_max.max(*weight);
            }
        }
        starts.push(start);
        sizes.push(size);
    }
    let mut precision = 0u32;
    while precision < 15 {
        let next = (0.5 + weight_max * f64::from(1u32 << (precision + 1))) as i64;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let weights = float_weights
        .iter()
        .map(|&weight| {
            let value = weight * f64::from(1u32 << precision);
            (if value < 0.0 { value - 0.5 } else { value + 0.5 }) as i32
        })
        .collect();
    Taps { starts, sizes, weights, stride, precision }
}

/// Antialiased bicubic resize of RGB8, horizontal pass then vertical pass, each rounded to uint8.
pub fn resize(source: &Rgb, height: usize, width: usize) -> Rgb {
    if source.height == height && source.width == width {
        return Rgb { width, height, data: source.data.clone() };
    }
    let horizontal = taps(source.width, width);
    let mut middle = vec![0u8; source.height * width * 3];
    let round = 1i32 << (horizontal.precision - 1);
    for y in 0..source.height {
        let row = &source.data[y * source.width * 3..(y + 1) * source.width * 3];
        let out = &mut middle[y * width * 3..(y + 1) * width * 3];
        for x in 0..width {
            let start = horizontal.starts[x];
            let weights = &horizontal.weights[x * horizontal.stride..x * horizontal.stride + horizontal.sizes[x]];
            let mut sum = [round; 3];
            for (j, &weight) in weights.iter().enumerate() {
                let pixel = &row[(start + j) * 3..(start + j) * 3 + 3];
                sum[0] += weight * i32::from(pixel[0]);
                sum[1] += weight * i32::from(pixel[1]);
                sum[2] += weight * i32::from(pixel[2]);
            }
            for channel in 0..3 {
                out[x * 3 + channel] = (sum[channel] >> horizontal.precision).clamp(0, 255) as u8;
            }
        }
    }
    let vertical = taps(source.height, height);
    let mut data = vec![0u8; height * width * 3];
    let round = 1i32 << (vertical.precision - 1);
    let row_len = width * 3;
    let mut sums = vec![0i32; row_len];
    for y in 0..height {
        let start = vertical.starts[y];
        let weights = &vertical.weights[y * vertical.stride..y * vertical.stride + vertical.sizes[y]];
        sums.fill(round);
        for (j, &weight) in weights.iter().enumerate() {
            let row = &middle[(start + j) * row_len..(start + j + 1) * row_len];
            for (sum, &value) in sums.iter_mut().zip(row) {
                *sum += weight * i32::from(value);
            }
        }
        for (out, &sum) in data[y * row_len..(y + 1) * row_len].iter_mut().zip(&sums) {
            *out = (sum >> vertical.precision).clamp(0, 255) as u8;
        }
    }
    Rgb { width, height, data }
}

/// Resize to the token budget, rescale to [0, 1], cut into 16x16 patches (row-major; each patch
/// flattened as row, column, channel) and pad to `max_tokens * 9` patches.
pub fn patchify(source: &Rgb, max_tokens: usize) -> Result<Patches> {
    let (height, width) = target_size(source.height, source.width, max_tokens)?;
    let resized = resize(source, height, width);
    let max_patches = max_tokens * POOL * POOL;
    let (rows, columns) = (height / PATCH, width / PATCH);
    let mut pixels = vec![0f32; max_patches * PATCH_DIM];
    let mut positions = vec![-1i64; max_patches * 2];
    for row in 0..rows {
        for column in 0..columns {
            let index = row * columns + column;
            let patch = &mut pixels[index * PATCH_DIM..(index + 1) * PATCH_DIM];
            for y in 0..PATCH {
                let offset = ((row * PATCH + y) * width + column * PATCH) * 3;
                for (out, &value) in patch[y * PATCH * 3..(y + 1) * PATCH * 3].iter_mut().zip(&resized.data[offset..offset + PATCH * 3]) {
                    *out = f32::from(value) * (1.0 / 255.0);
                }
            }
            positions[index * 2] = column as i64;
            positions[index * 2 + 1] = row as i64;
        }
    }
    Ok(Patches { pixels, positions, tokens: rows * columns / (POOL * POOL) })
}
