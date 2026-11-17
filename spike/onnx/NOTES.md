# EmbeddingGemma 2: how each modality becomes model input

Sources read: the installed worker environment (`D:/Seekgram/worker/.venv`, transformers 5.23.0,
sentence-transformers 6.3.0): `transformers/models/embedding_gemma2/{processing,video_processing,modeling}_embedding_gemma2.py`,
`transformers/models/gemma4/{feature_extraction,image_processing}_gemma4.py`, `transformers/audio_utils.py`,
`transformers/video_utils.py`, `sentence_transformers/base/modules/transformer.py`; the model files of
`google/embeddinggemma-2@f9972ac` (`config.json`, `processor_config.json`, `config_sentence_transformers.json`,
`modules.json`, `1_Pooling/config.json`, `tokenizer.json`); transformers.js `main`
(`models/embedding_gemma2/*.js`, copies in `runs/tjs/` are not kept); the ONNX graphs'
inputs/outputs/operators (`scripts/inspect_onnx.py`).

Every claim marked **[verified]** was checked by dumping the tensors the Python processor hands to
the model (`scripts/trace_oracle.py`, which wraps the worker's own `Worker` class) and comparing
them with the Rust program's tensors (`--dump-dir`, `scripts/compare_inputs.py`): all 63 test
requests produce identical `input_ids`, identical position ids and masks, and bit-identical
pixel / mel-feature tensors.

## What is inside the ONNX graphs and what the caller must do

| Step | Where |
|--|--|
| Tokenization, prompt, placeholder expansion | caller |
| Image/frame decode, resize, rescale, patchify, position ids, padding | caller |
| Audio decode, resample, log-mel features, frame mask | caller |
| Vision tower (16 layers) + 3x3 pooling + `embed_vision` (RMSNorm without scale, 768->512 linear) | `vision_encoder*.onnx`: in `pixel_values [n, 2520 or 1260, 768] f32`, `pixel_position_ids [n, rows, 2] i64`; out `image_features [soft_tokens, 512]` with padding already stripped |
| Audio tower (subsampling conv x2, 12 conformer layers) + `embed_audio` (1536->512) | `audio_encoder*.onnx`: in `input_features [n, frames, 128] f32`, `input_features_mask [n, frames] bool`; out `audio_features [valid_tokens, 512]` with invalid tokens already dropped |
| Token embedding (x sqrt(512)), replacing placeholder ids by the soft tokens in order (`masked_scatter`), per-layer inputs, 24 bidirectional layers (sliding window 512 on 20 layers), final norm, 512->768 `embedding_projection` per token, mean pooling over `attention_mask`, L2 normalisation | `model*.onnx`: in `input_ids`, `attention_mask` (i64), `image_features`, `video_features`, `audio_features` (each `[k, 512]`, `[0, 512]` when absent); out `last_hidden_state [b, s, 768]`, `sentence_embedding [b, 768]` (already mean-pooled and L2-normalised) |
| Final renormalisation and finiteness check (`worker.py`) | caller (no-op numerically) |

The worker's pooling module (`1_Pooling`: mean, `include_prompt: true`) and `2_Normalize` are
exactly what the graph's `sentence_embedding` output does: the mean runs over every attended
token, including `<bos>`, `<eos>`, the prompt, `<|image>`/`<image|>` markers and the soft tokens.
Projecting per token and then averaging equals averaging and then projecting (linear, no bias).

## Text (search query) [verified]

- sentence-transformers' `prompt_name="SearchQuery"` is `"task: search result | query: "`
  (`config_sentence_transformers.json`). Because the model declares a `message` modality, ST turns
  the text into a chat message, prepends the prompt to the text, and the chat template
  (`chat_template.jinja`) renders the plain string unchanged: the rendered text is exactly
  `"task: search result | query: " + query`.
- `tokenizer.json` (Gemma BPE, normalizer replaces `" "` with `"▁"`, no lowercasing) with the
  post-processor `<bos> A <eos>`: ids `2, ..., 1`. Example: `a bear looking at the camera` ->
  `2 8071 236787 3927 1354 1109 7609 236787 496 10591 3182 657 506 5873 1` (15 tokens).
- No truncation in practice (ST would truncate at the model limit; queries are short). Rust:
  `tokenizers` crate, `encode(prefix + text, add_special_tokens=true)`; identical ids for all 20 test
  queries, including Russian, Japanese, Arabic, emoji and a 540-character text.
- Media inputs get **no prefix** (`worker.py` passes `{kind: value}` without `prompt_name`).

## Image [verified]

1. Worker: `ImageOps.exif_transpose(Image.open(path)).convert("RGB")` (Pillow; JPEG decoded by
   libjpeg-turbo with ISLOW IDCT and fancy upsampling; RGBA -> RGB drops alpha, no compositing;
   L -> RGB replicates).
   Rust: JPEG via `mozjpeg` (libjpeg-turbo fork, same IDCT/upsampling, C build, no NASM), other
   formats via the `image` crate; EXIF orientation from the `image` crate. With the `image` crate's
   own JPEG decoder (zune-jpeg) the pixels differed from Pillow by up to 5/255 (cosine still
   >= 0.99968 on fp16); with libjpeg all 23 test images are bit-identical.
2. `Gemma4ImageProcessor` (torchvision backend), `max_soft_tokens=280`, `patch_size=16`,
   `pooling_kernel_size=3`: `max_patches = 280*9 = 2520`. Target size
   (`get_aspect_ratio_preserving_size`): `factor = sqrt(2520*256 / (h*w))`, then each side is
   `floor(factor*side / 48) * 48`; if one side rounds to 0 it becomes 48 and the other
   `min(floor(long/short)*48, (2520/9)*48 = 13440)`. No tiling, no pan-and-scan, no crop: one
   aspect-preserving resize. E.g. 1600x1067 -> 960x624 (2340 patches, 260 soft tokens),
   3000x300 strip -> 2496x240, 96x64 thumbnail -> upscaled to 960x624.
3. Resize: `torchvision.transforms.v2.functional.resize(uint8 tensor, BICUBIC, antialias=True)`
   which for uint8 CPU tensors is `aten::_upsample_bicubic2d_aa` on uint8: separable,
   horizontal pass then vertical pass, Keys cubic `a=-0.5`, support `2*scale` when downscaling,
   weights normalised then quantised to int16 fixed point with a per-axis precision, each pass
   rounded and clamped to uint8. Ported exactly (`src/vision.rs`): bit-identical output,
   including upscaling (`x17`) and extreme aspect ratios (`x21`).
4. `rescale` by `1/255`, `do_normalize=false` (mean 0, std 1): pixel values in `[0, 1]`.
5. Patchify: `(C, H, W) -> (H/16 * W/16, 16*16*3)`, patches row-major over the image, each patch
   flattened as (row, column, channel). Position ids `(x, y)` = (patch column, patch row).
   Padded to 2520 rows with zeros and position `(-1, -1)`.
6. Soft tokens = real patches / 9 (the encoder's 3x3 pooling), at most 280.
7. Tokens: `<bos> <|image> <|image|> x soft_tokens <image|> <eos>`
   (`2, 255999, 258880..., 258882, 1`), attention mask all ones.

## Audio [verified]

1. Worker: ffmpeg `-ss start -i file -t min(end-start, 30.001) -vn -ac 1 -ar 16000 -f f32le pipe:1`
   (ffmpeg's default resampler), reject more than 480000 samples. Rust runs the same command line.
2. `Gemma4AudioFeatureExtractor` (`preprocessor_config.json`): pad the clip with zeros on the right
   to a multiple of 128 samples (`pad_to_multiple_of=128`, `max_length=480000`, truncation on), mask
   = real samples. Prepend 160 zeros (semicausal padding). Frames of 321 samples every 160
   (`unfold`); `preemphasis=0` so the frame is the first 320 samples; multiply by the periodic
   Hann window `0.5-0.5cos(2 pi n/320)` (float32); `rfft` with n=512 (zero padded); magnitude
   (not power); multiply by a 257x128 HTK mel filter bank (0-8000 Hz, triangular, unnormalised,
   `mel_filter_bank(..., norm=None, mel_scale="htk")`); `log(mel + 0.001)`; no per-bin mean/std,
   no dither. A frame is valid if the last sample of its 321-sample window is real audio; invalid
   frames are zeroed.
   Rust: same arithmetic in f64 with `rustfft` (numpy computes in f64 too, after the f32 window);
   the resulting f32 features are bit-identical on all 10 test clips.
3. Soft tokens: `compute_sscp(mask)`: the mask is subsampled twice by stride 2
   (`mask[::2][:(len+2-3)//2+1]`) and summed; 30 s -> 2999 frames -> 750 tokens (25 per second).
4. Tokens: `<bos> <|audio> <|audio|> x n <audio|> <eos>` (`2, 256000, 258881..., 258883, 1`).
5. The audio encoder graph takes the padded features and the frame mask and returns only the valid
   soft tokens, already projected to 512.

## Video [verified]

1. Worker: ffmpeg cuts the chunk (`-ss start -i file -t length`), `-an -vf fps=1 -frames:v 59`,
   encodes FFV1 into a temporary `.mkv` (retrying with `fps=1:round=up` when 0 frames came out,
   which happens for clips under 0.5 s), rejects more than 58 frames, and passes the file path to
   sentence-transformers. **The audio track of a video is ignored**: ffmpeg drops it (`-an`) and
   the processor's `load_audio_from_video` is false.
2. transformers loads the clip with the `torchvision` backend (`torchvision.io.read_video`, PyAV
   underneath, frames converted with `to_ndarray(format="rgb24")`). Since the clip is already 1 fps,
   its metadata says fps=1, so 1-fps sampling keeps every frame: indices `0..n-1`.
   Then `max_frames=32`, `overflow_strategy="uniform"`: more than 32 frames are cut to
   `np.linspace(0, n-1, 32, dtype=int)` (a 58-frame chunk keeps frames 0, 1, 3, 5, ... 57).
   **Detail that matters**: FFV1 does not take full-range `yuvj420p` / `pc` frames as they are;
   ffmpeg squeezes them to limited range on the way into the clip. Converting the source frames
   to RGB directly differs from the worker by up to 3/255 on JPEG-derived (slideshow) clips.
   Rust therefore pipes the same FFV1 stream (in NUT) into a second ffmpeg that writes PPM: the
   RGB frames are then bit-identical to PyAV's on all 10 test clips.
3. Each frame goes through the image pipeline with the **video** budget `max_soft_tokens=140`
   (max 1260 patches): 640x360 -> 120 soft tokens, 640x480 -> 130, 720x1280 portrait -> 120.
4. Tokens: no timestamps (`add_timestamps=false`), one block per frame:
   `<bos> (<|image> <|video|> x tokens <image|>) x frames <eos>` (`<|video|>` = 258884; the frame
   markers are the image markers 255999 / 258882). 32 frames at 130 tokens = 4226 tokens.
5. All frames of one clip are one vision-encoder batch (`[frames, 1260, 768]`); their soft tokens
   fill the `<|video|>` slots in order.

## Model and numerics

- The worker runs the PyTorch model in **bfloat16** on CUDA (`torch.cuda.is_bf16_supported()`),
  fp32 on CPU. The ONNX fp16 graphs compute in fp16 with fp32 inputs/outputs; fp32 graphs in fp32.
- `sentence_embedding` from the graph is already unit length; the worker normalises once more.
- The graph inputs are fully dynamic (batch, sequence, token counts); several items can be batched
  in one call (texts padded with `<pad>`=0 and mask 0; images stacked; audio padded to the longest
  clip with the mask). Batched results equal unbatched ones up to kernel rounding.

## Files

- `src/vision.rs`: image decode, target size, antialiased bicubic (torch uint8 port), patchify.
- `src/audio.rs`: log-mel feature extractor and token count.
- `src/media.rs`: ffmpeg child processes (audio PCM, video FFV1->PPM), frame subsampling.
- `src/model.rs`: tokens, the three ONNX sessions, batching.
- `src/main.rs`: the worker's JSONL protocol, request validation, bulk mode, stats.
