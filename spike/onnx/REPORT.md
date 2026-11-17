# Spike: EmbeddingGemma 2 through ONNX Runtime from Rust, instead of the Python worker

Measured on 2026-11-15, Windows 11, Ryzen 5 5600, RTX 5060 Ti 16 GB (driver 617.42), with the
Seekgram service stopped. Desktop apps were still running (Chrome, NVIDIA Broadcast at about
12 % GPU). The speed numbers are single runs, not averages.

## Verdict: go, with caveats

The Rust program (`ort` 2.0.0-rc.15 + `tokenizers` + `mozjpeg`/`image` + ffmpeg as a child
process) reproduces the worker's preprocessing **bit for bit** on all 63 test requests: token
ids, pixel tensors, mel features, masks and positions are identical. Its vectors agree with the
live Python worker at cosine >= 0.9975 (audio) and >= 0.9999 (text, image, video). That gap is
the worker's own bfloat16 rounding: PyTorch in fp32 differs from the worker by the same amount,
and ONNX fp32 matches PyTorch fp32 to >= 0.99998. For every text query the top-1 photo and the
top-3 set are the same for both systems (20/20).

**DirectML works for all four modalities** after two exact rewrites of the published graphs
(below). It needs only `onnxruntime.dll` and `DirectML.dll` (15 MB to download). On images it
ran at 3.6 images/s (fp16) against the Python worker's 4.2. CUDA works too, but needs 0.9 to
1.4 GB of NVIDIA DLLs and a Turing (GTX 16xx / RTX 20xx) or newer GPU. CPU works anywhere, but at
0.28 images/s it is a last resort.

Caveats: (1) DirectML ran on one GPU only (this NVIDIA card). Behaviour on AMD/Intel/older NVIDIA
DX12 GPUs is untested, and that is the main remaining risk. (2) The DirectML fix patches
the onnx-community graphs, so Seekgram would ship its own patched `.onnx` protos (the weight
files stay unchanged). (3) VRAM peaks are higher than the worker's on long videos, because the
exported graphs do not fuse attention.

## 1. Fidelity: cosine against the Python worker (the oracle)

Reference: the installed worker (`D:/Seekgram/worker`, transformers 5.23, sentence-transformers
6.3, bf16 on CUDA) run as a process exactly as the service runs it (`scripts/run_oracle.py`).
Test set (`media/requests.jsonl`, list in `media/README.md`): 20 text queries (English, Russian
x9, German, Spanish, French, Japanese, Arabic, emoji, a 540-character text), 23 images (16 photos
plus PNG/WebP/greyscale/alpha/EXIF-rotated/96x64/3000x300 cases), 10 audio chunks (sweep,
pink noise, music-like chords x2, formant "speech-like", Opus 3.7 s, near-silence, Windows-SAPI
English and Russian speech), 10 video chunks (testsrc2 with audio, mandelbrot, three slideshow
chunks, a portrait 29.97 fps Ken Burns clip with music, 58 s VP9, 0.4 s, Game of Life, a
12.5-27.25 s cut). Retrieval: each of the 16 photo-matched queries (plus 4 others) ranks the 16
photos with both systems.

| run (provider, precision) | modality | n | min | median | mean | top-1 agree | top-3 set agree |
|--|--|--|--|--|--|--|--|
| **DirectML fp16** | text | 20 | 0.99992 | 0.99997 | 0.99997 | 20/20 | 20/20 |
| | image | 23 | 0.99989 | 0.99994 | 0.99993 | | |
| | audio | 10 | 0.99756 | 0.99967 | 0.99944 | | |
| | video | 10 | 0.99988 | 0.99995 | 0.99994 | | |
| **DirectML fp32** | text | 20 | 0.99992 | 0.99997 | 0.99997 | 20/20 | 20/20 |
| | image | 23 | 0.99989 | 0.99995 | 0.99994 | | |
| | audio | 10 | 0.99774 | 0.99969 | 0.99946 | | |
| | video | 10 | 0.99989 | 0.99996 | 0.99995 | | |
| **CPU fp32** | text | 20 | 0.99992 | 0.99997 | 0.99997 | 20/20 | 20/20 |
| | image | 23 | 0.99989 | 0.99995 | 0.99994 | | |
| | audio | 10 | 0.99774 | 0.99969 | 0.99946 | | |
| | video | 10 | 0.99989 | 0.99996 | 0.99995 | | |
| CUDA 12 fp16 | text / image / audio / video | 20/23/10/10 | 0.99992 / 0.99990 / 0.99772 / 0.99989 | 0.99997 / 0.99994 / 0.99969 / 0.99995 | 0.99997 / 0.99994 / 0.99946 / 0.99994 | 20/20 | 20/20 |
| CUDA 12 fp32 | text / image / audio / video | 20/23/10/10 | 0.99992 / 0.99988 / 0.99765 / 0.99989 | 0.99997 / 0.99995 / 0.99969 / 0.99996 | 0.99997 / 0.99994 / 0.99945 / 0.99995 | 20/20 | 20/20 |
| CUDA 13 fp16 | text / image / audio / video | 20/23/10/10 | 0.99992 / 0.99990 / 0.99745 / 0.99989 | 0.99997 / 0.99994 / 0.99969 / 0.99995 | 0.99997 / 0.99994 / 0.99943 / 0.99994 | 20/20 | 20/20 |
| q8 `quantized` (CUDA and CPU alike) | text / image / audio / video | | 0.99977 / 0.99334 / 0.99755 / 0.99662 | 0.99994 / 0.99881 / 0.99954 / 0.99907 | 0.99993 / 0.99832 / 0.99928 / 0.99890 | 20/20 | 18/20 |
| q4 and q4f16 (CUDA) | text / image / audio / video | | 0.97455 / 0.97705 / 0.94088 / 0.97484 | 0.9919 / 0.9858 / 0.9656 / 0.9850 | 0.9898 / 0.9851 / 0.9640 / 0.9843 | 20/20 | 14/20 |
| *Python itself: PyTorch fp32 (CPU) vs the bf16 worker* | text / image / audio / video | | 0.99992 / 0.99989 / 0.99774 / 0.99989 | 0.99997 / 0.99995 / 0.99969 / 0.99996 | 0.99997 / 0.99994 / 0.99946 / 0.99995 | 20/20 | 20/20 |

Against PyTorch **fp32** instead of the bf16 worker: ONNX CPU fp32 = 1.00000 minimum on every
modality, CUDA fp32 >= 0.99998, CUDA fp16 >= 0.99993 (`runs/fidelity_vs_torch_fp32.md`). So
every remaining difference from the worker is the worker's own bf16 noise. The lowest audio
cosine (a02, music) is the clip on which bf16 drifts most.

**Old vectors can stay in the index** for fp16/fp32 on any provider: the new vectors are as close
to the old ones as the old pipeline is to its own fp32 self, and retrieval order did not change on
the test set. Keep the identity string. Do **not** mix q4/q4f16 vectors (cosine 0.94 to 0.99,
top-3 changed for 6/20 queries); q8 is borderline (image min 0.993) and would need its own
identity.

## 2. Providers: does it load, run and match

| provider | runtime files | fp16 | fp32 | notes |
|--|--|--|--|--|
| **DirectML** (ORT 1.24.4 `Microsoft.ML.OnnxRuntime.DirectML` + `Microsoft.AI.DirectML` 1.15.4) | `onnxruntime.dll` 17.3 MB, `DirectML.dll` 18.5 MB | all 4 graphs load and match **with patched graphs** | same | Unpatched: text works; vision and audio fail (errors below). DirectML's last ORT release is 1.24.4; Microsoft has moved DirectML to maintenance. |
| CPU | the same `onnxruntime.dll` from any package | loads, not measured | all match (= PyTorch fp32 bit for bit in practice) | slow (section 3) |
| CUDA 12 (ORT 1.32.0 cuda12 build, CUDA 12.9 wheels, cuDNN 9.28.1) | 13 DLLs, 2.08 GB on disk | all match | all match | first run 15-19 s (one-time CUDA/cuDNN initialisation), later starts 3 s |
| CUDA 13 (ORT 1.32.0 PyPI wheel, CUDA 13 wheels, cuDNN 9.28.1 cu13) | 12 DLLs, 1.28 GB on disk | all match | all match | needs driver >= 580 |

Errors hit, exactly as printed:

- DirectML, unpatched vision encoder (fp16 and fp32): `Non-zero status code returned while running Reshape node. Name:'node_view_1' Status Message: ...\DmlExecutionProvider\src\MLOperatorAuthorImpl.cpp(2853)\onnxruntime.dll!00007FFB3D3B115C: (caller: 00007FFB3D3B0F54) Exception(1) tid(225e8) 80070057 The parameter is incorrect.` (audio encoder: same, node `node_view`). Cause: `Reshape` with `allowzero=1`, which DirectML rejects. Fix: set `allowzero=0` (128 nodes in vision, 109 in audio). This is exact here because no target dimension is 0.
- DirectML, audio encoder after that fix: `Non-zero status code returned while running MatMul node. Name:'node_matmul' Status Message: ...MLOperatorAuthorImpl.cpp(2508)... 80070057 The parameter is incorrect`. Cause: rank-5 `MatMul` (chunked local attention, `[1,8,63,12,128] x [1,8,63,128,24]`, 24 nodes); DirectML's MatMul takes at most rank 4. Fix: reshape both operands to rank 3, multiply, reshape back. Verified exact on the CPU (max abs difference 0.0, fp16 and fp32) and on DirectML (table above). `scripts/patch_dml.py` applies both fixes.
- CUDA without cuDNN on the search path: `RequireCudnnHandle ... cuDNN is unavailable or disabled for CUDA Execution Provider: LoadLibrary failed for cudnn64_9.dll with error 2`. The provider imports cudart/cuBLAS from its own folder but loads cuDNN with a bare `LoadLibrary`. Fix: `SetDllDirectoryW(runtime dir)` before loading ORT (done in `src/model.rs`), so no PATH change and no installer is needed.
- CUDA fp32, 32-frame videos with all frames in one vision-encoder run: `BFCArena::AllocateRawInternal Failed to allocate memory for requested buffer of size 179994624` (16 GB card full). Cause: unfused attention in the exported graphs (memory grows with frames x patches squared). Fix: at most 4 frames/images per encoder run (`--vision-batch 4`, now the default); same speed, peak 2.0 GB (fp16) / 5.3 GB (fp32) above idle instead of 8.5 / 14.4 GB (`runs/vr_*`).
- q4 on CUDA once died with `memory allocation of 3870720 bytes failed` (host RAM). That happened while a PyTorch CPU reference run was also using memory; the rerun on its own passed.
- `image` crate JPEG decoding (zune-jpeg) differed from Pillow by up to 5/255 (image cosine min 0.99968). Switched to libjpeg via `mozjpeg` (C, no NASM): bit-identical.
- Converting video frames to RGB directly in ffmpeg differed from the worker by up to 3/255 on full-range (`yuvj420p`/`pc`) sources. The worker's FFV1 intermediate squeezes them to limited range. Fix: pipe through FFV1/NUT into a second ffmpeg: bit-identical.

## 3. Speed (single stream: one item per request, like the worker today)

Images: 46 to 230 requests over the 23 test images. Items/s excludes the first 3 requests
(warm-up). These are single runs on a desktop with other apps open.

| provider | images/s | ms per image (preprocess / vision encoder / text model) | process start to first vector |
|--|--|--|--|
| Python worker (bf16, CUDA) | 4.15 (241 ms per request) | | 26-31 s |
| DirectML fp16 | 3.58 | 26 / 100 / 152 | 3.6 s |
| DirectML fp32 | 1.60 | 27 / 415 / 184 | 4.0 s |
| CPU fp32 (6-core Ryzen 5 5600) | 0.28 | 31 / 3190 / 300 | 5.6 s |
| CUDA 12 fp16 | 7.38 | 28 / 93 / 15 | 3.3-19 s (first run ever: 15-19 s) |

For the owner's ~350,000 photos at these single-stream rates: Python about 23 h, DirectML fp16
about 27 h, DirectML fp32 61 h, CPU fp32 about 350 h (two weeks), CUDA fp16 13 h. On DirectML
the text model takes 152 ms per image against 15 ms on CUDA, longer than the vision encoder.
[inference] The likely cause is that the `com.microsoft` contrib ops in the text graph
(`RotaryEmbedding`, `SkipSimplifiedLayerNormalization`) fall back to the CPU, with copies on
every layer. That points to a cheap speed-up nobody tried yet. The batching sweep was stopped by
the scope change; a partial CUDA fp16 run reached 11 images/s with 8 images per run and 6
preprocessing threads. Other single-stream numbers on CUDA fp16: audio 30 s chunks 9.8/s
(Python 4.1/s), video 30 s chunks 0.50/s (Python 0.48/s).

VRAM (whole-GPU peak above the pre-run level, nvidia-smi): Python worker 2.4 GB (images),
3.1 GB (videos). CUDA fp16: 2.4 GB (images), 3.3 GB (videos, vision batch 4). DirectML: 1.4 GB
(fp16, images only), 2.7 GB (fp32, images only); the full test set including 32-frame videos and
30 s audio peaked at 6.3 GB (fp16) and 8.9 GB (fp32). This is not yet tuned for small cards.

## 4. Footprint

Today: about 4.7 GB to download (Python, PyTorch cu128 and friends, the model). On disk here:
`.venv` 5.07 GB + model 1.52 GB + tools 0.30 GB.

| route | runtime download (only the needed DLLs, from zip members) | runtime on disk | model download (fp16 / fp32) | ffmpeg | **total download** |
|--|--|--|--|--|--|
| DirectML + fp16 | 15 MB (from 215 MB of nupkgs if fetched whole) | 36 MB | 1,497 MB | 106 MB (zip, already fetched today) | **about 1.6 GB** |
| DirectML + fp32 | 15 MB | 36 MB | 2,961 MB | 106 MB | about 3.1 GB |
| CPU only | 6 MB (`onnxruntime.dll`; the CPU runs here used the CUDA package's copy; [inference] the DirectML package's copy also runs on the CPU) | 17-21 MB | 1,497 / 2,961 MB | 106 MB | about 1.6 / 3.1 GB |
| CUDA 13 + fp16 | 897 MB (995 MB whole archives) | 1,276 MB | 1,497 MB | 106 MB | about 2.5 GB |
| CUDA 12 + fp16 | 1,377 MB (1,716 MB whole archives) | 2,085 MB | 1,497 MB | 106 MB | about 3.0 GB |

Model sizes include `tokenizer.json` (32 MB); `q8`: 882 MB, `q4`: 505 MB (not recommended,
section 1). Breakdown per DLL: `scripts/archive_members.py`.

CUDA details (footnote): CUDA 12 needs `onnxruntime.dll`, `onnxruntime_providers_shared.dll`,
`onnxruntime_providers_cuda.dll` (GitHub `onnxruntime-win-x64-gpu_cuda12-1.32.0.zip`),
`cudart64_12.dll` (`nvidia-cuda-runtime-cu12` 12.9.79), `cublas64_12.dll` and `cublasLt64_12.dll`
(`nvidia-cublas-cu12` 12.9.2.10), and `cudnn64_9`, `cudnn_graph64_9`, `cudnn_ops64_9`,
`cudnn_engines_precompiled64_9`, `cudnn_heuristic64_9`, `cudnn_engines_runtime_compiled64_9`,
`cudnn_engines_tensor_ir64_9` (`nvidia-cudnn-cu12` 9.28.1.4). Those are the DLLs that were loaded
over a full run; cufft, curand, nvrtc, nvJitLink, cudnn_adv, cudnn_cnn and cudnn_ext were not
needed. Every one of these is a plain zip member on PyPI or GitHub; no installer. CUDA 13 is the
same set with `-cu13`/13.x wheels (cudart is linked statically into the provider there). Minimum
driver (NVIDIA release notes and the cuDNN 9.28.1 support matrix): CUDA 12 route >= 527.41
(Windows), CUDA 13 route >= 580. cuDNN 9.28.1 supports compute capability 7.5 and newer only
(Turing, i.e. GTX 16xx / RTX 20xx, and later). The MSVC runtime (`msvcp140*.dll`,
`vcruntime140*.dll`) is imported by every ORT build; Seekgram's Rust exe already needs it.

## 5. Preprocessing reimplemented, and confidence

All verified bit-identical against tensors captured from the worker's own `Worker` class
(`scripts/trace_oracle.py`, `scripts/compare_inputs.py`). Details: `NOTES.md`.

| step | confidence | evidence |
|--|--|--|
| Query prompt `task: search result \| query: ` + `<bos>`/`<eos>`, media without prefix | high | identical ids, 20 texts in 7 scripts |
| Placeholder layout (`<|image>`/`<image|>` per image and per video frame, `<|audio>`/`<audio|>`) | high | identical ids, all 63 requests |
| Image decode (Pillow-equivalent JPEG via libjpeg, EXIF transpose, alpha drop, grey->RGB) | high for JPEG/PNG/WebP/greyscale; untested: CMYK JPEG (falls back to the `image` crate, may differ), HEIC, GIF/TIFF | bit-identical, 23 images |
| Aspect-preserving resize to <= 280 x 9 patches, multiples of 48 | high | identical sizes incl. 96x64 upscale and 3000x300 strip |
| torch antialiased bicubic on uint8 (int16 fixed point, two passes) | high | bit-identical |
| Rescale 1/255, 16x16 patchify (row, col, channel), position ids, padding (-1) | high | bit-identical |
| Audio: ffmpeg 16 kHz mono f32, pad to 128, semicausal pad, Hann 320, rfft 512, magnitude, HTK mel 128, log(x+0.001), frame mask, token count | high | bit-identical features and masks, 10 clips incl. 3.7 s and 45 s sources |
| Video: worker's ffmpeg cut, fps=1 (round=up retry), FFV1 intermediate, <=58 frames, uniform subsampling to 32, 140-token frames, audio track ignored | high | bit-identical frames, 10 clips incl. 0.4 s, 58 s, portrait, 29.97 fps, VP9 |
| Mean pooling, 512->768 projection, L2 norm | inside the graph (`sentence_embedding`) | cosine table |

## 6. What remains unknown

- **DirectML on other GPUs** (AMD, Intel iGPU, GTX 10xx, laptops with little VRAM). Only this
  RTX 5060 Ti was tested. This is the single most important follow-up: one run of the 63-request
  set on two or three friends' PCs (`scripts/embed.sh` + `scripts/compare.py`, oracle vectors in
  `runs/oracle_gpu.out.jsonl`).
- DirectML VRAM on small cards: the 32-frame video peak was 6.3 GB above idle; whether a 4 GB card
  copes (vision batch 1 or 2, fewer frames per run) is not measured.
- Pixel formats outside the test set: CMYK JPEG, 10-bit/HDR video, odd audio codecs (they all go
  through the same ffmpeg, so likely fine, but not compared).
- Long-run stability (hours of indexing) and memory growth over 350,000 items: not run.
- Why the first CUDA run after a while took 15-19 s (one-time; the next starts took 3 s).
- Speed on CPU-only machines other than this 6-core one.

## 7. Recommended integration plan

1. **Provider**: DirectML by default (works on any DX12 GPU, 15 MB of runtime). If no DirectML
   adapter is available, or a self-test vector does not match, fall back to CPU. Offer CUDA 13
   only as an optional "fast NVIDIA" download later; speed is not the priority.
2. **Precision**: fp16 (1.5 GB download, matches the old vectors). Keep fp32 as the automatic
   fallback if a DirectML adapter fails the self-test in fp16. On CPU use fp32 (fp16 on CPU is
   not faster and was not tested). Never q4.
3. **Index**: keep the existing vectors and the identity string (section 1). Make the self-test
   compare against a few stored reference vectors (text, image, audio, video) shipped with the app,
   threshold cosine >= 0.999 (0.995 for audio).
4. **Model files**: pin `onnx-community/embeddinggemma-2-ONNX@daa72c51...`, fetch
   `tokenizer.json`, the three `*_fp16.onnx_data` files, and ship Seekgram's own patched `.onnx`
   protos (`scripts/patch_dml.py` output, about 0.75 MB) inside the binary or as a pinned download.
   Verify sha256 (`scripts/model_manifest.tsv`).
5. **Code**: replace `src/service/workers.rs` (the Python process pool) and the Python half of
   `provision.rs` with the code in `src/` here: `vision.rs`, `audio.rs`, `media.rs`, `model.rs`
   (about 900 lines). Run the model in a child process of `seekgram.exe` (an `embed-worker`
   mode speaking the same JSONL), not in the service process, so a driver crash or VRAM leak
   cannot take the service down and memory is released by stopping the process, as today. Keep
   at most 4 images/frames per vision-encoder run. `ort` with `load-dynamic`, `SetDllDirectoryW`
   on the runtime folder.
6. **Delete**: uv, the Python venv, `worker/`, PyTorch: about 5 GB less on disk.
7. **Estimate** [inference]: 3-5 working days for steps 1-6 including tests, plus the
   other-GPU check from section 6, which should come first.

## Reproduce

All commands run from `spike/onnx/`. Long steps run in detached psmux sessions through
`scripts/run_logged.sh <name> <command>`, which writes `runs/<name>.log` and
`runs/<name>.status.json`. `scripts/matrix.sh <name> <plan.tsv>` runs a plan with resume
(`runs/<run>.done`), graceful stop (`touch runs/<name>.stop`), `runs/<name>.events.jsonl` and
`runs/<name>.status.json`.

```bash
bash scripts/download_model.sh dl1 fp16 fp32            # model (resumable, sha256-checked)
bash scripts/download_runtimes.sh rt1                   # ORT/DirectML/CUDA archives + unpack
# assemble runtimes/dml (onnxruntime.dll + DirectML.dll), runtimes/cuda12, runtimes/cuda13 as listed in section 4
uv run --no-project --with onnx python scripts/patch_dml.py fp16 fp32
bash scripts/make_media.sh; powershell -File scripts/make_speech.ps1
D:/Seekgram/worker/.venv/Scripts/python.exe -B scripts/make_exif.py
D:/Seekgram/worker/.venv/Scripts/python.exe -B scripts/make_requests.py
D:/Seekgram/worker/.venv/Scripts/python.exe -B scripts/make_bench.py
cargo build --release
psmux new-session -d -s oracle -- bash scripts/run_logged.sh oracle_session \
  D:/Seekgram/worker/.venv/Scripts/python.exe -B scripts/run_oracle.py oracle_gpu media/requests.jsonl 2
psmux new-session -d -s final -- bash scripts/run_logged.sh final_session bash scripts/matrix.sh final scripts/plan_final.tsv
D:/Seekgram/worker/.venv/Scripts/python.exe -B scripts/compare.py runs/oracle_gpu.out.jsonl runs/fid_dmlp_fp16.out.jsonl
D:/Seekgram/worker/.venv/Scripts/python.exe -B scripts/summarize.py sp_dml_fp16_image_single sp_cpu_fp32_image_single
```

Program usage: `onnx-spike --ort <dir>/onnxruntime.dll --model-dir <model dir> --ffmpeg <ffmpeg.exe>
--provider dml|cuda|cpu --variant fp16|fp32|q4|q4f16|quantized [--timing] [--stats]
[--graphs text,vision,audio] [--vision-batch 4] [--dump-dir d]`, worker JSONL on stdin/stdout.
The model, runtimes and downloaded photos are not committed (`.gitignore`); `runs/` holds every
raw result referenced here.
