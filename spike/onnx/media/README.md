# Test media

Nothing here is private data. Regenerate with `scripts/make_media.sh`, `scripts/make_speech.ps1`,
`scripts/make_exif.py`, then `scripts/make_requests.py` (writes `requests.jsonl` with absolute paths)
and `scripts/make_bench.py`.

| file | what |
|--|--|
| `photos/p00..p15_id<N>_<WxH>.jpg` | picsum.photos ids 10, 20, 28, 40, 64, 96, 103, 119, 152, 169, 177, 200, 237, 292, 433, 659 (Unsplash photos: sea and forest, desk, river, cat nose, woman with flowers, game controller, feet in a field, laptop, purple flower, dachshunds, hiker, highland cow, black puppy, vegetables, bear, husky) |
| `photos/x16_testsrc2_1280x720.png` | ffmpeg test pattern |
| `photos/x17_tiny_96x64.png` | downscaled photo (forces upscaling) |
| `photos/x18_gray_1000x1500.jpg` | greyscale JPEG |
| `photos/x19_alpha_800x800.png` | RGBA with a transparent hole |
| `photos/x20_webp_1280x853.webp` | WebP |
| `photos/x21_strip_3000x300.png` | 10:1 strip |
| `photos/x22_exif6_1200x900.jpg` | stored rotated, EXIF Orientation 6 |
| `audio/a0_sweep_30s.wav` | 100 Hz-8 kHz sine sweep, 16 kHz |
| `audio/a1_pink_30s.flac` | pink noise, 44.1 kHz stereo |
| `audio/a2_chords_45s.mp3` | synthesized chord progression with bass, 44.1 kHz stereo MP3 |
| `audio/a3_formant_20s.wav` | speech-like: pitch-varying pulse train through moving formants |
| `audio/a4_short_3.7s.ogg` | 440 Hz tremolo, Opus 48 kHz |
| `audio/a5_quiet_12s.wav` | near-silent white noise |
| `audio/a6_speech_en.wav`, `a7_speech_ru.wav` | Windows SAPI voices (Zira, Irina) reading a weather/museum text |
| `video/v0_testsrc2_30s.mp4` | testsrc2 640x360 30 fps with a sine audio track |
| `video/v1_mandelbrot_30s.mp4` | mandelbrot zoom 854x480 25 fps |
| `video/v2_slideshow_72s.mp4` | 12 photos x 6 s, 1280x720 (full-range) |
| `video/v3_kenburns_portrait_24s.mp4` | pan/zoom over a photo, 720x1280, 29.97 fps, music track |
| `video/v4_slides_58s.webm` | 29 photos x 2 s, 640x480 VP9 |
| `video/v5_short_0.4s.mp4` | 0.4 s clip (exercises the round=up retry) |
| `video/v6_life_10s.mkv` | Game of Life 960x960 15 fps |

`requests.jsonl` (63 requests: 20 text, 23 image, 10 audio, 10 video, with time bounds) is the
fidelity set; `_photo` on a text request is the index of the photo it describes (retrieval check).
`bench_*.jsonl` and `vram_video.jsonl` are throughput / memory sets built from it.
