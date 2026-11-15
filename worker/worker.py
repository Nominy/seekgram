"""Local EmbeddingGemma 2 JSONL worker. Media bounds are relative to the input file.

Model files must already be cached unless --allow-download is explicitly supplied.
Use --ffmpeg or OMNI_FFMPEG to select the ffmpeg executable.
"""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

MODEL = "google/embeddinggemma-2"
REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
IDENTITY = (MODEL + "@" + REVISION + "|omni-v1|query=task: search result | query: "
            "|media=no-prefix|audio=16khz-mono|video=1fps|dim=768")


class RequestError(Exception):
    pass


def run_ffmpeg(args):
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError as exc:
        raise RequestError("ffmpeg could not be started") from exc
    if result.returncode:
        raise RequestError("ffmpeg failed to decode media")
    return result.stdout


def bounds(request):
    start = request.get("start_seconds", 0)
    end = request.get("end_seconds")
    if start is None:
        start = 0
    if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0:
        raise RequestError("invalid start_seconds")
    if end is not None and (isinstance(end, bool) or not isinstance(end, (int, float)) or not math.isfinite(end) or end <= start):
        raise RequestError("invalid end_seconds")
    return float(start), float(end) if end is not None else None


def ffmpeg_input(ffmpeg, path, start, end):
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-protocol_whitelist", "file,pipe", "-ss", str(start), "-i", str(path)]
    if end is not None:
        args += ["-t", str(end - start)]
    return args


class Worker:
    def __init__(self, ffmpeg, allow_download):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.ffmpeg = ffmpeg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = SentenceTransformer(MODEL, revision=REVISION, device=device,
                                         local_files_only=not allow_download,
                                         model_kwargs={"dtype": dtype}, trust_remote_code=False)
        self.model.eval()

    def encode(self, kind, value):
        import numpy as np
        kwargs = {"normalize_embeddings": True, "show_progress_bar": False,
                  "convert_to_numpy": True}
        if kind == "text":
            output = self.model.encode({"text": value}, prompt_name="SearchQuery", **kwargs)
        else:
            output = self.model.encode({kind: value}, **kwargs)
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        if vector.size != 768 or not np.isfinite(vector).all():
            raise RequestError("model returned an invalid embedding")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise RequestError("model returned a zero or invalid embedding")
        return (vector / norm).tolist()

    def request(self, request):
        import numpy as np
        from PIL import Image, ImageOps
        if not isinstance(request, dict):
            raise RequestError("request must be an object")
        kind = request.get("kind")
        value = request.get("input")
        if kind not in ("text", "image", "audio", "video"):
            raise RequestError("invalid kind")
        if not isinstance(value, str) or not value:
            raise RequestError("input must be a nonempty string")
        if kind == "text":
            if "start_seconds" in request or "end_seconds" in request:
                raise RequestError("bounds apply only to audio and video")
            return self.encode(kind, value)
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise RequestError("media input must be an existing absolute local file")
        if kind == "image":
            if "start_seconds" in request or "end_seconds" in request:
                raise RequestError("bounds apply only to audio and video")
            with Image.open(path) as image:
                return self.encode(kind, ImageOps.exif_transpose(image).convert("RGB"))
        start, end = bounds(request)
        args = ffmpeg_input(self.ffmpeg, path, start, end)
        if kind == "audio":
            # The audio processor caps each audio item at 750 tokens (30 s).
            if end is not None and end - start > 30:
                raise RequestError("audio segment exceeds 30 seconds")
            duration = min(end - start, 30.001) if end is not None else 30.001
            audio_args = ffmpeg_input(self.ffmpeg, path, start, None)
            data = run_ffmpeg(audio_args + ["-t", str(duration), "-vn", "-ac", "1", "-ar", "16000",
                                            "-f", "f32le", "pipe:1"])
            if not data:
                raise RequestError("audio segment contains no samples")
            if len(data) > 30 * 16000 * 4:
                raise RequestError("audio segment exceeds 30 seconds")
            return self.encode(kind, {"array": np.frombuffer(data, dtype="<f4"), "sampling_rate": 16000})
        # 59 frames would exceed the model's 8192-token window at 140 tokens/frame.
        with tempfile.TemporaryDirectory(prefix="omni-video-") as tmp:
            clip = str(Path(tmp) / "clip.mkv")
            progress = run_ffmpeg(args + ["-an", "-vf", "fps=1", "-frames:v", "59",
                                          "-c:v", "ffv1", "-y", "-progress", "pipe:1", clip])
            frames = [int(line.split(b"=", 1)[1]) for line in progress.splitlines() if line.startswith(b"frame=")]
            # The default fps filter drops clips shorter than half a second.
            # Only retry empty outputs, leaving existing successful vectors unchanged.
            if not frames or frames[-1] == 0:
                progress = run_ffmpeg(args + ["-an", "-vf", "fps=1:round=up", "-frames:v", "59",
                                              "-c:v", "ffv1", "-y", "-progress", "pipe:1", clip])
                frames = [int(line.split(b"=", 1)[1]) for line in progress.splitlines() if line.startswith(b"frame=")]
            if not frames or frames[-1] == 0:
                raise RequestError("video segment contains no sampled frames")
            if frames[-1] > 58:
                raise RequestError("video segment exceeds 58 sampled frames")
            return self.encode(kind, clip)


def self_test(worker):
    from PIL import Image
    import numpy as np
    print("stage=text", file=sys.__stderr__, flush=True)
    worker.encode("text", "A small test image")
    print("stage=image", file=sys.__stderr__, flush=True)
    worker.encode("image", Image.new("RGB", (64, 64), (40, 120, 200)))
    print("stage=audio", file=sys.__stderr__, flush=True)
    worker.encode("audio", {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000})
    print("stage=video", file=sys.__stderr__, flush=True)
    with tempfile.TemporaryDirectory(prefix="omni-self-test-") as tmp:
        path = str(Path(tmp) / "video.mkv")
        run_ffmpeg([worker.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
                    "-i", "color=c=blue:s=64x64:r=1:d=2", "-c:v", "ffv1", "-y", path])
        worker.request({"kind": "video", "input": path, "start_seconds": 0, "end_seconds": 1})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default=os.environ.get("OMNI_FFMPEG", "ffmpeg"))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    # Third-party diagnostics may contain input content or local paths. Never forward them.
    if args.self_test:
        print("stage=loading_model", file=sys.__stderr__, flush=True)
    with open(os.devnull, "w", encoding="utf-8") as null:
        try:
            with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                worker = Worker(args.ffmpeg, args.allow_download)
                if args.self_test:
                    # Without a usable driver torch quietly runs on the CPU; setup must notice.
                    print("device=" + worker.device, file=sys.__stderr__, flush=True)
                    self_test(worker)
        except Exception as exc:
            print("worker initialization/self-test failed: " + type(exc).__name__, file=sys.stderr, flush=True)
            return 1
        if args.self_test:
            print("self-test passed", file=sys.stderr, flush=True)
            return 0
        for line in sys.stdin:
            request_id = None
            try:
                request = json.loads(line)
                if isinstance(request, dict):
                    request_id = request.get("id")
                with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null), worker.torch.inference_mode():
                    vector = worker.request(request)
                response = {"id": request_id, "vector": vector}
            except Exception as exc:
                message = str(exc) if isinstance(exc, RequestError) else "request processing failed"
                frames = traceback.extract_tb(exc.__traceback__)
                location = f"{Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else None
                response = {"id": request_id, "error": {"type": type(exc).__name__, "message": message, "location": location}}
            print(json.dumps(response, ensure_ascii=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
