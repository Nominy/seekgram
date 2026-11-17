"""Runs the unmodified Seekgram worker class in-process and records, per request, the exact tensors
the processor handed to the model, the encoder outputs and the final vector.

Debugging aid for the Rust port (the reference *vectors* and timings come from the real worker
process, see run_oracle.py). Run with the worker's own interpreter:

  D:/Seekgram/worker/.venv/Scripts/python.exe scripts/trace_oracle.py <worker_dir> <requests.jsonl> <out_dir> [--cpu-fp32]

Nothing is written outside <out_dir>.
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HOME", r"D:\Seekgram\models")
os.environ.setdefault("HF_HUB_CACHE", r"D:\Seekgram\models\hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

worker_dir, requests_path, out_dir = sys.argv[1:4]
cpu_fp32 = "--cpu-fp32" in sys.argv
sys.path.insert(0, worker_dir)
import numpy as np
import torch
import worker as worker_module

os.makedirs(out_dir, exist_ok=True)
if cpu_fp32:
    # fp32 reference on the CPU: separates "ONNX graph differs" from "bf16 rounding differs".
    torch.cuda.is_available = lambda: False
w = worker_module.Worker(r"D:\Seekgram\tools\ffmpeg.exe", False)
transformer = w.model[0]
model = transformer.auto_model
captured = {}


def to_np(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.dtype in (torch.bfloat16, torch.float16):
            value = value.float()
        return value.cpu().numpy()
    return np.asarray(value)


original_forward = model.forward


def forward(*args, **kwargs):
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            captured["in." + key] = to_np(value)
    output = original_forward(*args, **kwargs)
    captured["out.last_hidden_state"] = to_np(output.last_hidden_state)
    return output


model.forward = forward

for name in ("get_image_features", "get_video_features", "get_audio_features"):
    def wrap(name=name, original=getattr(model, name)):
        def inner(*args, **kwargs):
            output = original(*args, **kwargs)
            pooled = output.pooler_output
            if isinstance(pooled, (tuple, list)):
                pooled = torch.cat(list(pooled), dim=0)
            captured["feat." + name] = to_np(pooled)
            mask = getattr(output, "attention_mask", None)
            if mask is not None:
                captured["feat." + name + ".mask"] = to_np(mask)
            return output
        return inner
    setattr(model, name, wrap())

# Which video reader is used, and what it returns.
from transformers import video_utils

for backend, reader in list(video_utils.VIDEO_DECODERS.items()) if hasattr(video_utils, "VIDEO_DECODERS") else []:
    def wrap_reader(backend=backend, reader=reader):
        def inner(*args, **kwargs):
            frames, metadata = reader(*args, **kwargs)
            captured["video.backend"] = np.array(backend)
            captured["video.frames_shape"] = np.array(getattr(frames, "shape", ()))
            captured["video.first_frame"] = to_np(frames[0]) if len(frames) else np.zeros(0)
            captured["video.metadata"] = np.array(repr(metadata))
            return frames, metadata
        return inner
    video_utils.VIDEO_DECODERS[backend] = wrap_reader()

summary = {"device": w.device, "dtype": str(model.dtype), "video_decoders_hooked": hasattr(video_utils, "VIDEO_DECODERS")}
with open(requests_path, encoding="utf-8") as handle:
    requests = [json.loads(line) for line in handle if line.strip()]
results = []
for request in requests:
    captured.clear()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            vector = w.request(request)
        captured["vector"] = np.asarray(vector, dtype=np.float32)
        error = None
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    np.savez(os.path.join(out_dir, f"{request['id']}.npz"), **captured)
    shapes = {key: list(value.shape) for key, value in captured.items() if value.ndim}
    results.append({"id": request["id"], "kind": request["kind"], "seconds": round(elapsed, 4), "error": error, "shapes": shapes})
    print(json.dumps(results[-1]), flush=True)
summary["results"] = results
with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=1)
