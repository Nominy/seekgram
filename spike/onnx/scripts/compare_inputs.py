"""Compares the model inputs the Rust preprocessing produced (onnx-spike --dump-dir) with the
tensors the Python processor handed to the model (scripts/trace_oracle.py).
Usage: <worker python> scripts/compare_inputs.py runs/trace_gpu runs/dump_rust"""
import glob
import os
import sys

sys.dont_write_bytecode = True
import numpy as np

trace_dir, dump_dir = sys.argv[1:3]
print("| id | input_ids | pixels / features: max abs diff, mean abs diff, share of exactly equal values (pixels: share within 1/255) | positions / mask |")
print("|--|--|--|--|")
for path in sorted(glob.glob(os.path.join(trace_dir, "*.npz"))):
    rid = os.path.basename(path)[:-4]
    z = np.load(path)
    ids_path = os.path.join(dump_dir, rid + ".input_ids.i64")
    if not os.path.exists(ids_path):
        continue
    ids = np.fromfile(ids_path, dtype="<i8")
    ref_ids = z["in.input_ids"][0]
    cells = ["equal" if np.array_equal(ids, ref_ids) else f"DIFF len {len(ids)} vs {len(ref_ids)}, first mismatch {int(np.argmax(ids[:min(len(ids),len(ref_ids))] != ref_ids[:min(len(ids),len(ref_ids))]))}"]
    other = ""
    if "in.pixel_values" in z.files or "in.pixel_values_videos" in z.files:
        key = "in.pixel_values" if "in.pixel_values" in z.files else "in.pixel_values_videos"
        ref = z[key].reshape(-1)
        mine = np.fromfile(os.path.join(dump_dir, rid + ".pixel_values.f32"), dtype="<f4")
        if mine.shape != ref.shape:
            cells.append(f"shape {mine.shape} vs {ref.shape}")
        else:
            d = np.abs(mine.astype(np.float64) - ref)
            cells.append(f"{d.max():.4f}, {d.mean():.2e}, {(d == 0).mean():.4f} ({(d <= 1/255 + 1e-6).mean():.4f})")
        pkey = "in.image_position_ids" if key == "in.pixel_values" else "in.video_position_ids"
        pos = np.fromfile(os.path.join(dump_dir, rid + ".positions.i64"), dtype="<i8")
        other = "equal" if np.array_equal(pos, z[pkey].reshape(-1)) else "DIFF"
    elif "in.input_features" in z.files:
        ref = z["in.input_features"][0]
        mine = np.fromfile(os.path.join(dump_dir, rid + ".input_features.f32"), dtype="<f4").reshape(-1, 128)
        mask = np.fromfile(os.path.join(dump_dir, rid + ".input_features_mask.u8"), dtype=np.uint8).astype(bool)
        ref_mask = z["in.input_features_mask"][0]
        if mine.shape != ref.shape:
            cells.append(f"shape {mine.shape} vs {ref.shape}")
        else:
            d = np.abs(mine.astype(np.float64) - ref)
            cells.append(f"{d.max():.2e}, {d.mean():.2e}, {(d == 0).mean():.4f}")
        other = "equal" if mask.shape == ref_mask.shape and np.array_equal(mask, ref_mask) else f"DIFF {mask.sum()} vs {ref_mask.sum()}"
    else:
        cells.append("")
    print(f"| {rid} | " + " | ".join(cells) + f" | {other} |")
