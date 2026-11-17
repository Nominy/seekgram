"""Writes the throughput request sets media/bench_{image,audio30,video30}.jsonl from media/requests.jsonl.
Usage: <any python> scripts/make_bench.py"""
import json
import os
import sys

sys.dont_write_bytecode = True
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")
with open(os.path.join(root, "requests.jsonl"), encoding="utf-8") as handle:
    requests = {r["id"]: r for r in (json.loads(line) for line in handle if line.strip())}
sets = {
    # all 23 images, 10 rounds: 230 requests
    "image": ([k for k in requests if k.startswith("i_")], 10),
    # 30 s chunks: sweep, pink noise, music 0-30, music 15-45; 5 rounds: 20 requests
    "audio30": (["a00", "a01", "a02", "a03"], 5),
    # 30 s chunks: testsrc2 (30 frames, has an audio track), mandelbrot, slideshow 0-30, 30-60; 3 rounds: 12 requests
    "video30": (["v00", "v01", "v02", "v03"], 3),
}
for name, (ids, rounds) in sets.items():
    with open(os.path.join(root, f"bench_{name}.jsonl"), "w", encoding="utf-8", newline="\n") as handle:
        for r in range(rounds):
            for rid in ids:
                request = {k: v for k, v in requests[rid].items() if k != "_photo"}
                request["id"] = f"{rid}_r{r}"
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(name, len(ids) * rounds)
