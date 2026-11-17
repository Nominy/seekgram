"""Speed / memory summary of finished runs as a markdown table.

Usage: <python> scripts/summarize.py <run-name>...
- Rust single runs (`--timing`): items/s = n / sum(ms.total) over all responses after the first 3
  (warm-up excluded), plus the mean of each phase.
- Rust bulk runs: the `bulk` event (wall clock over the whole set, warm-up included).
- Python worker runs (run_oracle.py): steady items/s from the summary (first request excluded),
  measured on the client side of the pipe, so it includes JSON and process I/O.
- GPU: peak whole-GPU memory above the pre-run baseline (runs/<run>.gpu.json, nvidia-smi samples).
- first: time from process launch to the first response.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


print("| run | items | ok | items/s | mean ms/item (preprocess / encoder / text) | first response s | GPU peak above baseline MiB | errors |")
print("|--|--|--|--|--|--|--|--|")
for run in sys.argv[1:]:
    out = read_jsonl(os.path.join(RUNS, run + ".out.jsonl"))
    err = read_jsonl(os.path.join(RUNS, run + ".err.jsonl"))
    gpu = read_jsonl(os.path.join(RUNS, run + ".gpu.json"))
    gpu_peak = gpu[-1]["peak_minus_baseline_mib"] if gpu else ""
    ok = sum(1 for row in out if "vector" in row)
    errors = sorted({(row.get("error") or {}).get("message", "")[:80] for row in out if "error" in row})
    first = next((row["since_launch_ms"] / 1000 for row in err if row.get("event") == "first_response"), None)
    phases = ""
    rate = None
    bulk = next((row for row in err if row.get("event") == "bulk"), None)
    summary_path = os.path.join(RUNS, run + ".summary.json")
    if bulk:
        # Steady state (after the first batch) when available; whole run otherwise.
        rate = bulk.get("steady_items_per_second") or bulk["items_per_second"]
        phases = f"{bulk['preprocess_cpu_ms_per_item']:.0f} (CPU, parallel) / {bulk['encoder_ms_per_item']:.0f} / {bulk['text_ms_per_item']:.0f}"
    elif os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        first = summary["first_response_ms"] / 1000 if summary.get("first_response_ms") else None
        kinds = summary["per_kind"]
        if len(kinds) == 1:
            kind = next(iter(kinds.values()))
            rate = kind["items_per_second"]
            phases = f"{kind['steady_mean_ms']:.0f} total"
    else:
        timed = [row for row in out if "ms" in row and "vector" in row][3:]
        if timed:
            total = sum(row["ms"]["total"] for row in timed)
            rate = 1000 * len(timed) / total
            mean = lambda key: sum(row["ms"][key] for row in timed) / len(timed)
            phases = f"{mean('preprocess'):.0f} / {mean('encoder'):.0f} / {mean('text'):.0f}"
    rate_text = f"{rate:.2f}" if rate is not None else ""
    first_text = f"{first:.1f}" if first is not None else ""
    print(f"| {run} | {len(out)} | {ok} | {rate_text} | {phases} | {first_text} | {gpu_peak} | {'; '.join(errors)} |")
