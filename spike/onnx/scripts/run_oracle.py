"""Drives the real installed Python worker exactly as Seekgram does (same interpreter, script,
environment and --ffmpeg), one request at a time, and records vectors and per-request latency.

Usage: <any python 3> scripts/run_oracle.py <run-name> <requests.jsonl> [repeat]
Output: runs/<run>.out.jsonl (worker responses + "ms"), runs/<run>.events.jsonl (progress),
        runs/<run>.summary.json. Graceful stop: create runs/<run>.stop (checked between requests;
        the worker is then closed via stdin EOF). Read-only with respect to D:/Seekgram.
"""
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
run, requests_path = sys.argv[1], sys.argv[2]
repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 1
RUNS = os.path.join(ROOT, "runs")
stop_file = os.path.join(RUNS, run + ".stop")
events = open(os.path.join(RUNS, run + ".events.jsonl"), "a", encoding="utf-8")


def event(**fields):
    fields["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    events.write(json.dumps(fields) + "\n")
    events.flush()
    print(json.dumps(fields), flush=True)


env = dict(os.environ)
env.update({"HF_HOME": r"D:\Seekgram\models", "HF_HUB_CACHE": r"D:\Seekgram\models\hub", "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1"})
with open(requests_path, encoding="utf-8") as handle:
    requests = [json.loads(line) for line in handle if line.strip()]
for request in requests:
    request.pop("_photo", None)
launched = time.perf_counter()
worker = subprocess.Popen(
    [r"D:\Seekgram\worker\.venv\Scripts\python.exe", r"D:\Seekgram\worker\worker.py", "--ffmpeg", r"D:\Seekgram\tools\ffmpeg.exe"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True, encoding="utf-8", bufsize=1)
event(event="started", requests=len(requests), repeat=repeat)
out = open(os.path.join(RUNS, run + ".out.jsonl"), "w", encoding="utf-8")
latencies = {}
first_response = None
state = "completed"
total = len(requests) * repeat
done = 0
for round_index in range(repeat):
    for request in requests:
        if os.path.exists(stop_file):
            state = "stopped"
            break
        started = time.perf_counter()
        worker.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
        elapsed = (time.perf_counter() - started) * 1e3
        if not line:
            state = "failed"
            event(event="worker_exited", code=worker.poll())
            break
        if first_response is None:
            first_response = (time.perf_counter() - launched) * 1e3
            event(event="first_response", since_launch_ms=first_response)
        response = json.loads(line)
        response["ms"] = elapsed
        response["round"] = round_index
        if round_index == 0:
            out.write(json.dumps(response) + "\n")
            out.flush()
        latencies.setdefault(request["kind"], []).append((request["id"], round_index, elapsed))
        done += 1
        if done % 10 == 0 or done == total:
            event(event="progress", done=done, total=total, last=request["id"], last_ms=round(elapsed, 1))
    if state != "completed":
        break
# Dedicated GPU memory of the worker process (Windows performance counter; nvidia-smi has no
# per-process numbers under WDDM).
gpu = subprocess.run(
    ["powershell", "-NoProfile", "-NonInteractive", "-Command",
     f"((Get-Counter \"\\GPU Process Memory(pid_{worker.pid}_*)\\Dedicated Usage\").CounterSamples|Measure-Object CookedValue -Sum).Sum"],
    capture_output=True, text=True).stdout.strip()
worker.stdin.close()
code = worker.wait(timeout=60)
summary = {"state": state, "worker_exit": code, "first_response_ms": first_response, "gpu_dedicated_bytes_at_end": gpu, "per_kind": {}}
for kind, rows in latencies.items():
    steady = [ms for _, round_index, ms in rows if round_index > 0] or [ms for _, _, ms in rows[1:]]
    summary["per_kind"][kind] = {"n": len(rows), "steady_n": len(steady), "steady_mean_ms": sum(steady) / max(len(steady), 1),
                                  "items_per_second": 1000 * len(steady) / max(sum(steady), 1e-9)}
with open(os.path.join(RUNS, run + ".summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=1)
event(event=state, **{k: v for k, v in summary.items() if k != "per_kind"})
sys.exit(0 if state == "completed" else 1)
