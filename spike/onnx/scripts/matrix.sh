#!/usr/bin/env bash
# matrix.sh <matrix-name> <plan-file>
# Runs every line of a plan sequentially. Plan lines (tab-separated, # comments allowed):
#   rust   <run-name> <provider> <variant> <requests.jsonl> [onnx-spike flags...]
#   python <run-name> <requests.jsonl> [repeat]
# Resumable: a run whose runs/<run>.done exists is skipped. Graceful stop: create
# runs/<matrix-name>.stop (checked between runs; the current run finishes, bounded by its size).
# Progress: runs/<matrix-name>.events.jsonl and runs/<matrix-name>.status.json (atomic).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
NAME="$1"; PLAN="$2"
EVENTS="runs/$NAME.events.jsonl"; STATUS="runs/$NAME.status.json"; STOP="runs/$NAME.stop"
rm -f "$STATUS" "$STOP"
TOTAL=$(grep -v -E '^\s*(#|$)' "$PLAN" | wc -l); INDEX=0; FAILED=0
status() { printf '{"state":"%s","index":%s,"total":%s,"failed":%s,"current":"%s","ts":"%s"}\n' "$1" "$INDEX" "$TOTAL" "$FAILED" "${2:-}" "$(date -u +%FT%TZ)" > "$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"; }
event() { printf '{"ts":"%s","event":"%s","run":"%s","detail":"%s"}\n' "$(date -u +%FT%TZ)" "$1" "${2:-}" "${3:-}" >> "$EVENTS"; echo "[$(date +%T)] $1 ${2:-} ${3:-}"; }
event started "" "plan=$PLAN runs=$TOTAL"
while IFS=$'\t' read -r -a fields; do
  [ ${#fields[@]} -eq 0 ] && continue
  case "${fields[0]}" in \#*|"") continue;; esac
  INDEX=$((INDEX+1)); run="${fields[1]}"
  if [ -e "$STOP" ]; then status stopped; event stopped "$run"; exit 3; fi
  if [ -e "runs/$run.done" ]; then event skip "$run" "$INDEX/$TOTAL"; continue; fi
  status running "$run"; event begin "$run" "$INDEX/$TOTAL"
  started=$(date +%s)
  # Whole-GPU memory sampler (nvidia-smi, every 0.5 s): baseline before the run and peak during it.
  # Under WDDM there are no per-process numbers in nvidia-smi; the GPU is otherwise idle here.
  base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  ( peak=0; while :; do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$v" -gt "$peak" ] && peak=$v && printf '{"baseline_mib":%s,"peak_mib":%s,"peak_minus_baseline_mib":%s}\n' "$base" "$peak" $((peak-base)) > "runs/$run.gpu.json"; sleep 0.5; done ) &
  sampler=$!
  if [ "${fields[0]}" = rust ]; then
    bash scripts/embed.sh "$run" "${fields[2]}" "${fields[3]}" "${fields[4]}" "${fields[@]:5}"
  else
    D:/Seekgram/worker/.venv/Scripts/python.exe -B scripts/run_oracle.py "$run" "${fields[2]}" "${fields[3]:-1}"
  fi
  code=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  if [ $code = 0 ]; then touch "runs/$run.done"; event done "$run" "$INDEX/$TOTAL $(( $(date +%s) - started ))s"
  else FAILED=$((FAILED+1)); event failed "$run" "exit=$code"; fi
done < "$PLAN"
status completed; event completed "" "failed=$FAILED"
