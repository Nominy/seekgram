#!/usr/bin/env bash
# Resumable download + unpack of the ONNX Runtime builds and CUDA 12 / cuDNN 9 redistributable wheels
# listed in scripts/runtime_manifest.tsv (name<TAB>expected size or 0<TAB>url) into runtimes/.
# Completed archives (size matches) are skipped; partial ones resume (curl -C -).
# Graceful stop: create runs/<run-name>.stop (checked between files).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; RUN="${1:?run name}"
DL="$ROOT/runtimes/archives"; mkdir -p "$DL" "$ROOT/runs"
LOG="$ROOT/runs/$RUN.log"; STATUS="$ROOT/runs/$RUN.status.json"; STOP="$ROOT/runs/$RUN.stop"
rm -f "$STATUS" "$STOP"; exec > >(tee -a "$LOG") 2>&1
DONE=0; TOTAL=$(grep -c . "$ROOT/scripts/runtime_manifest.tsv")
finish() { printf '{"state":"%s","exit":%s,"done":%s,"total":%s,"ts":"%s"}\n' "$1" "$2" "$DONE" "$TOTAL" "$(date -u +%FT%TZ)" > "$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"; echo "[$(date +%T)] $1 exit=$2"; exit "$2"; }
while IFS=$'\t' read -r name size url; do
  [ -n "$name" ] || continue
  [ -e "$STOP" ] && finish stopped 3
  out="$DL/$name"
  if [ -f "$out" ]; then DONE=$((DONE+1)); echo "[$(date +%T)] skip $name ($DONE/$TOTAL)"; continue; fi
  echo "[$(date +%T)] begin $name ($((DONE+1))/$TOTAL) expected=$size"
  ok=0
  for attempt in 1 2 3 4 5; do
    if curl -L --fail -C - --retry 5 --retry-delay 3 -sS -o "$out.part" "$url"; then
      got=$(stat -c %s "$out.part"); if [ "$size" = 0 ] || [ "$got" = "$size" ]; then ok=1; break; fi
      echo "size mismatch got=$got want=$size"; rm -f "$out.part"
    fi
    sleep 3
  done
  [ $ok = 1 ] || finish failed 5
  mv "$out.part" "$out"; DONE=$((DONE+1)); echo "[$(date +%T)] done $name $(stat -c %s "$out") bytes ($DONE/$TOTAL)"
done < "$ROOT/scripts/runtime_manifest.tsv"
# Unpack (all are zip containers). Idempotent: a .ok marker per archive.
for f in "$DL"/*; do
  case "$f" in *.part|*.ok) continue;; esac
  d="$ROOT/runtimes/unpacked/$(basename "$f")"
  [ -e "$d.ok" ] && continue
  rm -rf "$d"; mkdir -p "$d"; echo "[$(date +%T)] unpack $(basename "$f")"
  unzip -q -o "$f" -d "$d" || finish failed 6
  touch "$d.ok"
done
finish completed 0
