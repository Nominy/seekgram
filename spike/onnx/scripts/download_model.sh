#!/usr/bin/env bash
# Resumable download of onnx-community/embeddinggemma-2-ONNX at a pinned revision.
# Usage: download_model.sh <run-name> [variant ...]   variants: fp16 fp32 q4 q4f16 quantized
# Completed files (sha256 verified against the LFS oid, or size for small files) are skipped.
# Partial files resume with HTTP range requests. Graceful stop: create runs/<run-name>.stop
# (checked between files; the current file keeps its .part and resumes next time).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${1:?run name}"; shift
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(fp16)
REPO="onnx-community/embeddinggemma-2-ONNX"
REV="daa72c51243991dfcaf9f9137d2c573d8f7790c0"
DEST="$ROOT/models/embeddinggemma-2-ONNX"
LOG="$ROOT/runs/$RUN.log"; EVENTS="$ROOT/runs/$RUN.events.jsonl"; STATUS="$ROOT/runs/$RUN.status.json"; STOP="$ROOT/runs/$RUN.stop"
mkdir -p "$DEST/onnx" "$ROOT/runs"
rm -f "$STATUS" "$STOP"
exec > >(tee -a "$LOG") 2>&1
ev() { printf '{"ts":"%s","event":"%s","file":"%s","detail":"%s"}\n' "$(date -u +%FT%TZ)" "$1" "${2:-}" "${3:-}" >> "$EVENTS"; echo "[$(date +%T)] $1 ${2:-} ${3:-}"; }
finish() { printf '{"state":"%s","exit":%s,"done":%s,"total":%s,"ts":"%s"}\n' "$1" "$2" "$DONE" "$TOTAL" "$(date -u +%FT%TZ)" > "$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"; ev "$1" "" "exit=$2"; exit "$2"; }

MANIFEST="$ROOT/scripts/model_manifest.tsv"   # path<TAB>size<TAB>sha256 (or -), from the HF tree API at $REV
[ -s "$MANIFEST" ] || { DONE=0; TOTAL=0; finish failed 2; }

suffix() { case "$1" in fp32) echo "";; *) echo "_$1";; esac; }
FILES=(tokenizer.json tokenizer_config.json config.json preprocessor_config.json processor_config.json chat_template.jinja config_sentence_transformers.json README.md)
for v in "${VARIANTS[@]}"; do s="$(suffix "$v")"; for g in model vision_encoder audio_encoder; do FILES+=("onnx/$g$s.onnx" "onnx/$g$s.onnx_data"); done; done
TOTAL=${#FILES[@]}; DONE=0
ev started "" "variants=${VARIANTS[*]} files=$TOTAL"
for f in "${FILES[@]}"; do
  [ -e "$STOP" ] && finish stopped 3
  size=$(awk -F'\t' -v p="$f" '$1==p{print $2}' "$MANIFEST"); oid=$(awk -F'\t' -v p="$f" '$1==p{print $3}' "$MANIFEST"); [ "$oid" = "-" ] && oid=""
  [ -n "$size" ] || { ev missing_in_tree "$f"; finish failed 4; }
  out="$DEST/$f"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then DONE=$((DONE+1)); ev skip "$f" "$DONE/$TOTAL size=$size"; continue; fi
  ev begin "$f" "$((DONE+1))/$TOTAL size=$size"
  ok=0
  for attempt in 1 2 3 4 5 6; do
    curl -L --fail -C - --retry 5 --retry-delay 3 -sS --progress-bar -o "$out.part" "https://huggingface.co/$REPO/resolve/$REV/$f" 2>&1 | tr '\r' '\n' | awk 'NF && (NR%40==0)' 
    got=$(stat -c %s "$out.part" 2>/dev/null || echo 0)
    if [ "$got" = "$size" ]; then ok=1; break; fi
    ev retry "$f" "attempt=$attempt got=$got want=$size"; [ "$got" -gt "$size" ] && rm -f "$out.part"; sleep 3
  done
  [ $ok = 1 ] || finish failed 5
  if [ -n "$oid" ]; then
    sum=$(sha256sum "$out.part" | cut -d' ' -f1)
    [ "$sum" = "$oid" ] || { ev sha_mismatch "$f" "$sum"; rm -f "$out.part"; finish failed 6; }
  fi
  mv "$out.part" "$out"; DONE=$((DONE+1)); ev done "$f" "$DONE/$TOTAL"
done
finish completed 0
