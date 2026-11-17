#!/usr/bin/env bash
# embed.sh <run-name> <provider cuda|dml|cpu> <variant> <requests.jsonl> [extra onnx-spike flags...]
# Runs the Rust embedder over a request file. Vectors: runs/<run>.out.jsonl; diagnostics: runs/<run>.err.jsonl.
# Use under run_logged.sh in a psmux session. The runtime directory is chosen by provider.
set -u
# Simulate a PC without the CUDA toolkit: drop every CUDA/NVIDIA directory from PATH.
PATH="$(printf %s "$PATH" | tr : "\n" | grep -v -i -E "cuda|nvidia" | paste -sd: -)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$1"; PROVIDER="$2"; VARIANT="$3"; REQUESTS="$4"; shift 4
MODEL="$ROOT/models/embeddinggemma-2-ONNX"
case "$PROVIDER" in
  cuda) RT="$ROOT/runtimes/cuda12";;
  cuda13) RT="$ROOT/runtimes/cuda13"; PROVIDER=cuda;;
  # DirectML rejects Reshape allowzero=1: use the graphs patched by scripts/patch_dml.py.
  dml) RT="$ROOT/runtimes/dml"; MODEL="$ROOT/models/embeddinggemma-2-ONNX-dml";;
  cpu) RT="$ROOT/runtimes/cuda12";;
esac
"$ROOT/target/release/onnx-spike.exe" --ort "$RT/onnxruntime.dll" --model-dir "$MODEL" \
  --ffmpeg D:/Seekgram/tools/ffmpeg.exe --provider "$PROVIDER" --variant "$VARIANT" "$@" \
  < "$REQUESTS" > "$ROOT/runs/$RUN.out.jsonl" 2> >(tee "$ROOT/runs/$RUN.err.jsonl" >&2)
