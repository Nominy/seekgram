#!/usr/bin/env bash
# run_logged.sh <run-name> <command...>: runs a command, tees output to runs/<name>.log and writes
# runs/<name>.status.json ({"state","exit","started","ended"}) when it ends. Meant to be the command
# of a detached psmux session:  psmux new-session -d -s <name> -- bash scripts/run_logged.sh <name> ...
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; NAME="$1"; shift
mkdir -p "$ROOT/runs"; rm -f "$ROOT/runs/$NAME.status.json"
START="$(date -u +%FT%TZ)"
cd "$ROOT"
export PATH="/c/Users/User/.cargo/bin:$PATH"
{ echo "[$START] start: $*"; "$@"; CODE=$?; echo "[$(date -u +%FT%TZ)] exit=$CODE"; exit $CODE; } 2>&1 | tee -a "$ROOT/runs/$NAME.log"
CODE=${PIPESTATUS[0]}
STATE=completed; [ "$CODE" != 0 ] && STATE=failed
printf '{"state":"%s","exit":%s,"started":"%s","ended":"%s"}\n' "$STATE" "$CODE" "$START" "$(date -u +%FT%TZ)" > "$ROOT/runs/$NAME.status.json.tmp"
mv "$ROOT/runs/$NAME.status.json.tmp" "$ROOT/runs/$NAME.status.json"
