#!/usr/bin/env bash
# Launch any long-running command in a detached tmux session that survives
# SSH / Antigravity IDE disconnects, with automatic log + exit-code tracking.
#
# Usage:
#   scripts/run_detached.sh <job_name> <log_file> -- <command ...>
#   scripts/run_detached.sh --status [job_name]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
META_DIR="$ROOT/logs/jobs"
mkdir -p "$META_DIR"

if [[ "${1:-}" == "--status" ]]; then
  FILTER="${2:-}"
  printf "%-28s %-10s %-8s %s\n" "JOB" "STATE" "EXIT" "LOG"
  for mf in "$META_DIR"/*.meta; do
    [[ -e "$mf" ]] || continue
    # shellcheck disable=SC1090
    source "$mf"
    if [[ -n "$FILTER" && "$JOB_NAME" != "$FILTER" ]]; then
      continue
    fi
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
      STATE="RUNNING"
      EXIT_CODE="-"
    elif [[ -f "$EXIT_FILE" ]]; then
      EXIT_CODE="$(cat "$EXIT_FILE")"
      if [[ "$EXIT_CODE" == "0" ]]; then STATE="DONE"; else STATE="FAILED"; fi
    else
      STATE="STOPPED"
      EXIT_CODE="?"
    fi
    printf "%-28s %-10s %-8s %s\n" "$JOB_NAME" "$STATE" "$EXIT_CODE" "$LOG_FILE"
  done
  exit 0
fi

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <job_name> <log_file> -- <command ...>" >&2
  exit 2
fi

JOB_NAME="$1"
LOG_FILE="$2"
shift 2
if [[ "$1" != "--" ]]; then
  echo "Expected '--' separator before command" >&2
  exit 2
fi
shift

TMUX_SESSION="agy_${JOB_NAME}"
EXIT_FILE="$META_DIR/${JOB_NAME}.exit"
META_FILE="$META_DIR/${JOB_NAME}.meta"
RUNNER_SCRIPT="$META_DIR/${JOB_NAME}.run.sh"

mkdir -p "$(dirname "$LOG_FILE")"
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$TMUX_SESSION' is already running." >&2
  exit 1
fi
rm -f "$EXIT_FILE"

{
  echo "#!/usr/bin/env bash"
  echo "trap '' HUP"
  echo "set -o pipefail"
  echo "cd $(printf '%q' "$ROOT")"
  echo "export PYTHONUNBUFFERED=1"
  echo "echo \"[run_detached] START \$(date -Iseconds) job=$JOB_NAME pid=\$\$\" >> $(printf '%q' "$LOG_FILE")"
  printf 'nohup '
  printf '%q ' "$@"
  echo "</dev/null >> $(printf '%q' "$LOG_FILE") 2>&1"
  echo "RC=\$?"
  echo "echo \"\$RC\" > $(printf '%q' "$EXIT_FILE")"
  echo "echo \"[run_detached] END \$(date -Iseconds) job=$JOB_NAME rc=\$RC\" >> $(printf '%q' "$LOG_FILE")"
  echo "exit \"\$RC\""
} > "$RUNNER_SCRIPT"
chmod +x "$RUNNER_SCRIPT"

cat > "$META_FILE" <<EOF
JOB_NAME=$(printf '%q' "$JOB_NAME")
TMUX_SESSION=$(printf '%q' "$TMUX_SESSION")
LOG_FILE=$(printf '%q' "$LOG_FILE")
EXIT_FILE=$(printf '%q' "$EXIT_FILE")
STARTED_AT=$(printf '%q' "$(date -Iseconds)")
EOF

nohup setsid tmux new-session -d -s "$TMUX_SESSION" "bash $(printf '%q' "$RUNNER_SCRIPT")" </dev/null >/dev/null 2>&1
sleep 0.3

echo "[OK] Launched detached job '$JOB_NAME' in tmux session '$TMUX_SESSION'"
echo "     Log   : $LOG_FILE"
echo "     Status: scripts/run_detached.sh --status $JOB_NAME"
echo "     Attach: tmux attach -t $TMUX_SESSION"
