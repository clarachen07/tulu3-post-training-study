#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session=ssp-tulu-runner
source "${root}/scripts/env.sh"

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}"
  exit 90
fi
if [[ -f "${root}/status/pipeline.failed" ]] || [[ -f "${root}/status/pipeline.complete" ]]; then
  echo "Existing terminal pipeline marker found; refusing implicit restart"
  exit 91
fi
tmux new-session -d -s "${session}" \
  "cd '${root}' && bash scripts/run_pipeline.sh"
sleep 1
tmux has-session -t "${session}"
tmux capture-pane -pt "${session}" -S -40
