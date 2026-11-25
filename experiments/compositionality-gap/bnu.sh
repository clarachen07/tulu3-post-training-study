#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
project=/home/ai/chenchen/projects/compositionality-gap-tulu3
local_run="$repo/runs/compositionality-gap/evaluation"

bnu() {
    local result=0
    ssh bnutg-direct "$@" || result=$?
    if [[ $result == 255 ]]; then
        /usr/bin/python3 -B /Users/clarachen/.ssh/bnu-connection-guard/guard.py --status >&2
    fi
    return "$result"
}

case "${1:-}" in
    launch)
        [[ -f $local_run/train.parquet ]]
        submit=$(cat <<'REMOTE'
set -eu
umask 077
project=/home/ai/chenchen/projects/compositionality-gap-tulu3
test ! -e "$project"
incoming=$(mktemp -d "$project.incoming.XXXXXXXX")
trap 'if test -d "$incoming"; then rm -rf -- "$incoming"; fi' EXIT
tar -xzf - -C "$incoming"
mkdir "$incoming/evaluation"
mv "$incoming/train.parquet" "$incoming/evaluation/"
mv -nT -- "$incoming" "$project"
test ! -e "$incoming"
cd "$project"
nohup bash experiments/compositionality-gap/bnu-job.sh < /dev/null > job.log 2>&1 &
echo $! > job.pid
cat job.pid
REMOTE
)
        COPYFILE_DISABLE=1 tar -czf - -C "$repo" \
            experiments/compositionality-gap/experiment.py \
            experiments/compositionality-gap/protocol.py \
            experiments/compositionality-gap/runner.py \
            experiments/compositionality-gap/report.py \
            experiments/compositionality-gap/requirements.txt \
            experiments/compositionality-gap/bnu-job.sh \
            experiments/prontoqa/tulu_behavior/__init__.py \
            experiments/prontoqa/tulu_behavior/assets.py \
            experiments/prontoqa/tulu_behavior/common.py \
            experiments/prontoqa/provenance/models.json \
            -C "$local_run" train.parquet | bnu "$submit"
        ;;
    resume)
        bnu "cd '$project' && test -f job.exit && test \"\$(cat job.exit)\" != 0 && {
            nohup bash experiments/compositionality-gap/bnu-job.sh < /dev/null >> job.log 2>&1 &
            echo \$! > job.pid
            cat job.pid
        }"
        ;;
    status)
        bnu "cd '$project' && tail -n 18 job.log && if test -f job.exit; then cat job.exit; else ps -p \$(cat job.pid) -o pid=,stat=,etime=,args=; fi"
        ;;
    follow|fetch)
        output="$repo/runs/compositionality-gap/bnu-results"
        [[ ! -e $output ]]
        # Create an isolated destination so an interrupted transfer never looks complete.
        destination=$(mktemp -d "$repo/runs/compositionality-gap/retrieval.XXXXXXXX")
        retrieve=$(cat <<'REMOTE'
set -eu
cd /home/ai/chenchen/projects/compositionality-gap-tulu3
if test "$1" = follow && ! test -f job.exit; then
    tail -n 24 --sleep-interval=120 --pid="$(cat job.pid)" -F job.log >&2
fi
if ! test -f job.exit || test "$(cat job.exit)" != 0; then
    tail -n 24 job.log >&2
    exit 1
fi
tar -czf - -C evaluation .
REMOTE
)
        bnu bash -s -- "$1" <<< "$retrieve" | tar -xzf - -C "$destination"
        mv "$destination" "$output"
        printf '%s\n' "$output"
        ;;
    *)
        echo "usage: bash bnu.sh {launch|resume|status|follow|fetch}" >&2
        exit 2
        ;;
esac
