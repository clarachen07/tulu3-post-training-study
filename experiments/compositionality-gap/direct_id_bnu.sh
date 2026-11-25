#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
local_run="$repo/runs/compositionality-gap/direct-id"
source_dir=experiments/compositionality-gap
prontoqa=experiments/prontoqa
action=${1:-}

bnu() {
    local rc=0
    ssh bnutg-direct "$@" || rc=$?
    if [[ $rc == 255 ]]; then
        /usr/bin/python3 -B /Users/clarachen/.ssh/bnu-connection-guard/guard.py --status >&2
    fi
    return "$rc"
}

case "$action" in
    launch|resume|fetch) ;;
    *) echo "usage: bash direct_id_bnu.sh {launch|resume|fetch}" >&2; exit 2 ;;
esac

receive="$local_run/retrieved"
[[ -f "$local_run/inputs.json.gz" && ! -e "$receive" ]]
destination=$(mktemp -d "$local_run/retrieval.XXXXXXXX")
remote=$(cat <<'REMOTE'
set -eu
umask 077
project=/home/ai/chenchen/projects/compositionality-gap-direct-id-fp32
action=$1
if test "$action" = launch; then
    mkdir "$project"
    cd "$project"
    tar -xzf -
    mkdir run
    mv inputs.json.gz run/
else
    cd "$project"
fi

if test "$action" = resume; then
    test -f job.exit && test "$(cat job.exit)" != 0
    mv -n job.exit "job.exit.$(date +%s)"
fi
if test "$action" != fetch; then
    nohup bash experiments/compositionality-gap/direct_id_bnu_job.sh < /dev/null >> job.log 2>&1 &
    pid=$!
    echo "$pid" > job.pid
    tail -n 40 --sleep-interval=120 --pid="$pid" -F job.log >&2
    wait "$pid"
fi
if test "$(cat job.exit)" != 0; then
    tail -n 35 job.log >&2
    exit 1
fi
tar -czf - -C run reports capture.json models.json tokenization.json     pilot/{base,sft,dpo,rlvr}/complete.json activations/{base,sft,dpo,rlvr}/complete.json
REMOTE
)
if [[ $action == launch ]]; then
    COPYFILE_DISABLE=1 tar --no-xattrs -czf - -C "$repo"         "$source_dir/direct_id.py" "$source_dir/direct_id_report.py" "$source_dir/direct_id_bnu_job.sh"         "$source_dir/protocol.py" "$source_dir/runner.py"         "$prontoqa/experiment.json" "$prontoqa/provenance/models.json"         "$prontoqa/tulu_id" "$prontoqa/tulu_behavior/__init__.py"         "$prontoqa/tulu_behavior/assets.py" "$prontoqa/tulu_behavior/common.py"         -C "$local_run" inputs.json.gz | bnu "bash -c '$remote' direct-id $action" > "$destination/packet.tar.gz"
else
    bnu "bash -c '$remote' direct-id $action" </dev/null > "$destination/packet.tar.gz"
fi
tar -xzf "$destination/packet.tar.gz" -C "$destination"
mv -n "$destination" "$receive"
printf '%s\n' "$receive"
