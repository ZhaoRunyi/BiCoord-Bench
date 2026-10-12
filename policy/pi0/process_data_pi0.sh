#!/usr/bin/env bash
set -euo pipefail
policy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$policy_dir/../.." && pwd)
cd "$policy_dir"
source /home/MLTL-INTERN/shaoyang/zvi/.uv/env.sh
exec uv run --project "$policy_dir" --no-sync python scripts/process_data.py "${1:?task_name}" "${2:?task_config}" "${3:?expert_data_num}" --data-root "$(dirname "$repo_root")/data/BiCoord" "${@:4}"
