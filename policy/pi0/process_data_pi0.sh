#!/usr/bin/env bash
set -euo pipefail
policy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$policy_dir/../.." && pwd)
cd "$policy_dir"
exec bash "$repo_root/script/run_uv.sh" "$(basename "$policy_dir")" python scripts/process_data.py "${1:?task_name}" "${2:?task_config}" "${3:?expert_data_num}"
