#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${3:?gpu_id}"
source /home/MLTL-INTERN/shaoyang/zvi/.uv/env.sh
exec uv run --project "$repo_root" --no-sync python script/collect_data.py "${1:?task_name}" "${2:?task_config}"
