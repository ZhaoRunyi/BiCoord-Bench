#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${3:?gpu_id}"
exec bash script/run_uv.sh sim python script/collect_data.py "${1:?task_name}" "${2:?task_config}"
