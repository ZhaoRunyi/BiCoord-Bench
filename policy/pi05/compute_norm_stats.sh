#!/usr/bin/env bash
set -euo pipefail
policy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$policy_dir/../.." && pwd)
cd "$policy_dir"
export JAX_PLATFORMS=cpu
source /home/MLTL-INTERN/shaoyang/zvi/.uv/env.sh
exec uv run --project "$policy_dir" --no-sync python scripts/compute_norm_stats.py --config-name "${1:?train_config_name}"
