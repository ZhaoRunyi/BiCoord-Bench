#!/usr/bin/env bash
set -euo pipefail
policy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$policy_dir/../.." && pwd)
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 train_config_name model_name gpu_ids [train.py options...]" >&2
    exit 2
fi
train_config_name=$1
model_name=$2
export CUDA_VISIBLE_DEVICES=$3
shift 3
cd "$policy_dir"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
source /home/MLTL-INTERN/shaoyang/zvi/.uv/env.sh
exec uv run --project "$policy_dir" --no-sync python scripts/train.py \
    "$train_config_name" --exp-name="$model_name" --checkpoint-base-dir "$(dirname "$repo_root")/ckpts/BiCoord_pi0" "$@"
