#!/usr/bin/env bash
set -euo pipefail
policy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$policy_dir/../.." && pwd)
policy_name=$(basename "$policy_dir")
if [[ $# -ne 6 ]]; then
    echo "Usage: $0 task_name task_config train_config_name model_name seed gpu_id" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$6"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}
cd "$repo_root"
exec bash script/run_uv.sh "$policy_name" python script/eval_policy.py \
    --config "policy/$policy_name/deploy_policy.yml" --overrides \
    --task_name "$1" --task_config "$2" --train_config_name "$3" \
    --model_name "$4" --ckpt_setting "$4" --seed "$5" --policy_name "$policy_name"
