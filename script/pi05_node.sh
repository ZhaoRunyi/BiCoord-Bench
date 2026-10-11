#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runner_args=("$@")
output_dir=""
rank="0"
total_ranks="1"
execution_flags=""
for ((index=0;index<${#runner_args[@]};index++)); do
    if [[ "${runner_args[index]}" == "--output" ]]; then output_dir="${runner_args[index+1]}"; fi
    if [[ "${runner_args[index]}" == "--rank" ]]; then rank="${runner_args[index+1]}"; fi
    if [[ "${runner_args[index]}" == "--ranks" ]]; then total_ranks="${runner_args[index+1]}"; fi
    if [[ "${runner_args[index]}" == "--xla-flags" ]]; then execution_flags="${runner_args[index+1]}"; fi
    if [[ "${runner_args[index]}" == --xla-flags=* ]]; then execution_flags="${runner_args[index]#--xla-flags=}"; fi
done
if [[ -n "$output_dir" ]]; then
    mkdir -p "$output_dir"
    rm -f "$output_dir/rank-$rank-exit.txt"
    trap 'exit_code=$?; printf "%s %s\n" "${PBS_JOBID:-missing}" "$exit_code" > "$output_dir/rank-$rank-exit.txt"' EXIT
    exec > >(tee "$output_dir/rank-$rank-node.log") 2>&1
fi
source "$repo_root/script/uv_env.sh"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export JAX_PLATFORMS=cuda,cpu
combined_flags="${XLA_FLAGS:-}"
if [[ -n "$execution_flags" ]]; then combined_flags="${combined_flags:+$combined_flags }$execution_flags"; fi
if ((total_ranks>1)); then
    # Avoid compiler autotuning exchanging results through distributed KVS.
    # Every rank autotunes locally; model math and compute dtype are unchanged.
    combined_flags="${combined_flags:+$combined_flags }--xla_gpu_shard_autotuning=false"
fi
if [[ -n "${combined_flags//[[:space:]]/}" ]]; then
    export XLA_FLAGS="$combined_flags"
else
    unset XLA_FLAGS
fi
if [[ " $* " == *" --mode benchmark "* ]]; then
    export NCCL_DEBUG=INFO
fi
cd "$repo_root"
# This runner accepts full-host allocations only. Never infer ownership from idle GPUs.
"$repo_root/policy/pi05/.venv/bin/python" "$repo_root/script/pi05_allocation.py"
"$repo_root/policy/pi05/.venv/bin/python" "$repo_root/script/pi05_cleanup_workers.py"
"$repo_root/policy/pi05/.venv/bin/python" policy/pi05/scripts/train_bicoord.py "$@"
