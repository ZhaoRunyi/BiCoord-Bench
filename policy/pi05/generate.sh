#!/usr/bin/env bash
set -euo pipefail
policy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$policy_dir/../.." && pwd)
cd "$policy_dir"
exec bash "$repo_root/script/run_uv.sh" "$(basename "$policy_dir")" python examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir "${1:?data_dir}" --repo_id "${2:?repo_id}"
