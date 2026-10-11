#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo_root/script/run_uv.sh" sim python "$repo_root/script/setup_assets.py"
