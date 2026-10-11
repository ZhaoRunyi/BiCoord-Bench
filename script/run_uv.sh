#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
env_name=${1:?Usage: run_uv.sh sim|pi0|pi05 command [args...]}
shift
case "$env_name" in
    sim) project="$repo_root" ;;
    pi0|pi05) project="$repo_root/policy/$env_name" ;;
    *) echo "Unknown environment: $env_name" >&2; exit 2 ;;
esac
if [[ ! -x "$project/.venv/bin/python" ]]; then
    echo "Run bash $repo_root/script/setup_uv.sh $env_name first." >&2
    exit 1
fi
source "$repo_root/script/uv_env.sh"
export PATH="$project/.venv/bin:$PATH"
# imageio-ffmpeg supplies an executable even on hosts without system FFmpeg.
if ! command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg_binary=$("$project/.venv/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
    mkdir -p "$project/.venv/bin"
    ln -sf "$ffmpeg_binary" "$project/.venv/bin/ffmpeg"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
if [[ -d "$repo_root/ckpts/pi05_base/params" ]]; then
    export BICOORD_PI05_BASE_PARAMS=${BICOORD_PI05_BASE_PARAMS:-"$repo_root/ckpts/pi05_base/params"}
fi
# Keep the simulator patches and compiled extensions installed by setup_uv.sh.
exec uv run --project "$project" --no-sync "$@"
