#!/usr/bin/env bash
# Shared workspace defaults; sourcing this file does not create an environment.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(dirname -- "$(dirname -- "$repo_root")")
export UV_ROOT=${UV_ROOT:-"$workspace_root/.uv"}
export UV_CACHE_DIR="$UV_ROOT/cache"
export UV_PYTHON_INSTALL_DIR="$UV_ROOT/python"
export UV_PYTHON_BIN_DIR="$UV_ROOT/bin"
export UV_TOOL_DIR="$UV_ROOT/tools"
export UV_TOOL_BIN_DIR="$UV_ROOT/bin"
export TMPDIR="${TMPDIR:-/tmp/zvi-uv-$UID}"
mkdir -p "$TMPDIR"
export PATH="$UV_ROOT/bin:$PATH"
if [[ ! -x "$UV_ROOT/bin/uv" ]]; then
    echo "Install workspace uv at $UV_ROOT/bin/uv (or set UV_ROOT)." >&2
    return 1
fi
