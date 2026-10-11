#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
target=${1:-all}
case "$target" in sim|pi0|pi05|all) ;; *) echo "Usage: $0 [sim|pi0|pi05|all]" >&2; exit 2 ;; esac
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$(dirname "$repo_root")/.uv-cache"}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-"$(dirname "$repo_root")/.uv-python"}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "Set CUDA_HOME to an installed CUDA 12.4 toolkit (nvcc required)." >&2
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
if [[ -z "${CC:-}" && -z "${CXX:-}" ]]; then
    compiler_root=/opt/rh/gcc-toolset-9/root/usr
    if [[ ! -x "$compiler_root/bin/g++" ]]; then
        if [[ $(gcc -dumpversion | cut -d. -f1) -lt 9 ]]; then
            bash "$repo_root/script/setup_toolchain.sh"
            compiler_root="$(dirname "$repo_root")/.toolchains/gcc9/root/opt/rh/gcc-toolset-9/root/usr"
        else
            compiler_root=""
        fi
    fi
    if [[ -n "$compiler_root" ]]; then
        export CC="$compiler_root/bin/gcc"
        export CXX="$compiler_root/bin/g++"
        export PATH="$compiler_root/bin:$PATH"
    fi
fi
export MAX_JOBS=${MAX_JOBS:-4}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
export FORCE_CUDA=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.7.6
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export PYTHONUNBUFFERED=1
uv python install 3.11
for env_name in sim pi0 pi05; do
    if [[ "$target" != all && "$target" != "$env_name" ]]; then continue; fi
    project="$repo_root"
    if [[ "$env_name" != sim ]]; then project="$repo_root/policy/$env_name"; fi
    echo "Preparing $env_name environment at $project/.venv"
    # Bootstrap torch and the build tools before compiling native extensions.
    uv sync --project "$project" --locked --no-dev
    uv sync --project "$project" --locked --no-dev --extra cuda
    "$project/.venv/bin/python" "$repo_root/script/patch_sim_dependencies.py"
    "$project/.venv/bin/python" "$repo_root/script/check_uv_env.py" "$env_name"
done
