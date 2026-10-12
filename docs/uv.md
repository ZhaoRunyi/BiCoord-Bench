# uv environment notes

Environment creation and execution commands are in [README](../README.md#installation),
with uv first and the original Conda option second. The three virtual environments
are `.venv`, `policy/pi0/.venv` and `policy/pi05/.venv`. Use
`source /home/MLTL-INTERN/shaoyang/zvi/.uv/env.sh` for the independent workspace uv.
Its cache, managed Python and tools stay in `zvi/.uv`; account-wide uv is untouched.

The simulation uses SAPIEN 3.0.0b1, MPLib 0.2.1, Open3D 0.18.0 (EL8/glibc 2.28),
and PyTorch 2.6.0 with CUDA 12.4. The policy environments retain distinct Orbax
versions. CUDA extras pin cuRobo 0.7.6, PyTorch3D 0.7.9 and Warp 1.2.1.
Install base dependencies before CUDA extras, since native builds import torch.
Use `--no-sync` for execution to preserve compiled extensions and simulator
patches. After a dependency sync, apply the existing `_install.sh --patch-only`
in that environment again. A bare sync without `--extra cuda` removes CUDA extras.

The patch command atomically replaces the installed SAPIEN/MPLib files so uv's
hardlinked wheel cache remains intact. It is a mode of the existing installer,
not a separate helper. Assets are handled by the existing `_download_assets.sh`;
it includes bounded download retries, extraction markers and the BiCoord overlay.

## Optional GCC 9 installation on EL8

The current workspace already has `zvi/gcc9`. To rebuild it without sudo, run:

```bash
compiler_install=/home/MLTL-INTERN/shaoyang/zvi/gcc9
rpm_base=https://dl.rockylinux.org/vault/rocky/8.4/AppStream/x86_64/os/Packages
mkdir -p "$compiler_install/rpms" "$compiler_install/root"
for package in gcc-toolset-9-gcc gcc-toolset-9-gcc-c++ gcc-toolset-9-libstdc++-devel; do
    archive="$compiler_install/rpms/$package-9.2.1-2.3.el8.x86_64.rpm"
    if [[ ! -f "$archive" ]]; then
        curl --fail --location --retry 3 --retry-max-time 120 \
            --connect-timeout 15 --max-time 180 \
            "$rpm_base/${archive##*/}" -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    rpm2cpio "$archive" | (cd "$compiler_install/root" && cpio -idm --quiet)
done
export CC="$compiler_install/root/opt/rh/gcc-toolset-9/root/usr/bin/gcc"
export CXX="$compiler_install/root/opt/rh/gcc-toolset-9/root/usr/bin/g++"
"$CXX" --version
```

## Artifact paths

Data: `/home/MLTL-INTERN/shaoyang/zvi/mul_ttt_ws/data/BiCoord`.
Checkpoints: `/home/MLTL-INTERN/shaoyang/zvi/mul_ttt_ws/ckpts`.
No repository data/checkpoint symlinks are needed. Existing conversion commands
accept `--data-root` / `--output-root`; model evaluation accepts `--checkpoint_dir`.
Tests, videos and benchmark outputs use dedicated `/tmp` directories. The normal
uv-created virtual-environment interpreter links remain implementation details of uv.
