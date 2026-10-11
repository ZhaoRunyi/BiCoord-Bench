#!/usr/bin/env bash
# Install an EL8-compatible compiler into the workspace without sudo.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
toolchain="$(dirname "$repo_root")/.toolchains/gcc9"
base=https://dl.rockylinux.org/vault/rocky/8.4/AppStream/x86_64/os/Packages
mkdir -p "$toolchain/rpms" "$toolchain/root"
for name in gcc-toolset-9-gcc gcc-toolset-9-gcc-c++ gcc-toolset-9-libstdc++-devel; do
    filename="$name-9.2.1-2.3.el8.x86_64.rpm"
    archive="$toolchain/rpms/$filename"
    if [[ ! -f "$archive" ]]; then
        curl --fail --location --retry 3 --retry-max-time 120 --connect-timeout 15 --max-time 180 \
            "$base/$filename" -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    marker="$toolchain/rpms/$filename.extracted"
    if [[ ! -f "$marker" ]]; then
        rpm2cpio "$archive" | (cd "$toolchain/root" && cpio -idm --quiet)
        touch "$marker"
    fi
done
compiler_root="$toolchain/root/opt/rh/gcc-toolset-9/root/usr"
"$compiler_root/bin/g++" --version
printf '#include <iostream>\nint main() { std::cout << "GCC toolchain OK\\n"; }\n' \
    | "$compiler_root/bin/g++" -x c++ - -o "$toolchain/compiler-check"
"$toolchain/compiler-check"
