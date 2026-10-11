# uv setup for simulation, Pi0, and Pi0.5

## Requirements and environment layout

Use Linux x86_64, `uv`, an NVIDIA driver compatible with CUDA 12.4, a CUDA 12.4
toolkit containing `nvcc`, and GCC 9 or newer. No Conda environment is required.
The setup script selects a complete `/opt/rh/gcc-toolset-9` when available.
On an EL8 host with GCC 8 it extracts official Rocky Linux GCC 9 RPMs into
`zvi/gcc9` (or `BICOORD_TOOLCHAIN_ROOT`), without installing system packages. Otherwise set
`CC` and `CXX` to a suitable compiler. Python 3.11 is installed by uv.

| Environment | Project | Virtual environment |
| --- | --- | --- |
| Simulation | repository root | `.venv` |
| Pi0 | `policy/pi0` | `policy/pi0/.venv` |
| Pi0.5 | `policy/pi05` | `policy/pi05/.venv` |

Each project has a committed `uv.lock`. The two vendored OpenPI projects remain
separate because their Orbax checkpoint versions differ. All three use PyTorch
2.6.0 / torchvision 0.21.0 CUDA 12.4 wheels, matching the original OpenPI lockfiles'
PyTorch versions. SAPIEN 3.0.0b1 and MPLib 0.2.1 are retained. Open3D 0.18.0
supports this host's glibc 2.28; 0.19.0 wheels require glibc 2.31.

SwanLab is included because both vendored training scripts import it. LeRobot's
standalone Rerun viewer is excluded through uv's `exclude-dependencies`: its
NumPy 1-compatible wheels require glibc 2.31, while its glibc 2.28 wheels require
NumPy 2. This optional viewer is not part of the conversion/training/evaluation
entrypoints; NumPy 1.26 is retained for the simulator and OpenPI.
Policy environments pin PyAV 14.2.0, which supplies compatible FFmpeg libraries.
Conversion and training select LeRobot's PyAV decoder explicitly; TorchCodec's
default decoder otherwise requires additional system FFmpeg shared libraries.

CUDA dependencies are fixed to cuRobo v0.7.6 and PyTorch3D v0.7.9 commits, with
Warp 1.2.1. They are installed through the `cuda` extra and compiled against the
environment's PyTorch without build isolation. Static dependency metadata lets
uv resolve the base environment before PyTorch is installed. The setup script
then builds native extensions and applies the two upstream RoboTwin simulator
patches to SAPIEN and MPLib. Patching replaces file inodes so uv's wheel cache
remains intact.

## Install and verify

From the repository root:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
# A100. Override this for a different GPU, e.g. 8.9 for an RTX 4090.
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
bash script/setup_uv.sh all
```

Use `sim`, `pi0`, or `pi05` instead of `all` to prepare one environment.
The workspace maintains an independent uv installation at `zvi/.uv`: its
`bin/uv` wrapper uses `cache/`, `python/`, and `tools/` under that root.
Scratch files default to `/tmp/zvi-uv-$UID`.
The project virtual environments stay at the three locations listed above.
`script/uv_env.sh` selects this installation automatically; `UV_ROOT` can select
a different workspace uv root. For an interactive shell, source
`/home/MLTL-INTERN/shaoyang/zvi/.uv/env.sh`. Account-wide uv settings are untouched.
`CC`, `CXX`, `CUDA_HOME`, and `MAX_JOBS` are respected. Check videos and environment
records default to `/tmp/zvi-bicoord-uv-check/`; set `BICOORD_ARTIFACT_DIR`
to override this for a particular run.

The installer checks simulation imports, cuRobo imports, PyTorch3D CUDA sampling,
and each OpenPI training/data-loader configuration. These checks do not establish
benchmark success rates. Full collection/evaluation also requires working Vulkan
drivers and the simulation assets.

For later commands use `script/run_uv.sh`. It preserves the current working
directory, uses the selected project's environment without resynchronizing, adds
its binaries to `PATH`, disables JAX preallocation by default, and supplies
FFmpeg from imageio-ffmpeg if the host does not have FFmpeg installed.

```bash
bash script/run_uv.sh sim python script/check_uv_env.py sim
bash script/run_uv.sh pi0 python script/check_uv_env.py pi0
bash script/run_uv.sh pi05 python script/check_uv_env.py pi05
```

After changing dependencies, rerun `setup_uv.sh` to resynchronize the `cuda`
extra and reapply simulator patches. A bare `uv sync` without `--extra cuda`
removes native CUDA dependencies.

## Data, assets, and checkpoints

Keep downloaded artifacts outside Git:

```text
workspace/
├── BiCoord-Bench/
├── data/BiCoord/<task>/demo_clean/
└── ckpts/
    ├── pi05_base/{params,assets}/
    ├── BiCoord_pi0/pi0_base_aloha_robotwin_full/
    │   ├── <task>/30000/{params,assets}/
    │   └── pi0/50000/{params,assets,train_state}/
    └── BiCoord_pi05/                # Future Pi0.5 fine-tuning outputs
```

```bash
python3 script/link_workspace.py
bash script/_download_assets.sh
```

The link script connects the repository's `data`, `ckpts`, and policy
`checkpoints` paths to the sibling artifact directories. It refuses to replace
existing paths. The assets script downloads RoboTwin embodiments, objects, and
background textures, then overlays BiCoord's modified objects. It keeps archives
and completion markers under ignored `assets/.archives`; repeated runs skip
completed extraction. Set `HF_ENDPOINT` before running it to select a Hub
endpoint. Public downloads do not require an account token.
The final step runs the upstream embodiment configuration generator, producing
the concrete cuRobo YAML files from the downloaded templates.

`run_uv.sh` selects local `ckpts/pi05_base/params` for Pi0.5 initialization when
available. `BICOORD_PI05_BASE_PARAMS` overrides this path. Without local weights
the original official GCS path remains the fallback.

## Collection, conversion, training, and evaluation

The policy shell entrypoints resolve their paths independently of the invoking
directory. Their arguments are quoted and commands run in the correct uv
environment. Fine-tuning accepts additional training options and no longer
silently overwrites an existing run; pass `--overwrite` explicitly when intended.

```bash
# From the repository root; task/config/experiment identifiers remain upstream's.
bash collect_data.sh balance_roller demo_clean 0
bash policy/pi0/process_data_pi0.sh balance_roller demo_clean 100
bash policy/pi05/process_data_pi05.sh balance_roller demo_clean 100

# Set the training config's repo_id to the converted LeRobot dataset first.
bash policy/pi0/compute_norm_stats.sh pi0_base_aloha_robotwin_full
bash policy/pi05/compute_norm_stats.sh pi05_aloha_full_base
bash policy/pi05/finetune.sh pi05_aloha_full_base balance_roller 0,1

# Published single-task Pi0 checkpoints use checkpoint_id 30000.
# Set this in policy/pi0/deploy_policy.yml before evaluation.
bash policy/pi0/eval.sh balance_roller demo_clean \
  pi0_base_aloha_robotwin_full balance_roller 0 0
```

The BiCoord release includes Pi0 single-task and multi-task checkpoints. Its
`pi05` code directory does not imply a released BiCoord fine-tuned Pi0.5 model.

Evaluation also accepts `--output_dir`, `--test_num` (default 100), and
`--video_episodes` (default all; use 1 for only the first rollout video) through
`script/eval_policy.py --overrides`. Configs, individual episode records, SR/SSR/TL
metrics, and requested videos are retained in the chosen output directory.
The default evaluation output directory is under `/tmp/bicoord-eval/`.
These options do not change the policy or the default 100-rollout protocol.
For Pi0, `--resume True` continues an existing output directory from its last
completed episode, preserving counters, seed progression, and policy sampling
keys. Use the same checkpoint and evaluation options when resuming.
