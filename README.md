# BiCoord: A Bimanual Manipulation Benchmark towards Long-Horizon Spatial-Temporal Coordination

**Xingyu Peng<sup>\*,1,2</sup>, Chen Gao<sup>\*,1,3</sup>,  Liankai Jin<sup>\*,1</sup>, Annan Li<sup>1</sup>, Si Liu<sup>†,1</sup>**

Beihang University<sup>1</sup>, Zhongguancun Academy<sup>2</sup>, National University of Singapore<sup>3</sup>

**Website:** https://buaa-colalab.github.io/BiCoord/

<img src="./assets/fig1.png" alt="description" style="display: block; margin: auto; width: 100%;">

# Overview, Tasks & LeaderBoard

Please see https://buaa-colalab.github.io/BiCoord/ for details.

# Installation

【Step 1】Clone the Repository

```
git clone --branch uv https://github.com/ZhaoRunyi/BiCoord-Bench.git
```

【Step 2】Build environments (uv first; original Conda option below)

The simulation, π0 and π0.5 environments remain separate because their checkpoint
libraries differ. Their dependencies are committed in `pyproject.toml` / `uv.lock`.
Use CUDA 12.4 with `nvcc`, GCC ≥9 and the workspace uv installation:

```bash
source /home/MLTL-INTERN/shaoyang/zvi/.uv/env.sh
cd /home/MLTL-INTERN/shaoyang/zvi/mul_ttt_ws/BiCoord-Bench
export CUDA_HOME=/usr/local/cuda-12.4
export CC=/home/MLTL-INTERN/shaoyang/zvi/gcc9/root/opt/rh/gcc-toolset-9/root/usr/bin/gcc
export CXX=/home/MLTL-INTERN/shaoyang/zvi/gcc9/root/opt/rh/gcc-toolset-9/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$(dirname "$CXX"):$PATH"
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 FORCE_CUDA=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.7.6
export GIT_LFS_SKIP_SMUDGE=1
uv python install 3.11
for project in "$PWD" "$PWD/policy/pi0" "$PWD/policy/pi05"; do
    uv sync --project "$project" --frozen --no-dev
    uv sync --project "$project" --frozen --no-dev --extra cuda
    uv run --project "$project" --no-sync bash "$PWD/script/_install.sh" --patch-only
done
uv run --no-sync bash script/_download_assets.sh
```

For another host, set `CC` and `CXX` to its GCC ≥9 compiler. Optional EL8 GCC
installation commands and dependency explanations are in [docs/uv.md](docs/uv.md).
The existing `_install.sh --patch-only` applies RoboTwin's SAPIEN/MPLib adjustments
without modifying uv's cached wheel files. `_download_assets.sh` downloads the
RoboTwin assets and overlays BiCoord objects. No additional setup wrappers are used.

Check the prepared environments without a GPU workload:

```bash
for project in "$PWD" "$PWD/policy/pi0" "$PWD/policy/pi05"; do
    JAX_PLATFORMS=cpu uv run --project "$project" --no-sync python -c 'import sapien, mplib, torch, curobo, pytorch3d; print("imports OK")'
done
```

For the original **Conda** installation path, follow the
[RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).
The existing pip installer remains available inside the activated environment:

```bash
conda create -n BiCoord python=3.10 -y
conda activate BiCoord
bash script/_install.sh
bash script/_download_assets.sh
```

See [the all-task π0.5 guide](docs/pi05_bicoord.md) for the validated JAX training
recipe and standard evaluation command. PBS submission and experiment monitoring
belong outside the code repository.

【Step 3】Download Additional Assets

We have modified some objects in the original RoboTwin-OD dataset. Please download the additional objects at https://huggingface.co/datasets/GradiusTwinbee/BiCoord/blob/main/objects.zip and extract them to `BiCoord-Bench/assets/objects`.

`bash script/_download_assets.sh` downloads the base RoboTwin assets and applies
these BiCoord object modifications automatically.

【Step 4】For the env installation of specific policies, please refer to [RoboTwin 2.0 Usage Guide - RoboTwin 2.0 Offical Document](https://robotwin-platform.github.io/doc/usage/index.html)

In our practice, we install the RoboTwin 2.0 Environment first, and then install the policies' envs respectively based on this basic environment. 

# Data Preparation

We recommend to download official training data at https://huggingface.co/datasets/GradiusTwinbee/BiCoord and keep data at `/home/MLTL-INTERN/shaoyang/zvi/mul_ttt_ws/data/BiCoord`. Otherwise, you can generate data following the guidance at [Collect Data - RoboTwin 2.0 Offical Document](https://robotwin-platform.github.io/doc/usage/collect-data.html), and run correspnding  `split_stages.py` to obtain stage information.

After preparation,  `/home/MLTL-INTERN/shaoyang/zvi/mul_ttt_ws/data/BiCoord` should look like

```
/home/MLTL-INTERN/shaoyang/zvi/mul_ttt_ws/data/BiCoord/
├── balance_roller/
│   └── demo_clean/
│       ├── _traj_data/
│       ├── .ipynb_checkpoints/
│       ├── data/					#Trajectory data, including endpose, joint_action and observation
│       ├── instructions/			#Task instruction
│       ├── stages/					#Stage information for each episode
│       ├── video/					#Video
│       ├── scene_info.json
│       ├── seed.txt				#The corresponding random seeds for producing the trajectory data
│       └── split_stages.py			#Produce stage information based on trajectory data
├── build_tower_with_blocks/
├── clean_table/
├── ......
```

After preparing data at the absolute path above, you need to convert the data into the training format according to the policy you choose. For data conversion of baseline policies, please follow the guidance at [RoboTwin 2.0 Usage Guide - RoboTwin 2.0 Offical Document](https://robotwin-platform.github.io/doc/usage/index.html).

# Training & Evaluation

Use the existing standard entrypoint `script/eval_policy.py`, with absolute
checkpoint and output paths. For example, within a PBS GPU allocation:

```bash
source /home/MLTL-INTERN/shaoyang/zvi/.uv/env.sh
cd /home/MLTL-INTERN/shaoyang/zvi/mul_ttt_ws/BiCoord-Bench
uv run --project "$PWD/policy/pi05" --no-sync python script/eval_policy.py \
    --config policy/pi05/deploy_policy.yml --overrides \
    --task_name balance_roller --task_config demo_clean --policy_name pi05_bicoord \
    --checkpoint_dir /home/MLTL-INTERN/shaoyang/zvi/mul_ttt_ws/ckpts/BiCoord_pi05/checkpoints/50000 \
    --ckpt_setting pi05_all_50000 --seed 0 --test_num 100 --resume True \
    --video_episodes 1 --output_dir /tmp/bicoord-pi05-eval/balance_roller
```

Task-list iteration, PBS resource allocation and aggregated reports stay in the
external experiment directory. Dataset/checkpoint symlinks are not required.
The existing π0/π0.5 conversion scripts accept `--data-root` and `--output-root`.

For convenience, we also provide all checkpoints of baseline policies (DP, RDT, OpenVLA-OFT, Pi0 under single&multi task setting) at https://huggingface.co/Oshwiciqwq/BiCoord-checkpoints.

# Acknowledgement

We sincerely thank [RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin) for their outstanding contributions to bimanual manipulation simulation, convenient action APIs and open-source release.

We also thank [DP](https://github.com/real-stanford/diffusion_policy), [RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer), [OpenVLA-OFT](https://github.com/moojink/openvla-oft) and [Pi0](https://github.com/Physical-Intelligence/openpi) for their outstanding and representive contributions in manipulation and VLAs.

# License
This benchmark is released under the **[MIT License](https://opensource.org/licenses/MIT)**.
