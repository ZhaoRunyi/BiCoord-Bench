# BiCoord all-task π0.5 baseline

This run fine-tunes official `pi05_base` on all 18 clean BiCoord tasks, with 100 demonstrations per task (647,479 eligible frames). It uses JAX and the existing openpi `init_train_state` / `train_step` without changing the loss, noise, precision, optimizer formulas, or EMA update.

## Agreed recipe

- Global batch 64; 50,000 optimizer updates; inherited seed 42.
- Official π0.5 LIBERO learning-rate recipe: warm up for 10,000 steps to 5e-5, then hold 5e-5. EMA 0.999. AdamW settings remain inherited from openpi defaults.
- Full fine-tuning; inherited model defaults including action horizon 50, action dimension 32 and bfloat16 compute. π0.5 quantile normalization and discrete state input remain enabled.
- BiCoord π0 physical representation: 14 joints/grippers ordered left arm 6, left gripper, right arm 6, right gripper. No Trossen coordinate adaptation. Observation at t, first action at t+1; joint actions are deltas from the current observation, grippers absolute. Action chunks repeat the final frame at episode boundaries.
- Three images reproduce the official π0 HDF5 converter's 640×480 JPEG round trip. CHW input, native Aloha decoding, PIL bilinear resize/pad to 224×224, and upstream model augmentation are retained.
- One seeded `seen` instruction per trajectory; globally shuffled training frames and global drop-last batching. No task balancing. All variants use the same seeded frame order and global training RNG.

Default settings that already match the request are inherited instead of restated in configuration code. See `policy/pi05/src/openpi/training/bicoord.py` for necessary overrides.

## Checkpoints and restart

Persistent root: `/home/MLTL-INTERN/shaoyang/zvi/mul_ttt_ws/ckpts/BiCoord_pi05`.

- `checkpoints/<completed_steps>/`: complete raw parameters, optimizer state, EMA inference parameters, normalization assets, and sampler cursor/manifest identity.
- `newest`: atomic symlink updated after a successful full save every 2,000 updates.
- 10k, 30k and 50k are retained. Other old saves are removed after newest is published.
- Restart reads the latest complete checkpoint and reconstructs the next global batch directly from its completed-step cursor; prefetched batches are not treated as consumed data.

## Validation and execution

`policy/pi05/scripts/validate_bicoord.py` checks physical I/O, actual upstream
converter pixels, delta/gripper conventions, sampler ordering and resume semantics.
`policy/pi05/scripts/train_bicoord.py` is the reusable training entrypoint; PBS
allocation, benchmark variants, worker cleanup and job monitoring belong to the
external experiment directory, not this repository.

Training uses unmodified openpi `init_train_state` / `train_step`. The validated
execution was 8 A100s, FSDP8, 8 loader workers, host prefetch 4, device prefetch 2
and logging-only norm omission. Its measured projection was 17.5 hours; the
50,000-step run finished in about 15h44m. Numerical validation allowed only
hardware/compiler/parallel floating-point differences while fixing the algorithm,
RNG, global data order, precision, optimizer and global batch.

Example commands (GPU training must run inside an external PBS allocation):

```bash
source /home/MLTL-INTERN/shaoyang/zvi/.uv/env.sh
source /home/MLTL-INTERN/shaoyang/zvi/.wandb/env.sh
cd /home/MLTL-INTERN/shaoyang/zvi/mul_ttt_ws/BiCoord-Bench
JAX_PLATFORMS=cpu uv run --project "$PWD/policy/pi05" --no-sync python \
    policy/pi05/scripts/train_bicoord.py --mode prepare --output /tmp/bicoord-pi05-prepare
uv run --project "$PWD/policy/pi05" --no-sync python \
    policy/pi05/scripts/train_bicoord.py --mode train --output /tmp/bicoord-pi05-train \
    --fsdp 8 --workers 8 --prefetch 4 --device-prefetch 2 --lean-metrics
```

The training entrypoint writes actual metrics directly to W&B project
`openpi_bicoord` and TensorBoard. Missing authentication uses offline W&B records;
`wandb sync` uploads them later. Learning rate is labelled derived from the saved
schedule at optimizer index `completed_steps - 1`. Benchmarks do not create W&B
runs. The original completed run predates native logging: its 501 recorded losses
and recorded step timings were backfilled; no unrecorded gradients or intermediate
losses were fabricated. Historical backfill and experiment monitoring live outside
this repository under `/tmp/bicoord-pi05-training-20261011`.

## Evaluation and reference

Use the existing `script/eval_policy.py` command in README; the external PBS runner iterates all 18 tasks ×100 rollouts using the repository's original evaluation success predicates and unseen instructions. Save one balance_roller inference video. Report mean task SR, SSR and successful-trajectory length.

CogWAM is the requested community reference: [paper](https://arxiv.org/html/2609.37721), [code](https://github.com/HorizonRobotics/CogWAM). It reports a multi-task BiCoord π0.5 baseline of SR 35.6%, SSR 50.8%. Its released code/checkpoint targets RoboDojo, and its π0.5 BiCoord training hyperparameters are not disclosed; CogWAM's own RoboDojo training settings are not substituted for the requested openpi recipe. Single-task BiCoord π0.5 results were not located.
