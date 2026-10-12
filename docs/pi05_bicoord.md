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

Persistent root: `../ckpts/BiCoord_pi05`.

- `checkpoints/<completed_steps>/`: complete raw parameters, optimizer state, EMA inference parameters, normalization assets, and sampler cursor/manifest identity.
- `newest`: atomic symlink updated after a successful full save every 2,000 updates.
- 10k, 30k and 50k are retained. Other old saves are removed after newest is published.
- Restart reads the latest complete checkpoint and reconstructs the next global batch directly from its completed-step cursor; prefetched batches are not treated as consumed data.

## Validation and acceleration

CPU checks compare actual upstream π0 conversion pixels/state/action, delta/gripper conventions, episode tails, 8/16-device batch ordering, and resume boundaries. A real spawned-worker loader checks the complete transformed input shapes.

Restoration explicitly supplies the current sharding layout rather than inheriting saved GPU IDs. Before production training, the selected PBS allocation also runs one simulation rollout with the official π0.5 base model and the prepared BiCoord normalization; success of this untrained rollout is not a requirement.

PBS benchmarks first export all parameter, Adam and EMA arrays after five updates. Each execution variant compares every element against that reference. The reference also performs a full checkpoint round trip and requires exact restoration. Floating-point rounding from hardware, compiler kernels and reduction layout is allowed; algorithm, data, random sampling, global batch and precision remain fixed. The numerical drift gate is relative L2 <1e-5 for parameters/EMA and <0.05 for Adam state, with maximum absolute errors reported. These thresholds supplement the unchanged-code and data/RNG audits; they do not redefine the optimizer.

Up to eight variants cover FSDP layout, loader workers/prefetch, ordered device prefetch, omission of logging-only norms, and 8/16 A100 devices. Start training as soon as a correctly validated variant projects below 16 hours. If eight variants cannot meet that target, select the fastest passing variant. ETA includes measured startup, conservative steady step timing, 25 full saves and a 10% margin; evaluation is separate.

The user subsequently accepted the validated 8-card estimate of approximately 17.5 hours and authorized at most five further 16-card attempts before launching the fastest passing option. Those five attempts are the repaired FSDP8/FSDP16 runs, latency hiding with FSDP8, latency hiding plus collective combining with FSDP16, and latency hiding plus Triton GEMM with FSDP16. A new benchmark attempt has a 20-minute deadline; execution flags follow the [official JAX GPU performance guidance](https://docs.jax.dev/en/latest/gpu_performance_tips.html), without changing precision or training semantics.

All GPU work runs in PBS allocations. The submit-host controller runs in detached screen `bicoord_pi05`. Benchmark logs, numerical references, questions, status and evaluation artifacts live in `/tmp/bicoord-pi05-training-20261011` on the relevant host; node `/tmp` is not shared.

JAX reserves 80% of device memory to leave room for NCCL. Numerical references are removed before the long training run. A full temporary restore-test checkpoint can be staged under the designated checkpoint root when `/tmp` capacity is insufficient; it is deleted once restoration has been validated.

Cross-host execution submits all JAX operations on the main thread and prefetches only host arrays. GPU autotuning runs locally (`xla_gpu_shard_autotuning=false`), and node-local persistent compilation caching is disabled because `/tmp` is not shared. The two original cross-host variants are retried after this execution repair; they do not count as additional algorithm variants.

## Evaluation and reference

### Logging the completed run

The original 50k run recorded JSONL rather than W&B or TensorBoard events.
`script/pi05_tracking.py` backfills its 501 recorded losses and recorded step
durations into both formats, without restarting training. Learning rate is
explicitly labelled as derived from the saved recipe at optimizer index
`completed_steps - 1`. Missing intermediate losses, gradients and original
per-point wall times are not fabricated. Backfilled events have export times.

The W&B project is `openpi_bicoord`. Without credentials, the training run is
saved offline. With `--watch`, a CPU-only screen process polls the existing
PBS evaluation every minute; after login it syncs that same run and appends
partial task results, the inference video and final evaluation metrics.
Partial task rates are labelled `sr_partial`/`ssr_partial`; complete aggregate
rates are published only after all 18 tasks finish. Checkpoints are not uploaded.

Credentials can be kept outside the repository in `zvi/.wandb/netrc` (directory
mode 700, file mode 600). Set `NETRC` to that path when running the policy
environment's `wandb login --cloud --relogin`; enter the key at its hidden
interactive prompt. Never put a key in command arguments, source or logs.
The tracker also loads this saved credential into `WANDB_API_KEY` in process
memory. For interactive tools, source `zvi/.wandb/env.sh` to export the same
key and `WANDB_PROJECT=openpi_bicoord`; no second plaintext key file is needed.

Logging artifacts: `/tmp/bicoord-pi05-training-20261011/tracking/`.
TensorBoard logs are under its `tensorboard/` directory; `status.json` records
the run ID, upload status and online run URL. A localhost-only TensorBoard
server on the submit host can serve these events with `--load_fast=false`.
Use an SSH tunnel or VS Code port forwarding to view port 6006.

After training, run all 18 tasks ×100 rollouts using the repository's original evaluation success predicates and unseen instructions. Save one balance_roller inference video. Report mean task SR, SSR and successful-trajectory length.

CogWAM is the requested community reference: [paper](https://arxiv.org/html/2609.37721), [code](https://github.com/HorizonRobotics/CogWAM). It reports a multi-task BiCoord π0.5 baseline of SR 35.6%, SSR 50.8%. Its released code/checkpoint targets RoboDojo, and its π0.5 BiCoord training hyperparameters are not disclosed; CogWAM's own RoboDojo training settings are not substituted for the requested openpi recipe. Single-task BiCoord π0.5 results were not located.
