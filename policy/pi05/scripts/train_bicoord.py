"""PBS-friendly BiCoord training/benchmark runner using unchanged openpi steps."""
from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import functools
import gc
import hashlib
import json
import logging
import os
import signal
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import jax
from jax.experimental import multihost_utils
import numpy as np
import orbax.checkpoint as ocp

import train as upstream
from openpi.shared import array_typing as at
from openpi.shared import normalize
from openpi.training import bicoord as bc
from openpi.training import checkpoints
from openpi.training import sharding

faulthandler.register(signal.SIGUSR1,all_threads=True)


def initialize_distributed(args):
    if args.ranks > 1:
        # Only global arrays enter the unchanged train_step; never fold RNG by rank.
        jax.distributed.initialize(coordinator_address=args.coordinator,
                                   num_processes=args.ranks, process_id=args.rank)


def manager_at(path):
    return ocp.CheckpointManager(
        path, item_handlers={"assets": checkpoints.CallbackHandler(),
                             "train_state": ocp.PyTreeCheckpointHandler(),
                             "params": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=None,
                                            async_options=ocp.AsyncOptions(timeout_secs=7200)),
    )


def save(manager, state, loader, step):
    if int(state.step) != step:
        raise ValueError("Checkpoint name must equal completed optimizer updates")

    def assets(directory):
        data = loader.data_config()
        normalize.save(directory / data.asset_id, data.norm_stats)
        recipe,recipe_sha256=bc.training_recipe(loader.config)
        bc.atomic_json(directory/"recipe.json",recipe)
        bc.atomic_json(directory / "sampler.json", {
            "completed_steps": step, "seed": loader.config.seed,
            "global_batch_size": loader.config.batch_size,
            "manifest_sha256": loader.manifest["sha256"],
            "recipe_sha256":recipe_sha256,
            "sampler": "torch.randperm(seed+epoch); global drop_last; contiguous rank slices",
        })

    with at.disable_typechecking():
        train_state, params = checkpoints._split_params(state)
    manager.save(step, {"assets": assets, "train_state": train_state, "params": {"params": params}})


def restore(manager, shape, step):
    """Restore complete state into the current layout, independent of saved device IDs."""
    with at.disable_typechecking():
        train_state, params = checkpoints._split_params(shape)
    items={"train_state":train_state,"params":{"params":params}}
    arguments={name:ocp.args.PyTreeRestore(item=item,restore_args=jax.tree.map(
        lambda value:ocp.ArrayRestoreArgs(dtype=value.dtype,global_shape=value.shape,
                                         sharding=value.sharding),item)) for name,item in items.items()}
    restored=manager.restore(step,args=ocp.args.Composite(**arguments))
    return checkpoints._merge_params(restored["train_state"],restored["params"])


def publish(manager, step):
    manager.wait_until_finished()
    if jax.process_index() == 0:
        link = bc.ROOT / "newest"
        temporary = link.with_name(".newest.tmp")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(f"checkpoints/{step}")
        os.replace(temporary, link)
        bc.atomic_json(bc.ROOT / "status.json", {"state": "training", "completed_steps": step,
                                               "newest": str(link), "updated_unix": time.time()})
    multihost_utils.sync_global_devices(f"published-{step}")
    for old in manager.all_steps():
        if old not in {step, 10_000, 30_000, 50_000}:
            manager.delete(old)


def state_arrays(state):
    return jax.tree.flatten_with_path((state.params, state.opt_state, state.ema_params))[0]


def compare_or_export(state, reference, export=False):
    """Check every element of raw params, Adam state, and EMA, streamed by leaf.

    Global collection is only a validation operation outside timed training.
    Reference resides on rank0's /tmp, so no persistent test data is created.
    """
    reference.mkdir(parents=True, exist_ok=True)
    numerator = np.zeros(3, np.float64)
    denominator = np.zeros(3, np.float64)
    maximum = np.zeros(3, np.float64)
    for index, (key, array) in enumerate(state_arrays(state)):
        value = np.asarray(multihost_utils.process_allgather(array, tiled=True))
        if jax.process_index() == 0:
            path = reference / f"{index:04d}.npy"
            if export:
                np.save(path, value)
            else:
                expected = np.load(path, mmap_mode="r")
                if expected.shape != value.shape or expected.dtype != value.dtype:
                    raise ValueError(f"State shape/dtype changed: {jax.tree_util.keystr(key)}")
                group = int(key[0].idx)
                for start in range(0, value.size, 1_000_000):
                    actual = value.reshape(-1)[start:start+1_000_000].astype(np.float64)
                    wanted = expected.reshape(-1)[start:start+1_000_000].astype(np.float64)
                    error = actual-wanted
                    numerator[group] += error @ error
                    denominator[group] += wanted @ wanted
                    maximum[group] = max(maximum[group], float(np.max(np.abs(error), initial=0)))
    packed = np.zeros(7, np.float64)
    if jax.process_index() == 0:
        relative = np.sqrt(numerator / np.maximum(denominator, 1e-30))
        # Full-state comparisons detect numerical drift; unchanged upstream functions,
        # global batch/RNG and data checks separately establish algorithm equivalence.
        packed[:3], packed[3:6] = relative, maximum
        packed[6] = export or bool(relative[0] < 1e-5 and relative[1] < .05 and relative[2] < 1e-5)
    packed = multihost_utils.broadcast_one_to_all(packed)
    return {"relative_l2_params_adam_ema": packed[:3].tolist(),
            "max_abs_params_adam_ema": packed[3:6].tolist(), "passed": bool(packed[6])}


def prefetch(iterator, depth):
    # Ordered host->device submission, preserving exact global batches and RNG.
    if not depth:
        yield from iterator
        return
    from collections import deque
    with ThreadPoolExecutor(max_workers=1) as executor:
        queue = deque(executor.submit(next, iterator) for _ in range(depth))
        while queue:
            batch = queue.popleft().result()
            queue.append(executor.submit(next, iterator))
            yield batch


def run(args):
    initialized = time.monotonic()
    if hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:16]))
    initialize_distributed(args)
    logging.basicConfig(level=logging.INFO)
    config = dataclasses.replace(bc.get_config(), fsdp_devices=args.fsdp, num_workers=args.workers)
    recipe,recipe_sha256=bc.training_recipe(config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if jax.process_index()==0:
        bc.atomic_json(output/"recipe.json",recipe)
    jax.config.update("jax_compilation_cache_dir", str(output.parent / "jax-cache"))
    if args.ranks>1:
        # Per-node /tmp is not a shared cache: cache hits must agree across ranks.
        jax.config.update("jax_enable_compilation_cache",False)
    manifest = json.loads(bc.MANIFEST.read_text())
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    if args.ranks>1:
        def global_key(key):
            bits=np.asarray(jax.random.key_data(key))
            return jax.random.wrap_key_data(jax.make_array_from_process_local_data(replicated,bits))
        train_rng,init_rng=global_key(train_rng),global_key(init_rng)
    checkpoint_manager = manager_at(bc.ROOT / "checkpoints") if args.mode == "train" else None
    latest = checkpoint_manager.latest_step() if checkpoint_manager else None
    print(json.dumps(dict(phase="initializing",devices=jax.device_count(),fsdp=args.fsdp,
                          rank=jax.process_index(),resume_step=latest)),flush=True)
    state, state_sharding = upstream.init_train_state(config, init_rng, mesh, resume=latest is not None)
    if latest is not None:
        metadata = json.loads((bc.ROOT / f"checkpoints/{latest}/assets/sampler.json").read_text())
        if (metadata["manifest_sha256"] != manifest["sha256"]
                or metadata["global_batch_size"] != config.batch_size or metadata["seed"]!=config.seed
                or metadata.get("recipe_sha256")!=recipe_sha256):
            raise ValueError("Resume input sampling or training recipe changed")
        state = jax.tree.map(lambda value, layout: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=layout),
                             state, state_sharding)
        state = restore(checkpoint_manager, state, latest)
    start_step = int(state.step)
    print(json.dumps(dict(phase="initialized",step=start_step)),flush=True)
    loader = bc.Loader(config, data_sharding, start_step=start_step, prefetch=args.prefetch)
    # Multihost JAX operations must be submitted in the same order on all ranks.
    # Prefetch only NumPy batches there; global array construction stays on the
    # main thread. Single-host device prefetch can safely remain asynchronous.
    host_prefetch=args.ranks>1 and bool(args.device_prefetch)
    source=loader.host_batches() if host_prefetch else iter(loader)
    data_iter = iter(prefetch(iter(source), args.device_prefetch))
    def step_function(rng, state, batch):
        state, info = upstream.train_step(config, rng, state, batch)
        # Logging-only norm reduction can be dead-code eliminated; training math
        # (including optimizer gradient clipping) is unchanged and fully checked.
        return state, ({"loss": info["loss"]} if args.lean_metrics else info)

    compiled = jax.jit(step_function,
                       in_shardings=(replicated, state_sharding, data_sharding),
                       out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    manifest_hash = manifest["sha256"]
    samples_hash = hashlib.sha256()
    for step in range(5):
        samples_hash.update(loader.sampler.indices(step).tobytes())
    timings, losses = [], []
    numerical = None
    save_seconds = 0.0
    stop = args.steps if args.mode == "benchmark" else config.num_train_steps
    for step in range(start_step, stop):
        tick = time.monotonic()
        batch = next(data_iter)
        if host_prefetch:
            batch=loader.to_device(batch)
        with sharding.set_mesh(mesh):
            state, info = compiled(train_rng, state, batch)
        jax.block_until_ready(state)
        if not np.isfinite(float(info["loss"])):
            raise FloatingPointError(f"Non-finite loss at update {step+1}")
        elapsed = time.monotonic()-tick
        if args.mode == "benchmark":
            if step >= args.warmup:
                timings.append(elapsed)
            if step < 5:
                losses.append(float(info["loss"]))
            if step == 4:
                numerical = compare_or_export(state, Path(args.reference), export=args.export_reference)
                if not numerical["passed"]:
                    raise ValueError(f"Numerical validation failed: {numerical}")
                if args.export_reference:
                    # Full float32 train state plus the numerical reference exceed
                    # node /tmp capacity. Stage only the checkpoint in its designated
                    # shared checkpoint root, and delete it after exact restoration.
                    restore_dir=bc.ROOT/".validation-checkpoints"/os.environ["PBS_JOBID"]
                    test_manager = manager_at(restore_dir)
                    saved = time.monotonic()
                    save(test_manager, state, loader, 5)
                    test_manager.wait_until_finished()
                    save_seconds = time.monotonic()-saved
                    shape = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=x.sharding),state)
                    del state
                    gc.collect()
                    state = restore(test_manager, shape, 5)
                    # Exact restoration at a fixed layout; compare every state leaf.
                    restore_check = compare_or_export(state, Path(args.reference))
                    if not restore_check["passed"] or any(restore_check["max_abs_params_adam_ema"]):
                        raise ValueError("Checkpoint restoration was not exact")
                    metadata = json.loads((restore_dir/"5/assets/sampler.json").read_text())
                    assert metadata["completed_steps"] == 5
                    resumed = bc.GlobalBatchSampler(len(bc.RawDataset(manifest)), config.batch_size, config.seed, start_step=5)
                    assert next(iter(resumed)) == loader.sampler.indices(5).tolist()
                    test_manager.delete(5)
                    test_manager.close()
                    if jax.process_index()==0:
                        bc.atomic_json(Path(args.reference)/"profile.json",dict(save_seconds=save_seconds,
                                      manifest_sha256=manifest["sha256"],checkpoint_restore_exact=True,
                                      devices=jax.device_count(),fsdp=args.fsdp,recipe_sha256=recipe_sha256))
        elif (step+1) % config.save_interval == 0 or step+1 == stop:
            save(checkpoint_manager, state, loader, step+1)
            publish(checkpoint_manager, step+1)
        if ((step+1) % config.log_interval == 0 or step == start_step
                or (args.mode=="benchmark" and (step<5 or (step+1)%10==0))):
            metrics = {key: float(value) for key, value in info.items()}
            print(json.dumps(dict(step=step+1, seconds=elapsed, **metrics)), flush=True)
            if jax.process_index() == 0:
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(step=step+1, **metrics))+"\n")
    if args.mode == "benchmark":
        result = dict(mode="benchmark", fsdp=args.fsdp, workers=args.workers,
                      prefetch=args.prefetch, device_prefetch=args.device_prefetch,
                      lean_metrics=args.lean_metrics,
                      devices=jax.device_count(), median_seconds=float(np.median(timings)),
                      mean_seconds=float(np.mean(timings)), p95_seconds=float(np.quantile(timings,.95)),
                      save_seconds=save_seconds, setup_seconds=time.monotonic()-initialized-sum(timings),
                      measured_steps=len(timings), losses=losses, numerical=numerical,
                      manifest_sha256=manifest_hash, first_batches_sha256=samples_hash.hexdigest(),
                      recipe_sha256=recipe_sha256,
                      device_stats=[device.memory_stats() for device in jax.local_devices()])
        if jax.process_index() == 0:
            bc.atomic_json(output / "result.json", result)
    else:
        checkpoint_manager.close()
        if jax.process_index() == 0:
            bc.atomic_json(bc.ROOT / "status.json", dict(state="trained", completed_steps=stop,
                                                        newest=str(bc.ROOT / "newest"), updated_unix=time.time()))
    if args.ranks > 1:
        multihost_utils.sync_global_devices("runner-complete")
        jax.distributed.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "benchmark", "train"], required=True)
    parser.add_argument("--output", default="/tmp/bicoord-pi05")
    parser.add_argument("--fsdp", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--device-prefetch", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reference", default="/tmp/bicoord-pi05/reference")
    parser.add_argument("--export-reference", action="store_true")
    parser.add_argument("--lean-metrics", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--coordinator")
    parser.add_argument("--xla-flags",default="",help="Execution flags applied by the shell launcher")
    return parser.parse_args()


if __name__ == "__main__":
    args_global = parse_args()
    if args_global.mode == "prepare":
        print(json.dumps(bc.prepare() | {"episodes": "omitted"}), flush=True)
    else:
        run(args_global)
