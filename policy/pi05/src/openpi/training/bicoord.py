"""BiCoord physical I/O, resumable global frame sampling, and the official pi05 recipe.

The random-access reader reproduces the two upstream image-mode converters without
creating 1.9 million PNG files. Model transforms and train_step remain upstream.
"""
from __future__ import annotations

from collections import OrderedDict
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import cv2
import h5py
import jax
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.shared import normalize
from openpi.training import config as configs
from openpi.training import data_loader
from openpi.training import optimizer
from openpi.training.weight_loaders import CheckpointWeightLoader

WS = Path(__file__).resolve().parents[6]
ROOT = Path(os.environ.get("BICOORD_PI05_ROOT", WS / "ckpts/BiCoord_pi05"))
DATA_ROOT = WS / "data/BiCoord"
NAME = "pi05_bicoord_all"
ASSET_ID = "bicoord_all_clean"
MANIFEST = ROOT / "data_manifest.json"
CAMERAS = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}


def get_config():
    """Official pi05_libero LR/EMA; unchanged AdamW/model defaults are inherited.

    Source: Physical-Intelligence/openpi@215abfb training/config.py pi05_libero.
    The vendored RoboTwin config lacks that named recipe.
    """
    template = configs.get_config("pi05_aloha_full_base").data
    return configs.TrainConfig(
        name=NAME, model=Pi0Config(pi05=True),
        data=dataclasses.replace(template, repo_id=ASSET_ID, adapt_to_pi=False),
        weight_loader=CheckpointWeightLoader(str(WS / "ckpts/pi05_base/params")),
        batch_size=64, num_train_steps=50_000, save_interval=2_000,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5,
                                                 decay_steps=1_000_000, decay_lr=5e-5),
        ema_decay=0.999,
        assets_base_dir=str(ROOT / "assets"), checkpoint_base_dir=str(ROOT),
        wandb_enabled=False,
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def training_recipe(config):
    """Record inherited values without overriding any training defaults."""
    recipe=dict(model=dataclasses.asdict(config.model),optimizer=dataclasses.asdict(config.optimizer),
                lr_schedule=dataclasses.asdict(config.lr_schedule),ema_decay=config.ema_decay,
                seed=config.seed,global_batch_size=config.batch_size,
                freeze_filter=str(config.freeze_filter),adapt_to_pi=config.data.adapt_to_pi,
                delta_joint_actions=config.data.use_delta_joint_actions,physical_format_version=1)
    return recipe,hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()


def joint_array(handle):
    group = handle["joint_action"]
    return np.concatenate([
        group["left_arm"][:], group["left_gripper"][:].reshape(-1, 1),
        group["right_arm"][:], group["right_gripper"][:].reshape(-1, 1),
    ], axis=1).astype(np.float32)


def make_manifest(seed):
    paths = sorted(DATA_ROOT.glob("*/demo_clean/data/episode0.hdf5"))
    if len(paths) != 18:
        raise ValueError(f"Expected 18 clean tasks, found {len(paths)}")
    rng = np.random.default_rng(seed)
    episodes = []
    for first in paths:
        task = first.parents[2].name
        for episode in range(100):
            path = first.with_name(f"episode{episode}.hdf5")
            instruction_path = path.parents[1] / "instructions" / f"episode{episode}.json"
            prompt = str(rng.choice(json.loads(instruction_path.read_text())["seen"]))
            with h5py.File(path, "r") as handle:
                joints = joint_array(handle)
            episodes.append(dict(task=task, episode=episode, path=str(path),
                                 frames=len(joints)-1, prompt=prompt,
                                 joint_sha256=hashlib.sha256(joints.tobytes()).hexdigest(),
                                 source_bytes=path.stat().st_size))
    result = dict(version=1, seed=seed, episodes=episodes,
                  frames=sum(ep["frames"] for ep in episodes),
                  convention="state[t], action[t+1], 640x480 JPEG roundtrip; episode-clamped horizon")
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


class RawDataset:
    def __init__(self, manifest, horizon=50, images=True):
        self.manifest = manifest
        self.horizon = horizon
        self.images = images
        self.ends = np.cumsum([ep["frames"] for ep in manifest["episodes"]])
        self._handles = OrderedDict()
        self._joints = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        episode = int(np.searchsorted(self.ends, index, side="right"))
        offset = int(index - (self.ends[episode-1] if episode else 0))
        return episode, offset

    def joints(self, episode):
        if episode not in self._joints:
            with h5py.File(self.manifest["episodes"][episode]["path"], "r") as handle:
                self._joints[episode] = joint_array(handle)
        self._joints.move_to_end(episode)
        if len(self._joints) > 64:
            self._joints.popitem(last=False)
        return self._joints[episode]

    def __getitem__(self, index):
        episode, offset = self.locate(int(index))
        ep = self.manifest["episodes"][episode]
        q = self.joints(episode)
        action_ids = np.minimum(offset + 1 + np.arange(self.horizon), len(q)-1)
        images = {}
        if self.images:
            if episode not in self._handles:
                self._handles[episode] = h5py.File(ep["path"], "r")
            handle = self._handles[episode]
            self._handles.move_to_end(episode)
            if len(self._handles) > 16:
                _, old = self._handles.popitem(last=False)
                old.close()
            for destination, camera in CAMERAS.items():
                encoded = handle[f"observation/{camera}/rgb"][offset]
                img = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (640, 480))
                ok, encoded = cv2.imencode(".jpg", img)
                if not ok:
                    raise ValueError(f"JPEG encoding failed: {ep['path']}, frame {offset}")
                images[destination] = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        else:
            images = {camera: np.zeros((224, 224, 3), np.uint8) for camera in CAMERAS}
        return {"observation.state": q[offset], "action": q[action_ids], "prompt": ep["prompt"],
                **{f"observation.images.{camera}":np.transpose(image,(2,0,1)) for camera,image in images.items()}}


class GlobalBatchSampler:
    """Same global frame permutation for every rank and performance variant.

    Each global batch is cut into contiguous rank slices; partial epoch tails are
    dropped globally. Resume computes the epoch permutation without image decoding.
    """
    def __init__(self, size, batch_size, seed, start_step=0, rank=0, ranks=1):
        if batch_size % ranks:
            raise ValueError("Global batch must be divisible by number of processes")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.start_step, self.rank, self.ranks = start_step, rank, ranks
        self.batches_per_epoch = size // batch_size

    def indices(self, step):
        epoch, batch = divmod(step, self.batches_per_epoch)
        if getattr(self, "_epoch", None) != epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self._order = torch.randperm(self.size, generator=generator).numpy()
            self._epoch = epoch
        return self._order[batch*self.batch_size:(batch+1)*self.batch_size]

    def __iter__(self):
        step = self.start_step
        local_size = self.batch_size // self.ranks
        while True:
            indices = self.indices(step)
            yield indices[self.rank*local_size:(self.rank+1)*local_size].tolist()
            step += 1


def worker_init(worker_id):
    # Spawned workers only execute transforms on the CPU, never allocate GPUs.
    os.environ["JAX_PLATFORMS"] = "cpu"
    jax.config.update("jax_platforms", "cpu")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))[:16]
        os.sched_setaffinity(0, {cpus[worker_id % len(cpus)]})
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class Loader:
    def __init__(self, config, sharding, start_step=0, prefetch=2):
        self.config = config
        self._data_config = config.data.create(config.assets_dirs, config.model)
        self.manifest = json.loads(MANIFEST.read_text())
        raw = RawDataset(self.manifest, config.model.action_horizon)
        dataset = data_loader.transform_dataset(raw, self._data_config)
        self.sampler = GlobalBatchSampler(len(raw), config.batch_size, config.seed,
                                         start_step, jax.process_index(), jax.process_count())
        kwargs = {}
        if config.num_workers:
            kwargs.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=prefetch)
        self.torch_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=self.sampler, num_workers=config.num_workers,
            worker_init_fn=worker_init, collate_fn=data_loader._collate_fn, **kwargs,
        )
        self.sharding = sharding

    def data_config(self):
        return self._data_config

    def __iter__(self):
        for batch in self.host_batches():
            yield self.to_device(batch)

    def host_batches(self):
        yield from self.torch_loader

    def to_device(self,batch):
        batch=jax.tree.map(lambda x:jax.make_array_from_process_local_data(self.sharding,x),batch)
        return model_lib.Observation.from_dict(batch),batch["actions"]


def prepare():
    config = get_config()
    ROOT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    else:
        manifest = make_manifest(config.seed)
        atomic_json(MANIFEST, manifest)
    stats_dir = config.assets_dirs / ASSET_ID
    if (stats_dir / "norm_stats.json").exists():
        return manifest
    raw = RawDataset(manifest, config.model.action_horizon, images=False)
    data = config.data.create(config.assets_dirs, config.model)
    transform = transforms.compose([*data.repack_transforms.inputs, *data.data_transforms.inputs])
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    # Same ordered, drop_last, global64 batching as official compute_norm_stats;
    # omitting image decode changes neither state nor horizon50 delta chunks.
    for start in range(0, len(raw)-config.batch_size+1, config.batch_size):
        samples = [transform(raw[i]) for i in range(start, start+config.batch_size)]
        for key, running in stats.items():
            running.update(np.stack([item[key] for item in samples]))
        if start % 64_000 == 0:
            print(f"norm stats: {start}/{len(raw)} frames", flush=True)
    result={key: running.get_statistics() for key, running in stats.items()}
    atomic_json(stats_dir/"norm_stats.json",json.loads(normalize.serialize_json(result)))
    return manifest
