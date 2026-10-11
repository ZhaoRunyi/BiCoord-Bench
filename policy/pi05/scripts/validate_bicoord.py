"""CPU integration checks against the actual upstream pi0 converter."""
import importlib.util
import json
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

from openpi.training import bicoord as bc
from openpi.training import config
from openpi import transforms


def main():
    cfg = bc.get_config()
    assert cfg.optimizer == config.TrainConfig(name="defaults").optimizer
    schedule = cfg.lr_schedule.create()
    np.testing.assert_allclose([schedule(step) for step in (10_000,30_000,50_000)],5e-5,rtol=1e-6)
    assert cfg.ema_decay == .999
    assert cfg.seed == config.TrainConfig(name="defaults").seed
    assert cfg.model.action_horizon == 50 and cfg.model.action_dim == 32
    assert cfg.model.discrete_state_input and not cfg.data.adapt_to_pi
    factory_data = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert factory_data.use_quantile_norm
    manifest = json.loads(bc.MANIFEST.read_text())
    assert len(manifest["episodes"]) == 1800
    raw = bc.RawDataset(manifest)
    first = manifest["episodes"][0]
    source = Path(first["path"]).parents[1]
    converter_path = bc.WS / "BiCoord-Bench/policy/pi0/scripts/process_data.py"
    spec = importlib.util.spec_from_file_location("official_pi0_converter", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    with tempfile.TemporaryDirectory(prefix="bicoord-data-check-", dir="/tmp") as out:
        converter.data_transform(str(source), 1, out)
        with h5py.File(Path(out) / "episode_0/episode_0.hdf5", "r") as expected:
            count = len(expected["action"])
            assert count == first["frames"]
            for frame in (0, 1, count-1):
                actual = raw[frame]
                np.testing.assert_array_equal(actual["observation.state"], expected["observations/qpos"][frame])
                indices = np.minimum(frame+np.arange(50), count-1)
                wanted_actions = expected["action"][:][indices]
                np.testing.assert_array_equal(actual["action"], wanted_actions)
                for camera in bc.CAMERAS:
                    wanted = cv2.imdecode(np.frombuffer(expected[f"observations/images/{camera}"][frame],np.uint8),cv2.IMREAD_COLOR)
                    np.testing.assert_array_equal(np.transpose(actual[f"observation.images.{camera}"],(1,2,0)), wanted)
                physical = transforms.compose([*factory_data.repack_transforms.inputs,
                                               *factory_data.data_transforms.inputs])(actual)
                mask = np.array([True]*6+[False]+[True]*6+[False])
                wanted = wanted_actions.copy()
                wanted[:,mask] -= actual["observation.state"][mask]
                np.testing.assert_array_equal(physical["actions"], wanted)
                np.testing.assert_array_equal(physical["state"], actual["observation.state"])
    for step in (0,1,4,5,6,1999,2000,9999,10000,29999,30000):
        whole = bc.GlobalBatchSampler(321,64,cfg.seed).indices(step)
        ranks = [bc.GlobalBatchSampler(321,64,cfg.seed,start_step=step,rank=i,ranks=2) for i in range(2)]
        joined = np.concatenate([next(iter(rank)) for rank in ranks])
        np.testing.assert_array_equal(whole,joined)
        np.testing.assert_array_equal(whole,next(iter(bc.GlobalBatchSampler(321,64,cfg.seed,start_step=step))))
    epoch = np.concatenate([bc.GlobalBatchSampler(321,64,cfg.seed).indices(i) for i in range(5)])
    assert len(np.unique(epoch)) == 320
    print(json.dumps(dict(passed=True, frames=len(raw), checks=["official pi0 state/action/JPEG pixel equality",
          "episode tail clamp", "delta/gripper convention", "global 8/16 sampler and resume",
          "official optimizer recipe inherited", "pi05 representation retained"])),flush=True)


if __name__ == "__main__":
    main()
