"""Check simulator imports, native CUDA ops, and policy configuration imports."""
import importlib
import os
from pathlib import Path
import runpy
import sys
import tempfile

import torch

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
for name in ["sapien", "mplib", "toppra", "open3d", "cv2", "h5py", "transforms3d",
             "curobo.wrap.reacher.motion_gen", "pytorch3d.ops", "envs.balance_roller"]:
    importlib.import_module(name)
    print(f"Import OK: {name}")
print(f"torch={torch.__version__}, CUDA runtime={torch.version.cuda}, CUDA available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    from pytorch3d.ops import sample_farthest_points
    points = torch.rand(1, 32, 3, device="cuda")
    sampled, indices = sample_farthest_points(points, K=8)
    assert sampled.shape == (1, 8, 3)
    torch.cuda.synchronize()
    print("PyTorch3D CUDA sampling OK")
if sys.argv[1] in ["pi0", "pi05"]:
    import jax
    from openpi.training import config
    from openpi.training import data_loader
    import av
    import numpy as np
    from lerobot.common.datasets.video_utils import decode_video_frames

    with tempfile.TemporaryDirectory(prefix="bicoord-uv-") as temporary:
        clip = Path(temporary) / "clip.mp4"
        with av.open(str(clip), "w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for value in [0, 128]:
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), value, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        frames = decode_video_frames(clip, [0.0, 0.1], 0.02, backend="pyav")
        assert frames.shape == (2, 3, 32, 32)
        assert frames[1].mean() > frames[0].mean()
        print("LeRobot PyAV video decoding OK")
    name = "pi05_aloha_full_base" if sys.argv[1] == "pi05" else "pi0_base_aloha_robotwin_full"
    cfg = config.get_config(name)
    runpy.run_path(str(repo_root / "policy" / sys.argv[1] / "scripts/train.py"), run_name="bicoord_import_check")
    assert float(jax.numpy.sum(jax.numpy.ones(8))) == 8.0
    print(f"Policy config OK: {cfg.name}; JAX devices: {jax.devices()}")
