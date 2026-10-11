"""Apply the two simulator dependency adjustments required by RoboTwin."""
import importlib.util
from pathlib import Path


def patch(module, relative, replacements):
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Missing simulator dependency: {module}")
    path = Path(spec.origin).parent / relative
    original = path.read_text()
    changed = original
    for old, new in replacements:
        if old in changed:
            changed = changed.replace(old, new)
        elif new not in changed:
            raise RuntimeError(f"Unsupported {module} version: expected patch target absent in {path}")
    if changed != original:
        # uv can hardlink wheel files; replace the inode to leave its cache intact.
        temporary = path.with_suffix(path.suffix + ".bicoord-tmp")
        temporary.write_text(changed)
        temporary.replace(path)
        print(f"Patched {path}")
    else:
        print(f"Already patched: {path}")


patch("sapien", "wrapper/urdf_loader.py", [
    ('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
    ('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")'),
])
patch("mplib", "planner.py", [
    ("if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
     "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"),
])
