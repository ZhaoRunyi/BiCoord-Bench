"""Link the sibling data/checkpoint layout without copying large artifacts."""
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
workspace = repo.parent
links = {
    repo / "data": workspace / "data/BiCoord",
    repo / "ckpts": workspace / "ckpts",
    repo / "policy/pi0/checkpoints": workspace / "ckpts/BiCoord_pi0",
    repo / "policy/pi05/checkpoints": workspace / "ckpts/BiCoord_pi05",
}
for link, target in links.items():
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == target.resolve():
        print(f"Already linked: {link}")
    elif link.exists() or link.is_symlink():
        raise RuntimeError(f"Refusing to replace existing path: {link}")
    else:
        import os
        link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        print(f"Linked {link} -> {target}")
