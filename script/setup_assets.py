"""Download RoboTwin assets, then overlay the BiCoord object modifications."""
import os
from pathlib import Path
import time
import zipfile

from huggingface_hub import hf_hub_download

repo = Path(__file__).resolve().parent.parent
assets = repo / "assets"
endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
for dataset, filenames in [
    ("TianxingChen/RoboTwin2.0", ["embodiments.zip", "objects.zip", "background_texture.zip"]),
    ("GradiusTwinbee/BiCoord", ["objects.zip"]),
]:
    for filename in filenames:
        archive_dir = assets / ".archives" / dataset.split("/")[1]
        marker = archive_dir / (filename + ".extracted")
        if marker.exists():
            print(f"Already extracted: {dataset}/{filename}", flush=True)
            continue
        for attempt in range(3):
            try:
                archive = hf_hub_download(dataset, filename, repo_type="dataset", local_dir=archive_dir, endpoint=endpoint, token=False)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            for name in names:
                if not (assets / name).resolve().is_relative_to(assets.resolve()):
                    raise ValueError(f"Invalid archive path: {name}")
            # BiCoord's archive contains the objects directory's contents.
            destination = assets
            if dataset.endswith("BiCoord") and not any(name.startswith("objects/") for name in names):
                destination = assets / "objects"
            destination.mkdir(parents=True, exist_ok=True)
            zipped.extractall(destination)
        marker.write_text(str(time.time()) + "\n")
        print(f"Extracted {dataset}/{filename} -> {destination}", flush=True)

for config in (assets / "embodiments").glob("**/*.yml"):
    content = config.read_text()
    if "${ASSETS_PATH}" in content:
        config.write_text(content.replace("${ASSETS_PATH}", str(repo)))
print("ASSET SETUP COMPLETE", flush=True)
