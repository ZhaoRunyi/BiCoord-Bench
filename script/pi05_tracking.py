"""Backfill recorded training metrics and track the existing PBS evaluation.

No GPU computation. Unrecorded gradients, intermediate losses and timestamps
are not reconstructed. LR is derived from the saved optimizer schedule.
"""
import argparse
import json
import netrc
import os
from pathlib import Path
import subprocess
import time
import uuid

import wandb
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent / "ckpts/BiCoord_pi05"
DEFAULT_RUN = Path("/tmp/bicoord-pi05-training-20261011")
NETRC = REPO.parent.parent / ".wandb/netrc"


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def credential_available():
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        auth = netrc.netrc(os.environ.get("NETRC", str(Path.home()/".netrc"))).authenticators("api.wandb.ai")
        return bool(auth and auth[2])
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def remote_progress(host):
    code = """import json
from pathlib import Path
root=Path('/tmp/bicoord-pi05-training-20261011/evaluation')
rows=[]
for path in sorted(root.glob('*/episodes.jsonl')):
    # A writer may still be appending the final line.
    records=[]
    for line in path.read_text().splitlines():
        try: records.append(json.loads(line))
        except json.JSONDecodeError: pass
    rows.append(dict(task=path.parent.name,episodes=len(records),successes=sum(r['success'] for r in records),stage_sum=sum(r['stage_score'] for r in records)))
final=root/'metrics.json'
print(json.dumps(dict(tasks=rows,final=json.loads(final.read_text()) if final.exists() else None)))
"""
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                             str(REPO/"policy/pi05/.venv/bin/python"), "-"], input=code,
                            text=True, capture_output=True, check=True, timeout=45)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--host", default="recgpu116")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args()
    os.environ.setdefault("NETRC", str(NETRC))
    os.environ.setdefault("WANDB_BASE_URL", "https://api.wandb.ai")
    output = args.run_dir / "tracking"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    run_id = state.get("run_id", uuid.uuid4().hex[:8])
    state.update(run_id=run_id, project="openpi_bicoord", tensorboard=str(output/"tensorboard"))
    writer = EventFileWriter(str(output/"tensorboard"))

    def scalar(values, step):
        writer.add_event(Event(wall_time=time.time(), step=step,
                         summary=Summary(value=[Summary.Value(tag=k, simple_value=float(v)) for k,v in values.items()])))
        writer.flush()

    recipe = json.loads((ROOT/"checkpoints/50000/assets/recipe.json").read_text())
    records = [json.loads(line) for line in (args.run_dir/"train-metrics.jsonl").read_text().splitlines()]
    timings = {}
    for line in (args.run_dir/"train-runner.log").read_text().splitlines():
        try:
            row = json.loads(line)
            if "step" in row and "seconds" in row:
                timings[row["step"]] = row["seconds"]
        except json.JSONDecodeError:
            pass
    if len(records) != 501 or records[-1]["step"] != 50000:
        raise ValueError("Expected the complete 501-point training log through step 50,000")
    online = credential_available()
    run = None
    if not state.get("training_exported"):
        run = wandb.init(project=state["project"], entity=args.entity, id=run_id,
                         name="pi05-bicoord-all-bs64-50k", mode="online" if online else "offline",
                         dir=str(output), config=dict(recipe=recipe, training_steps=50000, devices=8,
                         backfilled=True, training_log_points=len(records),
                         observation_notes="loss and step_seconds: recorded every 100 updates plus step 1; learning_rate: derived from saved schedule at optimizer step completed_steps-1; original per-point wall times and unrecorded gradients unavailable"),
                         settings=wandb.Settings(x_disable_stats=True, console="off", disable_git=True))
        for row in records:
            completed = row["step"]
            schedule = recipe["lr_schedule"]
            warmup = schedule["warmup_steps"]
            initial = schedule["peak_lr"] / (warmup + 1)
            if schedule["peak_lr"] != schedule["decay_lr"]:
                raise ValueError("This backfill supports the recorded constant-after-warmup schedule only")
            lr = initial + (schedule["peak_lr"]-initial)*min((completed-1)/warmup, 1)
            values = {"train/loss": row["loss"], "train/learning_rate_derived": lr}
            if completed in timings:
                values["train/step_seconds"] = timings[completed]
            scalar(values, completed)
            run.log(values, step=completed)
        run.summary.update(dict(completed_steps=50000, final_loss=records[-1]["loss"],
                           last_100_logged_loss_mean=sum(r["loss"] for r in records[-100:])/100,
                           checkpoint_path=str(ROOT/"checkpoints/50000")))
        state.update(training_exported=True, offline_dir=str(Path(run.dir).parent), uploaded=online)
        if online:
            state["url"] = run.url
        run.finish()
        run = None
        atomic_json(state_path, state)
    previous = {}
    deadline = time.monotonic() + 72*3600
    while True:
        online = credential_available()
        if online and not state.get("uploaded"):
            cmd = [str(REPO/"policy/pi05/.venv/bin/wandb"), "sync", "--project", state["project"]]
            if args.entity:
                cmd += ["--entity", args.entity]
            subprocess.run(cmd+[state["offline_dir"]], check=True, timeout=180)
            state["uploaded"] = True
        if online and run is None:
            run = wandb.init(project=state["project"], entity=args.entity, id=run_id, resume="must",
                             mode="online", dir=str(output),
                             settings=wandb.Settings(x_disable_stats=True, console="off", disable_git=True))
            state["url"] = run.url
            video = args.run_dir / "evaluation/balance_roller/episode0.mp4"
            if video.exists() and not state.get("video_uploaded"):
                run.log({"eval/balance_roller_video": wandb.Video(str(video), format="mp4")})
                state["video_uploaded"] = True
        progress = remote_progress(args.host)
        atomic_json(output/"eval-progress.json", progress)
        total = sum(r["episodes"] for r in progress["tasks"])
        values = {"eval/episodes_completed": total}
        for row in progress["tasks"]:
            if row["episodes"]:
                prefix = "eval_task/" + row["task"]
                values.update({prefix+"/episodes": row["episodes"],
                               prefix+"/sr_partial": 100*row["successes"]/row["episodes"],
                               prefix+"/ssr_partial": 100*row["stage_sum"]/row["episodes"]})
        if values != previous:
            scalar(values, 50000+total)
            if run:
                run.log(values)
            previous = values
        state.update(eval_episodes_completed=total, updated_unix=time.time(),
                     state="watching" if args.watch else "exported",
                     waiting_for_wandb_login=not state.get("uploaded"))
        final = progress["final"]
        if final:
            atomic_json(output/"eval-final.json", final)
            values = {"eval/sr": final["sr"], "eval/ssr": final["ssr"]}
            if final["tl"] is not None:
                values["eval/tl"] = final["tl"]
            if not state.get("final_tensorboard_written"):
                scalar(values, 51800)
                state["final_tensorboard_written"] = True
            if run:
                run.log(values)
                run.summary.update(values)
                columns = ["task", "episodes", "successes", "sr", "ssr", "tl"]
                run.log({"eval/task_results": wandb.Table(columns=columns,
                         data=[[row[key] for key in columns] for row in final["tasks"]])})
            state["state"] = "complete" if state.get("uploaded") else "evaluation_complete_waiting_login"
        atomic_json(state_path, state)
        if not args.watch or (final and state.get("uploaded")) or time.monotonic() >= deadline:
            break
        time.sleep(60)
    if run:
        run.finish()
    writer.close()


if __name__ == "__main__":
    main()
