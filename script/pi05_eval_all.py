"""Run the official 18x100 protocol within a full-host PBS allocation."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading

REPO=Path(__file__).resolve().parents[1]
WS=REPO.parent
OUT=Path("/tmp/bicoord-pi05-training-20261011/evaluation")
tokens=list(range(8))
lock=threading.Lock()


def evaluate(task):
    with lock:
        gpu=tokens.pop()
    env=os.environ.copy()
    masks=env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    env.update(CUDA_VISIBLE_DEVICES=masks[gpu],JAX_PLATFORMS="cuda,cpu",XLA_PYTHON_CLIENT_PREALLOCATE="false",
               OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",OIDN_NUM_THREADS="2")
    dest=OUT/task
    dest.mkdir(parents=True,exist_ok=True)
    cpus=sorted(os.sched_getaffinity(0))[:16]
    assigned=cpus[gpu*2:gpu*2+2]
    cmd=["taskset","-c",",".join(map(str,assigned)),"/bin/bash",str(REPO/"script/run_uv.sh"),"pi05","python","script/eval_policy.py",
         "--config","policy/pi05/deploy_policy.yml","--overrides","--task_name",task,
         "--task_config","demo_clean","--policy_name","pi05_bicoord","--ckpt_setting","pi05_all_50000",
         "--seed","0","--test_num","100","--video_episodes","1" if task=="balance_roller" else "0",
         "--resume","True","--output_dir",str(dest)]
    try:
        with (dest/"runner.log").open("a") as stream:
            subprocess.run(cmd,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        records=[json.loads(line) for line in (dest/"episodes.jsonl").read_text().splitlines()]
        if len(records)!=100:
            raise RuntimeError(f"{task}: only {len(records)} episodes")
        good=[item for item in records if item["success"]]
        return dict(task=task,episodes=len(records),successes=len(good),sr=100*len(good)/len(records),
                    ssr=100*sum(item["stage_score"] for item in records)/len(records),
                    tl=sum(item["steps"] for item in good)/len(good) if good else None)
    finally:
        with lock:
            tokens.append(gpu)


def main():
    subprocess.run([str(REPO/"policy/pi05/.venv/bin/python"),str(REPO/"script/pi05_allocation.py")],check=True)
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((WS/"ckpts/BiCoord_pi05/data_manifest.json").read_text())
    tasks=sorted({ep["task"] for ep in manifest["episodes"]})
    with ThreadPoolExecutor(max_workers=8) as executor:
        rows=list(executor.map(evaluate,tasks))
    lengths=[row["tl"] for row in rows if row["tl"] is not None]
    result=dict(state="complete",tasks=rows,sr=sum(row["sr"] for row in rows)/18,
                ssr=sum(row["ssr"] for row in rows)/18,
                tl=sum(lengths)/len(lengths) if lengths else None,
                cogwam_pi05=dict(sr=35.6,ssr=50.8),protocol="18 tasks x 100 rollouts; original success predicates")
    (OUT/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    result["sr_delta_pp"]=result["sr"]-35.6
    result["ssr_delta_pp"]=result["ssr"]-50.8
    (OUT/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
