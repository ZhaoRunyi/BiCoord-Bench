"""Run bounded benchmark variants, or the selected production recipe, inside PBS."""
import argparse
import json
import os
import shutil
import shlex
import signal
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
RUN = Path("/tmp/bicoord-pi05-training-20261011")
# Each variant changes execution only; all share model, global data/RNG and optimizer.
VARIANTS = [
    dict(devices=8,fsdp=8,workers=2,prefetch=2,device_prefetch=0,lean_metrics=False),
    dict(devices=8,fsdp=4,workers=2,prefetch=2,device_prefetch=0,lean_metrics=False),
    dict(devices=8,fsdp=8,workers=4,prefetch=2,device_prefetch=0,lean_metrics=False),
    dict(devices=8,fsdp=8,workers=8,prefetch=2,device_prefetch=0,lean_metrics=False),
    dict(devices=8,fsdp=8,workers=12,prefetch=4,device_prefetch=0,lean_metrics=False),
    dict(devices=8,fsdp=8,workers=8,prefetch=4,device_prefetch=2,lean_metrics=True),
    dict(devices=16,fsdp=8,workers=6,prefetch=4,device_prefetch=2,lean_metrics=True),
    dict(devices=16,fsdp=16,workers=6,prefetch=4,device_prefetch=2,lean_metrics=True),
    dict(devices=16,fsdp=8,workers=6,prefetch=4,device_prefetch=2,lean_metrics=True,
         xla_flags="--xla_gpu_enable_latency_hiding_scheduler=true"),
    dict(devices=16,fsdp=16,workers=6,prefetch=4,device_prefetch=2,lean_metrics=True,
         xla_flags="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_all_gather_combine_threshold_bytes=67108864 --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 --xla_gpu_all_reduce_combine_threshold_bytes=67108864"),
    dict(devices=16,fsdp=16,workers=6,prefetch=4,device_prefetch=2,lean_metrics=True,
         xla_flags="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_triton_gemm_any=true"),
]


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    os.replace(tmp,path)


def command(variant,mode,output,reference,export=False):
    argv=["/bin/bash",str(REPO/"script/pi05_node.sh"),"--mode",mode,"--output",str(output),
          "--reference",str(reference),"--fsdp",str(variant["fsdp"]),"--workers",str(variant["workers"]),
          "--prefetch",str(variant["prefetch"]),"--device-prefetch",str(variant["device_prefetch"])]
    if variant["lean_metrics"]:
        argv.append("--lean-metrics")
    if export:
        argv.append("--export-reference")
    if variant.get("xla_flags"):
        argv.append("--xla-flags="+variant["xla_flags"])
    return argv


def execute(variant,mode,output,reference,hosts,export=False):
    output.mkdir(parents=True,exist_ok=True)
    argv=command(variant,mode,output,reference,export)
    processes=[]
    if variant["devices"]==16:
        coordinator=f"{hosts[0]}:{14500+os.getpid()%1000}"
        remote=["/opt/pbs/bin/pbsdsh","-n","1","--",*argv,"--ranks","2","--rank","1","--coordinator",coordinator]
        with (output/"rank1-launch.log").open("w") as stream:
            processes.append(subprocess.Popen(remote,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        argv += ["--ranks","2","--rank","0","--coordinator",coordinator]
    with (output/"runner.log").open("w") as stream:
        processes.insert(0,subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
    deadline=time.monotonic()+(1200 if mode=="benchmark" else 71*3600)
    try:
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError("A rank exited with an error; stopping its peers")
            if variant["devices"]==16 and processes[1].poll()==0:
                marker=subprocess.run(["ssh","-o","BatchMode=yes",hosts[1],
                                       "cat "+shlex.quote(str(output/"rank-1-exit.txt"))],
                                      capture_output=True,text=True,timeout=30)
                if marker.returncode:
                    raise RuntimeError("Remote TM task exited without a rank completion marker")
                job,code=marker.stdout.split()
                if job!=os.environ["PBS_JOBID"] or int(code)!=0:
                    raise RuntimeError(f"Remote rank exited with code {code}")
            if time.monotonic()>deadline:
                raise TimeoutError("Bounded benchmark/training deadline exceeded")
            time.sleep(5)
        if any(p.returncode for p in processes):
            raise RuntimeError(f"Rank exit codes: {[p.returncode for p in processes]}")
        if variant["devices"]==16:
            marker=subprocess.run(["ssh","-o","BatchMode=yes",hosts[1],
                                   "cat "+shlex.quote(str(output/"rank-1-exit.txt"))],
                                  capture_output=True,text=True,timeout=30,check=True)
            job,code=marker.stdout.split()
            if job!=os.environ["PBS_JOBID"] or int(code)!=0:
                raise RuntimeError(f"Remote rank completion failed: {marker.stdout.strip()}")
    finally:
        interrupted=any(p.poll() is None for p in processes)
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL)
                p.wait(timeout=30)
        if interrupted and variant["devices"]==16:
            subprocess.run(["ssh","-o","BatchMode=yes",hosts[1],shlex.join([
                str(REPO/"policy/pi05/.venv/bin/python"),str(REPO/"script/pi05_stop_rank.py"),
                os.environ["PBS_JOBID"],str(output)])],check=True,timeout=45)
    return json.loads((output/"result.json").read_text()) if mode=="benchmark" else None


def simulation_smoke():
    marker=RUN/"inference-smoke.json"
    if marker.exists() and json.loads(marker.read_text()).get("passed"):
        return
    subprocess.run([str(REPO/"policy/pi05/.venv/bin/python"),str(REPO/"script/pi05_allocation.py")],check=True)
    # Exercise the complete pi05 + simulation stack before the long run.
    smoke=RUN/"inference-smoke"
    smoke.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=env.get("CUDA_VISIBLE_DEVICES","0,1,2,3,4,5,6,7").split(",")[0],
               JAX_PLATFORMS="cuda,cpu",XLA_PYTHON_CLIENT_PREALLOCATE="false",
               OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    argv=["/bin/bash",str(REPO/"script/run_uv.sh"),"pi05","python","script/eval_policy.py",
          "--config","policy/pi05/deploy_policy.yml","--overrides","--task_name","balance_roller",
          "--task_config","demo_clean","--policy_name","pi05_bicoord","--ckpt_setting","pi05_base_smoke",
          "--checkpoint_dir",str(REPO.parent/"ckpts/pi05_base"),"--seed","0","--test_num","1",
          "--video_episodes","0","--resume","True","--output_dir",str(smoke)]
    with (smoke/"runner.log").open("w") as stream:
        subprocess.run(argv,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1800)
    if len((smoke/"episodes.jsonl").read_text().splitlines())!=1:
        raise RuntimeError("Pre-training inference smoke test did not produce one rollout")
    write(RUN/"inference-smoke.json",dict(passed=True,checkpoint="pi05_base",task="balance_roller"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["benchmark","train"])
    parser.add_argument("--selection")
    parser.add_argument("--indices",help="Retry specified original variant indices after an execution repair")
    parser.add_argument("--rebuild-reference",action="store_true")
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    hosts=list(dict.fromkeys(Path(os.environ["PBS_NODEFILE"]).read_text().split()))
    if args.mode=="train":
        selection=json.loads(Path(args.selection).read_text())
        reference=RUN/"reference"
        if (reference/"profile.json").exists():
            write(RUN/"reference-profile.json",json.loads((reference/"profile.json").read_text()))
        if reference.exists():
            shutil.rmtree(reference)
        retained=RUN/"reference-retained"
        if retained.exists():
            shutil.rmtree(retained)
        simulation_smoke()
        execute(selection["variant"],"train",RUN/"train",RUN/"reference",hosts)
        write(RUN/"job-complete.json",dict(state="trained",job=os.environ["PBS_JOBID"]))
        return
    retry=set(map(int,args.indices.split(","))) if args.indices else None
    results=json.loads((RUN/"benchmarks.json").read_text()) if retry else []
    reference=RUN/"reference"
    retained=RUN/"reference-retained"
    if retry and not reference.exists() and retained.exists():
        retained.rename(reference)
    if args.rebuild_reference:
        simulation_smoke()
        for directory in (reference,retained):
            if directory.exists():
                shutil.rmtree(directory)
        # Recreate the known-good single-host baseline, not a self-reference on
        # 16 cards. Keep earlier multihost speed records but exclude them from
        # selection until checked against this independently validated baseline.
        for entry in results:
            if entry["variant"]["devices"]==16:
                entry["passed"]=False
                entry["selection_exclusion"]="Earlier multihost reference was not compared directly to 8-card state"
        execute(VARIANTS[5],"benchmark",RUN/"reference-rebuild",reference,hosts,export=True)
    save_seconds=None
    for index,variant in enumerate(VARIANTS):
        if retry and index+1 not in retry:
            continue
        output=RUN/(f"variant-{index+1}-retry" if retry else f"variant-{index+1}")
        results=[entry for entry in results if entry["index"]!=index+1]
        try:
            export=not (reference/"profile.json").exists()
            result=execute(variant,"benchmark",output,reference,hosts,export=export)
            previous=next((entry for entry in results if entry.get("passed")),None)
            if previous is not None:
                for key in ("manifest_sha256","first_batches_sha256"):
                    if result[key]!=previous[key]:
                        raise RuntimeError(f"Benchmark changed global data sequence: {key}")
            if export:
                save_seconds=result["save_seconds"]
            else:
                save_seconds=json.loads((reference/"profile.json").read_text())["save_seconds"]
            if save_seconds is None:
                raise RuntimeError("Reference checkpoint timing unavailable")
            # Include startup/compilation, all 25 full saves, p95/mean timing conservatism,
            # and 10% margin. Evaluation is reported separately from training ETA.
            seconds=max(result["mean_seconds"],result["p95_seconds"])*50_000
            seconds += 25*save_seconds+result["setup_seconds"]
            result.update(index=index+1,variant=variant,estimated_train_hours=1.1*seconds/3600,passed=True)
            results.append(result)
            write(RUN/"benchmarks.json",results)
            print(json.dumps(result),flush=True)
            if result["estimated_train_hours"]<16:
                break
        except Exception as error:
            results.append(dict(index=index+1,variant=variant,passed=False,error=repr(error)))
            write(RUN/"benchmarks.json",results)
            print(json.dumps(results[-1]),flush=True)
    valid=[result for result in results if result["passed"]]
    if not valid:
        raise RuntimeError("No correctly validated performance variant; training is blocked, not silently altered")
    selected=min(valid,key=lambda result:result["estimated_train_hours"])
    write(RUN/"selection.json",selected)
    write(RUN/"reference-profile.json",json.loads((reference/"profile.json").read_text()))
    shutil.rmtree(reference)


if __name__=="__main__":
    main()
