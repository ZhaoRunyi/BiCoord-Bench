"""Detached submit-host controller: benchmark -> validated train -> official evaluation.

GPU computation runs only in PBS. Each qsub is issued once; ambiguous submissions
are never retried. Questions/errors are retained for the AFK user in /tmp.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
WS=REPO.parent
RUN=Path("/tmp/bicoord-pi05-training-20261011")
REMOTE=RUN
HOSTS=["recgpu116","recgpu118"]
PYTHON=str(REPO/"policy/pi05/.venv/bin/python")


def write(name, value):
    path=RUN/name
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    os.replace(tmp,path)


def status(state,**kwargs):
    write("status.json",dict(state=state,updated_unix=time.time(),**kwargs))
    print(json.dumps(dict(state=state,**kwargs)),flush=True)


def fetch(host,path,destination):
    result=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15",host,
                           "cat "+shlex.quote(str(path))],capture_output=True,timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    destination.write_bytes(result.stdout)


def submit(name,devices,command,walltime):
    prior=RUN/f"{name}-submission.json"
    if prior.exists():
        return json.loads(prior.read_text())["job_id"]
    # Recheck PBS and physical GPU state before submitting the authorized job.
    gpuctl=WS.parent/".codex/skills/pbs-gpu/scripts/pbs-gpu"
    deadline=time.monotonic()+6*3600
    while True:
        snapshot=subprocess.run([str(gpuctl),"gpus","--hosts",",".join(HOSTS[:devices//8]),
                                 "--processes","--json"],capture_output=True,text=True,timeout=90,check=True)
        record=json.loads(snapshot.stdout)
        write(f"{name}-resources.json",record)
        if all(len(record["hosts"][host]["gpus"])==8 and
               all(gpu["status"]=="idle-snapshot" for gpu in record["hosts"][host]["gpus"])
               for host in HOSTS[:devices//8]):
            break
        if time.monotonic()>deadline:
            raise TimeoutError("Selected full-host resources have not become available within 6 hours")
        status("waiting_resources",next_job=name,hosts=HOSTS[:devices//8])
        time.sleep(45)
    chunks="+".join(f"1:ncpus=16:ngpus=8:mpiprocs=1:host={host}" for host in HOSTS[:devices//8])
    script=RUN/f"{name}.pbs"
    script.write_text("#!/bin/bash\n"+f"#PBS -N {name}\n#PBS -q research\n#PBS -l select={chunks}\n"
          +f"#PBS -l place=scatter:excl\n#PBS -l walltime={walltime}\n#PBS -j oe\n"
          +f"#PBS -o {RUN/name}.log\nset -euo pipefail\ncd {shlex.quote(str(REPO))}\n"
          +f"trap 'printf \"%s\\n\" \"$?\" > {REMOTE}/job-$PBS_JOBID.exit' EXIT\n"
          +f"mkdir -p {REMOTE}\n"+shlex.join(command)+"\n")
    result=subprocess.run(["/opt/pbs/bin/qsub",str(script)],capture_output=True,text=True,timeout=60)
    if result.returncode:
        raise RuntimeError(f"qsub rejected {name}: {result.stderr}")
    job=result.stdout.strip()
    write(f"{name}-submission.json",dict(job_id=job,script=str(script),devices=devices,hosts=HOSTS[:devices//8]))
    return job


def wait_job(job,phase):
    while True:
        result=subprocess.run(["/opt/pbs/bin/qstat","-f","-F","json",job],capture_output=True,text=True,timeout=30)
        info=json.loads(result.stdout or "{}").get("Jobs",{}).get(job)
        if info is None:
            marker=RUN/f"job-{job}.exit"
            try:
                fetch(HOSTS[0],REMOTE/marker.name,marker)
            except RuntimeError:
                # The first benchmark predates the EXIT trap. Its validated
                # selection is written only after successful driver completion.
                if phase=="benchmarking":
                    fetch(HOSTS[0],REMOTE/"selection.json",RUN/"selection.json")
                    return
                raise RuntimeError(f"PBS job {job} disappeared without completion evidence: {result.stderr}")
            if int(marker.read_text().strip())!=0:
                raise RuntimeError(f"{phase} job {job} failed with exit {marker.read_text().strip()}")
            return
        write(f"{phase}-job.json",info)
        progress={}
        if phase=="benchmarking":
            try:
                fetch(HOSTS[0],REMOTE/"benchmarks.json",RUN/"benchmarks.json")
                rows=json.loads((RUN/"benchmarks.json").read_text())
                progress["finished_variants"]=len(rows)
            except RuntimeError:
                pass
        elif phase=="training":
            try:
                fetch(HOSTS[0],REMOTE/"train/metrics.jsonl",RUN/"train-metrics.jsonl")
                last=json.loads((RUN/"train-metrics.jsonl").read_text().splitlines()[-1])
                progress.update(last_step=last["step"],last_loss=last["loss"])
            except (RuntimeError,IndexError):
                pass
        status(phase,job_id=job,pbs_state=info["job_state"],comment=info.get("comment"),**progress)
        if info["job_state"]=="F":
            if info.get("Exit_status")!=0:
                raise RuntimeError(f"{phase} job {job} failed: Exit_status={info.get('Exit_status')}")
            return
        time.sleep(45)


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    questions=RUN/"pending_questions.json"
    if not questions.exists():
        write("pending_questions.json",[])
    norm=WS/"ckpts/BiCoord_pi05/assets/pi05_bicoord_all/bicoord_all_clean/norm_stats.json"
    status("preparing")
    deadline=time.monotonic()+7200
    while not norm.exists():
        if time.monotonic()>deadline:
            raise TimeoutError("Normalization preparation has not completed within 2 hours")
        time.sleep(20)
    cpu_log=(RUN/"cpu-validation.log").read_text()
    if '"passed": true' not in cpu_log:
        raise RuntimeError("CPU integration validation did not pass")
    bench=submit("pi05bench",16,[PYTHON,str(REPO/"script/pi05_job_driver.py"),"benchmark"],"12:00:00")
    wait_job(bench,"benchmarking")
    # Retry the two original cross-host variants after fixing compiler KVS waits.
    repair=submit("pi05repair",16,[PYTHON,str(REPO/"script/pi05_job_driver.py"),
                                  "benchmark","--indices","7,8"],"12:00:00")
    wait_job(repair,"benchmark_repair")
    fetch(HOSTS[0],REMOTE/"selection.json",RUN/"selection.json")
    additional=submit("pi05tune",16,[PYTHON,str(REPO/"script/pi05_job_driver.py"),
                                    "benchmark","--indices","9,10,11","--rebuild-reference"],"12:00:00")
    wait_job(additional,"benchmark_tuning")
    for name in ("selection.json","benchmarks.json"):
        fetch(HOSTS[0],REMOTE/name,RUN/name)
    selection=json.loads((RUN/"selection.json").read_text())
    if not selection.get("passed") or not selection["numerical"]["passed"]:
        raise RuntimeError("Selected execution variant did not pass numerical validation")
    if '"passed": true' not in (RUN/"loader-smoke.log").read_text():
        raise RuntimeError("Full spawned-worker input pipeline validation did not pass")
    if '"passed": true' not in (RUN/"restore-layout-smoke.log").read_text():
        raise RuntimeError("Explicit current-layout checkpoint restoration validation did not pass")
    benchmarks=json.loads((RUN/"benchmarks.json").read_text())
    passed=[entry for entry in benchmarks if entry.get("passed")]
    if any(entry["manifest_sha256"]!=selection["manifest_sha256"]
           or entry["first_batches_sha256"]!=selection["first_batches_sha256"] for entry in passed):
        raise RuntimeError("Benchmark variants changed global data order")
    if selection["estimated_train_hours"]>=16:
        saved=json.loads(questions.read_text())
        saved.append(dict(kind="runtime_target_not_met",estimated_train_hours=selection["estimated_train_hours"],
                          needs_review=False,decision="Use fastest passing variant as explicitly authorized"))
        write("pending_questions.json",saved)
    # Selection is execution metadata, not a test checkpoint; retain with the run.
    (WS/"ckpts/BiCoord_pi05/selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    selection_path=WS/"ckpts/BiCoord_pi05/selection.json"
    train=submit("pi05train",selection["variant"]["devices"],
                 [PYTHON,str(REPO/"script/pi05_job_driver.py"),"train","--selection",str(selection_path)],"72:00:00")
    wait_job(train,"training")
    evaluation=submit("pi05eval",8,[PYTHON,str(REPO/"script/pi05_eval_all.py")],"72:00:00")
    wait_job(evaluation,"evaluating")
    for name in ("metrics.json","comparison.json"):
        fetch(HOSTS[0],REMOTE/"evaluation"/name,RUN/name)
    # Bring the requested video to the submit host so the IDE path is usable.
    query=subprocess.run(["ssh","-o","BatchMode=yes",HOSTS[0],
                          "find "+shlex.quote(str(REMOTE/"evaluation/balance_roller"))+" -name '*.mp4' -print"],
                         capture_output=True,text=True,timeout=30,check=True)
    if query.stdout.splitlines():
        fetch(HOSTS[0],Path(query.stdout.splitlines()[0]),RUN/"inference.mp4")
    else:
        raise RuntimeError("Evaluation completed without the requested inference video")
    status("complete",metrics=str(RUN/"metrics.json"),video=str(RUN/"inference.mp4"),
           checkpoint=str(WS/"ckpts/BiCoord_pi05/checkpoints/50000"))


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        status("failed",error=repr(error))
        path=RUN/"pending_questions.json"
        previous=json.loads(path.read_text()) if path.exists() else []
        previous.append(dict(kind="execution_error",detail=repr(error),needs_review=True,updated_unix=time.time()))
        write("pending_questions.json",previous)
        raise
