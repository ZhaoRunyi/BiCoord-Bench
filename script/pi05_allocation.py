"""Validate own full-host PBS GPU allocation before creating a JAX backend."""
import json
import os
import re
import socket
import subprocess

job = os.environ["PBS_JOBID"]
record = json.loads(subprocess.check_output(["/opt/pbs/bin/qstat","-f","-F","json",job]))["Jobs"][job]
host = socket.gethostname().split(".")[0]
assigned = sum(int(count) for node,count in re.findall(r"(recgpu\d+)\[\d+\]:[^+)]*?ngpus=(\d+)",record["exec_vnode"]) if node == host)
if assigned != 8:
    raise SystemExit(f"Need all 8 GPUs assigned to own job on {host}; got {assigned}")
mask = os.environ.get("CUDA_VISIBLE_DEVICES")
if mask is not None and len(mask.split(",")) != 8:
    raise SystemExit(f"Scheduler GPU mask contains {len(mask.split(','))} devices, expected 8")
print(json.dumps(dict(job=job,host=host,assigned_gpus=assigned,scheduler_mask_present=mask is not None)),flush=True)
