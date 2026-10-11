"""Clean only orphaned multiprocessing workers belonging to this PBS job."""
import json
import os
from pathlib import Path
import signal

job=os.environ["PBS_JOBID"]
stopped=[]
for path in Path("/proc").glob("[0-9]*"):
    try:
        if path.stat().st_uid!=os.getuid():
            continue
        status=(path/"status").read_text()
        if "\nPPid:\t1\n" not in status:
            continue
        env=(path/"environ").read_bytes().split(b"\0")
        if f"PBS_JOBID={job}".encode() not in env:
            continue
        command=(path/"cmdline").read_bytes()
        if not any(marker in command for marker in (b"multiprocessing.spawn",b"multiprocessing.resource_tracker")):
            continue
        pid=int(path.name)
        os.kill(pid,signal.SIGTERM)
        stopped.append(pid)
    except (OSError,ProcessLookupError):
        continue
print(json.dumps(dict(cleaned_orphan_workers=stopped,job=job)),flush=True)
