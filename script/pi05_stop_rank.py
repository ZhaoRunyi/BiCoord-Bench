"""Stop only an owned stalled training rank matching an exact PBS job and output."""
import os
from pathlib import Path
import signal
import sys
import time

job,output=sys.argv[1:]
owned=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    try:
        if path.stat().st_uid!=os.getuid():
            continue
        args=(path/'cmdline').read_bytes().split(b'\0')
        env=(path/'environ').read_bytes().split(b'\0')
        if b'PBS_JOBID='+job.encode() not in env:
            continue
        if not any(arg.endswith(b'train_bicoord.py') for arg in args):
            continue
        position=args.index(b'--output')
        if args[position+1]!=output.encode():
            continue
        owned.append(int(path.name))
    except (OSError,ValueError,IndexError):
        continue
for pid in owned:
    try:
        os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(2)
for pid in owned:
    try:
        os.kill(pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
print({'stopped_owned_ranks':owned,'job':job},flush=True)
