"""Persist subprocess output directly to files, including failed/time-limited runs.

No automatic retries. Callers authorize the command. This utility itself does
not launch graphics or attach to processes; only its supplied child is stopped
on timeout, never a process found by name/PID.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import time

def run_logged(argv,out,*,timeout,env=None,cwd=None):
    out=Path(out)
    if out.exists():raise FileExistsError('Fresh process receipt directory required')
    out.mkdir(parents=True)
    stdout_path=out/'stdout.bin';stderr_path=out/'stderr.bin';started=time.time()
    receipt={'argv':list(map(str,argv)),'timeout_seconds':timeout,'started_unix':started,
             'status':'not_started','stdout':str(stdout_path),'stderr':str(stderr_path)}
    with stdout_path.open('wb') as stdout,stderr_path.open('wb') as stderr:
        try:
            proc=subprocess.Popen(list(map(str,argv)),stdout=stdout,stderr=stderr,env=env,cwd=cwd,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            receipt['pid']=proc.pid
            try:
                proc.wait(timeout=timeout);receipt['status']='pass' if proc.returncode==0 else 'failed_exit'
            except subprocess.TimeoutExpired:
                receipt['status']='timeout';proc.kill();proc.wait()
            receipt['returncode']=proc.returncode
        except Exception as e:
            receipt['status']='launch_failed';receipt['error']=repr(e)
    receipt['ended_unix']=time.time()
    for label,path in [('stdout',stdout_path),('stderr',stderr_path)]:
        data=path.read_bytes();receipt[label+'_bytes']=len(data);receipt[label+'_sha256']=hashlib.sha256(data).hexdigest()
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt
