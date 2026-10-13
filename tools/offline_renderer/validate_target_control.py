"""One explicit standalone 32x32 WARP target-control acceptance; never HS2.

Requires the new MSVC build. Persists logs before launch and inventories exact
related PIDs before/after. No retry, global hook, forced client takeover or
process-name termination. First failure ends the experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from logged_process import run_logged

def inventory(out):
    script=r"Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(renderdoccmd|qrenderdoc|renderdoc_target_msvc|renderdoc_control_fixture_msvc)\.exe$' } | ForEach-Object { $p=Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; [pscustomobject]@{pid=$_.ProcessId;parent_pid=$_.ParentProcessId;name=$_.Name;executable=$_.ExecutablePath;command_line=$_.CommandLine;window_title=$p.MainWindowTitle} } | ConvertTo-Json -Depth 3"
    r=run_logged(['powershell','-NoProfile','-Command',script],out,timeout=10)
    if r['status']!='pass':raise RuntimeError('Related-process inventory failed; no graphics launch authorized')
    raw=(Path(out)/'stdout.bin').read_bytes().decode('utf-8-sig',errors='replace').strip()
    result=json.loads(raw) if raw else []
    return result if isinstance(result,list) else [result]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build',required=True,type=Path)
    ap.add_argument('--portable',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--run-one-authorized-fixture',action='store_true',required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Fresh single-attempt output required')
    a.out.mkdir(parents=True);out=a.out.resolve();build=a.build.resolve();portable=a.portable.resolve()
    receipt={'status':'not_started','scope':'one independent 32x32 WARP fixture; no HS2 launch or injection','retries':0}
    launch=None
    try:
        before=inventory(out/'inventory_before');receipt['inventory_before']=before
        if before:raise RuntimeError('Existing related process prevents isolated single-attempt validation')
        declared=json.loads((build/'build_receipt.json').read_text())
        for name in ('renderdoc_target_msvc.exe','renderdoc_control_fixture_msvc.exe'):
            p=build/name;actual=hashlib.sha256(p.read_bytes()).hexdigest()
            match=[r for r in declared['outputs'] if Path(r['path']).name==name]
            if len(match)!=1 or match[0]['sha256']!=actual:raise RuntimeError('Executable changed since the MSVC build receipt')
        env=dict(os.environ);env['PATH']=str(portable)+';'+env['PATH']
        capture=out/'capture';ready=out/'fixture_ready_pid.txt'
        with (out/'launch_stdout.bin').open('wb') as stdout,(out/'launch_stderr.bin').open('wb') as stderr:
            launch=subprocess.Popen([str(portable/'renderdoccmd.exe'),'capture','-w','-c',str(capture),str(build/'renderdoc_control_fixture_msvc.exe'),str(ready)],stdout=stdout,stderr=stderr,env=env,cwd=out,creationflags=subprocess.CREATE_NO_WINDOW)
            receipt['launcher_pid']=launch.pid
            deadline=time.monotonic()+10
            while not ready.exists() and launch.poll() is None and time.monotonic()<deadline:time.sleep(.05)
            if not ready.exists():raise RuntimeError('Fixture readiness deadline failed')
            pid=int(ready.read_text());receipt['fixture_pid']=pid
            listed=run_logged([build/'renderdoc_target_msvc.exe','list'],out/'enumerate',timeout=8,env=env,cwd=out)
            if listed['status']!='pass':raise RuntimeError('Target enumeration failed')
            targets=[json.loads(row)['ident'] for row in (out/'enumerate/stdout.bin').read_text().splitlines()]
            if len(targets)!=1:raise RuntimeError('One isolated target required; no guessing target ownership')
            receipt['ident']=targets[0]
            client=run_logged([build/'renderdoc_target_msvc.exe',str(targets[0]),str(pid),'trigger','8',str(capture)],out/'client',timeout=12,env=env,cwd=out)
            receipt['client']=client
            if client['status']!='pass':raise RuntimeError('Client acceptance failed; no retries')
            messages=[json.loads(row) for row in (out/'client/stdout.bin').read_text().splitlines()]
            accepted=[row for row in messages if row['event']=='new_capture']
            if len(accepted)!=1:raise RuntimeError('Exactly one reported capture required')
            p=Path(accepted[0]['path'])
            if not p.is_file() or p.stat().st_size!=accepted[0]['bytes']:raise RuntimeError('Actual RDC file evidence mismatch')
            receipt.update(status='pass',messages=messages,capture_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            launch.wait(timeout=35)
    except Exception as e:
        receipt.update(status='failed_no_retry',error=repr(e))
    finally:
        if launch and launch.poll() is None:
            # Only the Popen-owned launcher is stopped. Any surviving child/helper
            # remains explicitly reported for exact-owner review, never killed by name.
            launch.kill();launch.wait()
        try:receipt['inventory_after']=inventory(out/'inventory_after')
        except Exception as e:receipt['inventory_after_error']=repr(e)
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
    if receipt['status']!='pass':raise SystemExit(1)
if __name__=='__main__':main()
