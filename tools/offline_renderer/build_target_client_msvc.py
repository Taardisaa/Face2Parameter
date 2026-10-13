"""Build the official C++ target client with MSVC; no RenderDoc DLL is executed.

Required version-matched headers come from prepare_renderdoc_headers.py. This
build does not launch/inject/capture any graphical target or run the client.
"""
import argparse
import hashlib
import json
from pathlib import Path
from logged_process import run_logged

def quoted(path):
    value=str(Path(path).resolve())
    if any(c in value for c in '&^%|<>"\r\n'):raise ValueError('Unsupported command path character')
    return '"'+value+'"'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--headers',required=True,type=Path)
    ap.add_argument('--vcvars',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Fresh build directory required')
    if not (a.headers/'renderdoc_replay.h').is_file() or not (a.headers/'renderdoc.def').is_file() or not a.vcvars.is_file():raise FileNotFoundError('Official replay headers/export definition and installed vcvars64 required')
    a.out.mkdir(parents=True);root=Path(__file__).resolve().parents[2];out=a.out.resolve()
    lines=['@echo off','call '+quoted(a.vcvars),'if errorlevel 1 exit /b 1',
        f'lib /nologo /def:{quoted(a.headers/"renderdoc.def")} /machine:x64 /out:{quoted(out/"renderdoc.lib")}',
        'if errorlevel 1 exit /b 1',
        f'cl /nologo /std:c++17 /EHsc /O2 /MT /I{quoted(a.headers)} /FAs /Fa{quoted(out/"client_static.asm")} /Fo{quoted(out/"client.obj")} /Fe{quoted(out/"renderdoc_target_msvc.exe")} {quoted(root/"tools/offline_renderer/renderdoc_target.cpp")} {quoted(out/"renderdoc.lib")}',
        'if errorlevel 1 exit /b 1',
        f'cl /nologo /std:c++17 /EHsc /O2 /MT /Fo{quoted(out/"abi.obj")} /Fe{quoted(out/"abi_only.exe")} {quoted(root/"tools/offline_renderer/test_member_result.cpp")}',
        'if errorlevel 1 exit /b 1',
        f'cl /nologo /std:c++17 /EHsc /O2 /MT /I{quoted(a.headers)} /Fo{quoted(out/"layout.obj")} /Fe{quoted(out/"layout_only.exe")} {quoted(root/"tools/offline_renderer/test_replay_layout.cpp")}',
        'if errorlevel 1 exit /b 1',
        f'cl /nologo /std:c++17 /EHsc /O2 /MT /Fo{quoted(out/"guard.obj")} /Fe{quoted(out/"guard_only.exe")} {quoted(root/"tools/offline_renderer/test_capture_guard.cpp")}',
        'if errorlevel 1 exit /b 1',
        f'cl /nologo /std:c++17 /EHsc /O2 /MT /Fo{quoted(out/"fixture.obj")} /Fe{quoted(out/"renderdoc_control_fixture_msvc.exe")} {quoted(root/"tools/offline_renderer/renderdoc_control_fixture.cpp")} d3d11.lib dxgi.lib user32.lib',
        'if errorlevel 1 exit /b 1']
    script=out/'build_cpu_only.cmd';script.write_text('\n'.join(lines)+'\n')
    receipt=run_logged(['cmd','/d','/c',script],out/'build_log',timeout=120,cwd=out)
    if receipt['status']!='pass':raise RuntimeError('MSVC compilation failed; output retained')
    result={'scope':'MSVC compilation only; no native DLL/client or graphics execution','vcvars':str(a.vcvars.resolve()),
        'headers':str(a.headers.resolve()),'outputs':[]}
    for name in ('renderdoc_target_msvc.exe','abi_only.exe','layout_only.exe','guard_only.exe','renderdoc_control_fixture_msvc.exe','client_static.asm'):
        p=out/name;result['outputs'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (out/'build_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
