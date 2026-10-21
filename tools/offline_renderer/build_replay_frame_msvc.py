"""Compile the official full-frame conversion/replay client without executing it.

The patched-capture pipeline (sections/index/session modes) additionally compiles
the pinned v1.46 `renderdoc/3rdparty/lz4` codec source, i.e. the same LZ4 source
the official DLL embeds, and links the Windows BCrypt SHA-256 provider.
"""
import argparse
import hashlib
import json
from pathlib import Path
from build_target_client_msvc import quoted
from logged_process import run_logged

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--headers',required=True,type=Path)
    ap.add_argument('--vcvars',required=True,type=Path)
    ap.add_argument('--lz4',required=True,type=Path,help='pinned renderdoc-<commit>/renderdoc/3rdparty/lz4 directory')
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Fresh build directory required')
    if not (a.headers/'renderdoc_replay.h').is_file() or not a.vcvars.is_file():raise FileNotFoundError('Pinned official headers and installed MSVC required')
    if not (a.lz4/'lz4.c').is_file() or not (a.lz4/'lz4.h').is_file():raise FileNotFoundError('Pinned RenderDoc 3rdparty lz4 source required')
    if 'e4bd23b671d3d5a747ff5221dbe08a63eb6ca200' not in str(a.lz4.resolve()):raise ValueError('LZ4 source must come from the pinned v1.46 source tree')
    a.out.mkdir(parents=True);out=a.out.resolve();source=Path(__file__).resolve().parent/'renderdoc_replay_frame.cpp'
    exports=['RENDERDOC_AllocArrayMem','RENDERDOC_FreeArrayMem','RENDERDOC_InitialiseReplay','RENDERDOC_ShutdownReplay','RENDERDOC_OpenCaptureFile','RENDERDOC_GetVersionString','RENDERDOC_GetCommitHash']
    definition=out/'renderdoc.def';definition.write_text('LIBRARY renderdoc.dll\nEXPORTS\n'+'\n'.join(exports)+'\n')
    script=out/'build_cpu_only.cmd';script.write_text('\n'.join(['@echo off','call '+quoted(a.vcvars),'if errorlevel 1 exit /b 1',
      f'lib /nologo /def:{quoted(definition)} /machine:x64 /out:{quoted(out/"renderdoc.lib")}',
      'if errorlevel 1 exit /b 1',
      f'cl /nologo /c /O2 /MT /I{quoted(a.lz4)} /Fo{quoted(out/"lz4.obj")} {quoted(a.lz4/"lz4.c")}',
      'if errorlevel 1 exit /b 1',
      f'cl /nologo /std:c++17 /EHsc /O2 /MT /I{quoted(a.headers)} /I{quoted(a.lz4)} /FAs /Fa{quoted(out/"replay_static.asm")} /Fo{quoted(out/"replay.obj")} /Fe{quoted(out/"renderdoc_replay_msvc.exe")} {quoted(source)} {quoted(out/"lz4.obj")} {quoted(out/"renderdoc.lib")} bcrypt.lib',
      'if errorlevel 1 exit /b 1'])+'\n')
    result=run_logged(['cmd','/d','/c',script],out/'build_log',timeout=180,cwd=out)
    if result['status']!='pass':raise RuntimeError('MSVC compile failed; logs retained')
    receipt={'scope':'MSVC compilation/static C++ layout only; no DLL/replay/graphics execution or runtime acceptance',
      'headers':str(a.headers.resolve()),'vcvars':str(a.vcvars.resolve()),'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'lz4_sources':[{'path':str((a.lz4/n).resolve()),'sha256':hashlib.sha256((a.lz4/n).read_bytes()).hexdigest()} for n in ('lz4.c','lz4.h')],'outputs':[]}
    for name in ('renderdoc_replay_msvc.exe','replay_static.asm'):
        p=out/name;receipt['outputs'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (out/'build_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
