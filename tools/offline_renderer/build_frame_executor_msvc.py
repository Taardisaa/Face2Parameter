"""Compile tools/offline_renderer/frame_executor.cpp with MSVC into a fresh directory.

    python tools/offline_renderer/build_frame_executor_msvc.py \
        --vcvars "C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Auxiliary/Build/vcvars64.bat" \
        --out outputs/offline_renderer_executor_20261020/build_vN

Links only system D3D11/DXGI/BCrypt import libraries (no RenderDoc). Writes
build_receipt.json with the source and output SHA-256 values. Compilation only;
nothing is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path

from build_target_client_msvc import quoted
from logged_process import run_logged


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vcvars', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError('Fresh build directory required')
    if not a.vcvars.is_file():
        raise FileNotFoundError('Installed MSVC vcvars64.bat required')
    source = Path(__file__).resolve().parent / 'frame_executor.cpp'
    a.out.mkdir(parents=True)
    out = a.out.resolve()
    exe = out / 'frame_executor.exe'
    script = out / 'build.cmd'
    script.write_text('\n'.join([
        '@echo off', 'call ' + quoted(a.vcvars), 'if errorlevel 1 exit /b 1',
        f'cl /nologo /std:c++17 /EHsc /O2 /MT /W3 /permissive- /Fo{quoted(out / "frame_executor.obj")} '
        f'/Fe{quoted(exe)} {quoted(source)} d3d11.lib dxgi.lib bcrypt.lib',
        'if errorlevel 1 exit /b 1']) + '\n')
    result = run_logged(['cmd', '/d', '/c', script], out / 'build_log', timeout=300, cwd=out)
    if result['status'] != 'pass':
        raise RuntimeError('MSVC compile failed; logs retained in ' + str(out / 'build_log'))
    receipt = {'scope': 'MSVC compilation of the frame-program executor; no execution',
               'vcvars': str(a.vcvars.resolve()), 'source': str(source), 'source_sha256': sha(source),
               'compiler_stdout': result['stdout'],
               'outputs': [{'path': str(exe), 'bytes': exe.stat().st_size, 'sha256': sha(exe)}]}
    (out / 'build_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
