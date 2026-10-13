"""Build strict-C-ABI scoped capture helper; run only CPU official-table mock.

Never launches RenderDoc, HS2, any fixture or graphics device. Requires the pinned
official application SDK header and installed x64 MSVC. Generated files ignored.
"""
import argparse
import hashlib
import json
from pathlib import Path
from build_target_client_msvc import quoted
from logged_process import run_logged

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sdk',required=True,type=Path)
    ap.add_argument('--vcvars',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--cecil',type=Path,default=Path('E:/HoneySelect2_ArcticFox/BepInEx/core/Mono.Cecil.dll'))
    a=ap.parse_args()
    header=a.sdk/'renderdoc_app.h'
    if not header.is_file() or not a.vcvars.is_file():raise FileNotFoundError('Pinned SDK/installed vcvars64 required')
    if a.out.exists():raise FileExistsError('Fresh build output required')
    a.out.mkdir(parents=True);out=a.out.resolve();root=Path(__file__).resolve().parent/'native_audit'
    lines=['@echo off','call '+quoted(a.vcvars),'if errorlevel 1 exit /b 1',
      f'cl /nologo /std:c++17 /EHsc /O2 /MT /LD /I{quoted(a.sdk)} /FAs /Fa{quoted(out/"scope_static.asm")} /Fo{quoted(out/"scope.obj")} /Fe{quoted(out/"hs2_renderdoc_scope_native.dll")} {quoted(root/"renderdoc_scope.cpp")} bcrypt.lib',
      'if errorlevel 1 exit /b 1',
      f'cl /nologo /std:c++17 /EHsc /O2 /MT /I{quoted(a.sdk)} /Fo{quoted(out/"mock.obj")} /Fe{quoted(out/"scope_cpu_mock.exe")} {quoted(root/"test_renderdoc_scope.cpp")} bcrypt.lib',
      'if errorlevel 1 exit /b 1',
      f'cl /nologo /std:c++17 /EHsc /O2 /MT /I{quoted(a.sdk)} /Fo{quoted(out/"present_mock.obj")} /Fe{quoted(out/"present_cpu_mock.exe")} {quoted(root/"test_present_scope.cpp")} bcrypt.lib',
      'if errorlevel 1 exit /b 1',
      f'dumpbin /imports {quoted(out/"hs2_renderdoc_scope_native.dll")} > {quoted(out/"imports.txt")}',
      'if errorlevel 1 exit /b 1',
      f'dumpbin /exports {quoted(out/"hs2_renderdoc_scope_native.dll")} > {quoted(out/"exports.txt")}',
      'if errorlevel 1 exit /b 1']
    script=out/'build_cpu_only.cmd';script.write_text('\n'.join(lines)+'\n')
    build=run_logged(['cmd','/d','/c',script],out/'build_log',timeout=120,cwd=out)
    if build['status']!='pass':raise RuntimeError('Build failed; actual logs retained')
    imports=(out/'imports.txt').read_text()
    if 'renderdoc.dll' in imports.lower() or 'd3d11.dll' in imports.lower():raise RuntimeError('Forbidden native dependency import')
    test=run_logged([out/'scope_cpu_mock.exe',out/'synthetic_mock_files'],out/'cpu_mock_log',timeout=15,cwd=out)
    if test['status']!='pass':raise RuntimeError('CPU mock failed; logs retained')
    present=run_logged([out/'present_cpu_mock.exe',out/'synthetic_present_files'],out/'present_cpu_mock_log',timeout=15,cwd=out)
    if present['status']!='pass':raise RuntimeError('CPU Present protocol failed; logs retained')
    managed=run_logged(['dotnet','build',root/'scope_managed/HS2_RenderDocScope.csproj','-c','Release','-o',out],out/'managed_build_log',timeout=120,cwd=out)
    if managed['status']!='pass':raise RuntimeError('Managed build failed; logs retained')
    json_audit=run_logged(['powershell','-NoProfile','-File',root/'audit_scope_json.ps1','-Helper',out/'HS2_RenderDocScope.dll','-Cecil',a.cecil.resolve(),'-Output',out/'json_dependency_audit.json'],out/'json_il_audit_log',timeout=30,cwd=out)
    if json_audit['status']!='pass':raise RuntimeError('JSON source/IL dependency audit failed')
    json_build=run_logged(['dotnet','build',root/'scope_json_cpu/JsonCpuTest.csproj','-c','Release','-o',out/'json_cpu_only'],out/'json_cpu_build_log',timeout=120,cwd=out)
    if json_build['status']!='pass':raise RuntimeError('JSON CPU fixture compile failed')
    json_test=run_logged([out/'json_cpu_only/ScopeJsonCpuTest.exe'],out/'json_cpu_log',timeout=15,cwd=out)
    if json_test['status']!='pass':raise RuntimeError('JSON CPU dependency/roundtrip checks failed')
    managed_test=run_logged(['powershell','-NoProfile','-File',root/'test_scope_managed.ps1','-BuildDirectory',out],out/'managed_cpu_log',timeout=15,cwd=out)
    if managed_test['status']!='pass':raise RuntimeError('Managed CPU checks failed; logs retained')
    for variant,define in [('good',''),('wrong_signature','WRONG_SIGNATURE'),('wrong_ctor','WRONG_CTOR')]:
        fixture=out/('fixture_'+variant)
        fixture_build=run_logged(['dotnet','build',root/'scope_fixture/HS2_McpBridge.csproj','-c','Release','-o',fixture,'-p:DefineConstants='+define],out/('fixture_build_'+variant),timeout=120,cwd=out)
        if fixture_build['status']!='pass':raise RuntimeError('CPU geometry fixture compilation failed: '+variant)
        fixture_test=run_logged(['powershell','-NoProfile','-File',root/'test_scope_geometry.ps1','-BuildDirectory',out,'-FixtureDirectory',fixture,'-Variant',variant],out/('geometry_cpu_'+variant+'_log'),timeout=15,cwd=out)
        if fixture_test['status']!='pass':raise RuntimeError('CPU geometry fixture failed: '+variant)
    result={'scope':'CPU official C table mock, x64 MSVC build and managed compile; no RenderDoc DLL/graphics/native game execution',
      'vcvars':str(a.vcvars.resolve()),'sdk_header':str(header.resolve()),'sdk_header_sha256':hashlib.sha256(header.read_bytes()).hexdigest(),'outputs':[]}
    for name in ('hs2_renderdoc_scope_native.dll','HS2_RenderDocScope.dll','Newtonsoft.Json.dll','scope_cpu_mock.exe','present_cpu_mock.exe','imports.txt','exports.txt','scope_static.asm'):
        path=out/name;result['outputs'].append({'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (out/'build_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
