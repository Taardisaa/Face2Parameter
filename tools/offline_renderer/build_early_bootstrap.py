"""CPU compile/audit/mock of early Doorstop bootstrap. Never starts/loads graphics."""
import argparse
import hashlib
import json
from pathlib import Path
from logged_process import run_logged
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preloader',required=True,type=Path);ap.add_argument('--cecil',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Fresh CPU build output required')
    out=a.out.resolve();out.mkdir(parents=True);root=Path(__file__).resolve().parent/'native_audit'
    for name,project,destination in [('bootstrap','HS2_RenderDocBootstrap',out),('fixture','Fixture',out/'fixture_only'),('test','CpuTest',out/'cpu_test_only')]:
        result=run_logged(['dotnet','build',root/'early_bootstrap'/(project+'.csproj'),'-c','Release','-o',destination],out/(name+'_build_log'),timeout=120,cwd=out)
        if result['status']!='pass':raise RuntimeError('CPU compile failed: '+name)
    audit=run_logged(['powershell','-NoProfile','-File',root/'audit_preloader.ps1','-Preloader',a.preloader.resolve(),'-Cecil',a.cecil.resolve(),'-Output',out/'installed_preloader_audit.json','-Bootstrap',out/'HS2_RenderDocBootstrap.dll'],out/'metadata_audit_log',timeout=30,cwd=out)
    if audit['status']!='pass':raise RuntimeError('CPU metadata audit failed')
    test=run_logged([out/'cpu_test_only/HS2_BootstrapCpuTest.exe',out/'synthetic_cpu_files',out/'fixture_only/BepInEx.Preloader.dll'],out/'cpu_sequence_log',timeout=15,cwd=out)
    if test['status']!='pass':raise RuntimeError('CPU sequence fixture failed')
    dll=out/'HS2_RenderDocBootstrap.dll';receipt={'scope':'CPU installed IL/metadata, compile and synthetic sequence only; no installed preloader, RenderDoc DLL, graphics or game execution',
      'bootstrap_path':str(dll),'bytes':dll.stat().st_size,'sha256':hashlib.sha256(dll.read_bytes()).hexdigest(),
      'installed_preloader_audit':json.loads((out/'installed_preloader_audit.json').read_text(encoding='utf-8-sig')),
      'cpu_test_output':(out/'cpu_sequence_log/stdout.bin').read_text(),'native_early_load_accepted':False}
    (out/'build_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'bootstrap':str(dll),'sha256':receipt['sha256'],'native_accepted':False}))
if __name__=='__main__':main()
