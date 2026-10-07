"""Fresh measured marker against actual saved targets and six altered scope negatives."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from analyze import read, bound, ROOT, certify_pixel_contract

out=Path(sys.argv[1]).resolve()
marker=out/'marker_measurement.json'
live_path=Path('C:/Users/13666/Workspace/HS2Mod/artifacts/infrastructure_live_20261005/abmx_lowered_v2/live_cases.json')
windows=read(live_path)['cases'][0]['windows']; successes=[]
for window in windows:
    for view in window['capture']['views']:
        cert=certify_pixel_contract(marker,view)
        successes.append({'window':window['window'],'yaw':view['yaw'],'certificate':cert.__dict__})
view=windows[0]['capture']['views'][0]
def projection(v):v['capture_camera']['projection'][0]*=1.01
def mvid(v):v['capture_camera']['bridge_mvid']='00000000-0000-0000-0000-000000000000'
def aa(v):v['capture_camera']['anti_aliasing']=4
def msaa(v):v['capture_camera']['allow_msaa']=True
def near(v):v['capture_camera']['near_clip']=.16
def colorspace(v):v['color_space']='Gamma'
negatives=[]
for mutation in [projection,mvid,aa,msaa,near,colorspace]:
    target=deepcopy(view); mutation(target)
    try:
        certify_pixel_contract(marker,target)
        negatives.append({'case':mutation.__name__,'rejected':False})
    except (ValueError,KeyError,TypeError,OSError) as exc:
        negatives.append({'case':mutation.__name__,'rejected':True,'reason':str(exc)})
result={'scope':'coordinate contract only; no shader depth or anatomy', 'marker':bound(marker),
        'live_cases':bound(live_path),'positive_count':len(successes),'certificates':successes,
        'negative_count':len(negatives),'negatives':negatives,'source':bound(__file__),
        'historical_mismatch_report':bound(ROOT/'outputs/crossmesh_semantics_20261005/actual_lowered_v2/report.json'),
        'threshold_or_classifier_exemptions_changed':False}
(out/'pixel_scope_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'positive_count':len(successes),'negative_rejections':sum(n['rejected'] for n in negatives)}))
raise SystemExit(0 if len(successes)==9 and all(n['rejected'] for n in negatives) else 1)
