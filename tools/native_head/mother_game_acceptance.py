"""Read-only validation that native card loading selected the packaged meshes.

Checks complete source mesh arrays, frame/channel identity and native scalar
state in the final frozen capture. This does not certify every animation pose.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import UnityPy

from tools.native_head.mother_template_inputs import save_json, source_file


def verify(bindings, geometry_path, out):
    if out.exists():
        raise FileExistsError('Preserve integration evidence')
    authoring=json.loads((bindings/'receipt.json').read_text())
    actual=json.loads(geometry_path.read_text())
    inputs=json.loads(Path(authoring['inputs']['path']).read_text())
    prefab=json.loads(Path(inputs['native_prefab_export']['path']).read_text())
    donor={r.path_id:r for r in UnityPy.load(inputs['native_original_copy']['path']).objects}
    checks=[]
    for row in authoring['renderer_outputs']:
        name=row['mesh']
        meshes=[r for r in actual['meshes'] if r['mesh_name']==name and
            '/ct_head[' in r['renderer_path'] and r['active_in_hierarchy']]
        if len(meshes)!=1:
            raise ValueError('Native head component missing or ambiguous: '+name)
        mesh=meshes[0]
        arrays=np.load(row['arrays']['path'],allow_pickle=False)
        source_row=next(r for r in prefab['renderers'] if r['mesh']==name)
        original_channels=donor[source_row['mesh_path_id']].read_typetree()['m_VertexData']['m_Channels']
        fields=dict(verts='vertices',normals='normals',tangents='tangents',
            uv='uv',uv1='uv2',faces='triangles',bone_idx='bone_indices',bone_w='bone_weights',bindpose='bindposes')
        matched={}
        for key,field in fields.items():
            if key=='uv1' and original_channels[5]['dimension']==0:
                # Extraction NPZ has a zero-filled convenience array for absent
                # UV1; Unity correctly exposes the actual absent channel as [].
                matched[key]=len(mesh['source'][field])==0
                continue
            exported=np.asarray(mesh['source'][field],dtype=arrays[key].dtype)
            if key=='faces':
                exported=exported.reshape(-1,3)
            if key=='bindpose':
                exported=exported.reshape(-1,4,4)
            if arrays[key].dtype.kind=='f':
                matched[key]=bool(np.array_equal(exported.astype(np.float32),arrays[key].astype(np.float32)))
            else:
                matched[key]=bool(np.array_equal(exported,arrays[key]))
        frames=json.loads(Path(row['frames']['path']).read_text())
        channels=mesh['blendshapes']
        matched['original_channel_names']=([r['name'] for r in channels]==[r['name'] for r in frames['channels']])
        matched['source_bone_names']=mesh['bone_names']==arrays['bone_names'].tolist()
        if not all(matched.values()):
            raise ValueError('Native loaded asset differs from authored '+name+': '+str(matched))
        checks.append(dict(mesh=name,enabled=mesh['enabled'],source_arrays_and_channels_match=matched,
            source_geometry_sha256=mesh['source_geometry_sha256'],channel_count=len(channels)))
    char=actual['character']
    if char['shape_value_face']!=[.5]*59:
        raise ValueError('Trial card did not select the authored nominal driver state')
    save_json(out,dict(format='native_mother_game_acceptance_v1',
        source_bindings=source_file(bindings/'receipt.json'),geometry=source_file(geometry_path),
        code=source_file(Path(__file__)),native_head_id=char['head_id'],native_skin_id=char['skin_id'],
        original_eight_components_loaded=len(checks)==8,components=checks,
        native_nominal_face_values=True,installed_source_arrays_match=True,
        all_expression_poses_certified=False,appearance_certified=False,
        limitations=['Source arrays and original channel identities checked; full individual runtime sparse frame values remain in exported evidence',
            'No full animation/slider/ABMX compatibility claim',
            'Posterior neck contour and shader/material appearance require further work']))
    print(json.dumps(dict(native_head_id=char['head_id'],all_eight_components_match=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bindings',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    verify(a.bindings.resolve(),a.geometry.resolve(),a.out.resolve())
