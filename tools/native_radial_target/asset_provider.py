"""Explicit identity-source, per-renderer asset revision. No shared cache writes.

Original native radial compiler, adapter and runtime validator are unchanged.
Targets can use this provider only after the earlier clean identity source and
explicit authored asset selections have been verified. Candidate after states
are never asset selections or target-generation inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import types
import zipfile
from pathlib import Path

import numpy as np
import UnityPy

from src.hs2_mesh_deform import HeadRig
from tools.abmx_multibone.geometry import select_cursor_predictions
from tools.abmx_replay.model import IDENTITY, modifier, serialized
from tools.abmx_replay.validate_trace import read, sha
from tools.abmx_stable_lowering.compiler import check_files, digest_json
from tools.native_radial_target.adapter import NAMES, PROFILE, native_source_bindings
from tools.native_radial_target.audit_source_assets import (
    asset_renderers,
    compare,
    select_head,
)
from tools.native_radial_target.compiler import _clean, require

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'identity_source_assets_v1'
_CAPABILITY = object()


def _save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(serialized(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def _array_digest(value, dtype=np.float32):
    if dtype is np.int64:
        raw = np.asarray(value)
        require(raw.dtype.kind in 'iu', 'Exact integer topology/influence arrays required; bool/fractional indices refused')
    a = np.asarray(value, dtype=dtype)
    require(np.isfinite(a).all(), 'Nonfinite material source arrays')
    # Unity JSON may print authored -0 as +0. Canonical numerical equality
    # normalizes only exact zeros; every nonzero bit remains unchanged.
    a = a.copy()
    a[a == 0] = 0
    return {'shape': list(a.shape), 'dtype': str(a.dtype),
            'canonical_zero_policy': 'exact signed zeros normalized to positive zero; no nonzero rounding',
            'sha256': hashlib.sha256(str(a.shape).encode()+a.tobytes(order='C')).hexdigest()}


def _frames(mesh, count):
    """Decode authored Unity sparse frames into GetBlendShapeFrameVertices arrays."""
    s = mesh.read_typetree()['m_Shapes']; result = []
    for index, channel in enumerate(s['channels']):
        require(type(channel['frameIndex']) is int and type(channel['frameCount']) is int, 'Invalid authored shape channel')
        frames = []
        for local_index in range(channel['frameCount']):
            i = channel['frameIndex']+local_index; frame=s['shapes'][i]
            arrays = {name:np.zeros((count,3),np.float32) for name in ('vertex','normal','tangent')}
            entries=s['vertices'][frame['firstVertex']:frame['firstVertex']+frame['vertexCount']]
            require(len(entries)==frame['vertexCount'] and len({e['index'] for e in entries})==len(entries), 'Invalid sparse frame coverage')
            for e in entries:
                require(type(e['index']) is int and 0<=e['index']<count, 'Invalid authored shape vertex index')
                for name, array in arrays.items():
                    if name=='normal' and not frame['hasNormals']:continue
                    if name=='tangent' and not frame['hasTangents']:continue
                    array[e['index']]=[e[name][axis] for axis in ('x','y','z')]
            frames.append({'frame_index':local_index,'weight':float(np.float32(s['fullWeights'][i])),
                           'delta_vertices':arrays['vertex'].tolist(),'delta_normals':arrays['normal'].tolist(),
                           'delta_tangents':arrays['tangent'].tolist()})
        result.append({'shape_index':index,'name':channel['name'],'frame_count':channel['frameCount'],'frames':frames})
    return result


def _shape_inventory(rows, count):
    require(type(rows) is list, 'Actual authored blendshape rows missing')
    result = []
    for index, row in enumerate(rows):
        require(type(row['shape_index']) is int and row['shape_index']==index and type(row['name']) is str,
                'Shape ordering/name mismatch')
        frames=row.get('frames')
        require(type(row['frame_count']) is int and row['frame_count']>0 and type(frames) is list
                and len(frames)==row['frame_count'], 'All authored blendshape frames required, including inactive frames')
        fs=[]
        for i, frame in enumerate(frames):
            require(type(frame['frame_index']) is int and frame['frame_index']==i, 'Shape frame ordering mismatch')
            weight=np.asarray(frame['weight'],np.float32)
            require(weight.shape==() and np.isfinite(weight), 'Finite frame weight required')
            arrays={}
            for k in ('delta_vertices','delta_normals','delta_tangents'):
                arrays[k]=_array_digest(frame[k])
                require(arrays[k]['shape']==[count,3], 'Full vertex-aligned blendshape frame required')
            fs.append({'frame_index':i,'weight':float(weight),**arrays})
        result.append({'shape_index':index,'name':row['name'],'frame_count':row['frame_count'],'frames':fs})
    return result


def _mesh_inventory(mesh):
    source=mesh['source']
    require(source['space']=='shared_mesh_local_bind_geometry', 'Unknown source mesh coordinate space')
    fields={k:_array_digest(v,np.int64 if k in ('triangles','bone_indices') else np.float32)
            for k,v in source.items() if k not in ('space','submeshes')}
    require(set(fields)=={'vertices','normals','tangents','uv','uv2','triangles','bone_indices','bone_weights','bindposes'},
            'Unknown/missing exported source arrays')
    colors=_array_digest(mesh['source_vertex_colors_rgba'])
    submeshes=source['submeshes']
    require(type(submeshes) is list, 'Submesh topology metadata missing')
    weights=np.asarray([r['current_weight'] for r in mesh['blendshapes']],np.float32)
    require(np.isfinite(weights).all(), 'Nonfinite active source blendshape weights')
    return {'arrays':fields,'submeshes':submeshes,'source_vertex_colors_rgba':colors,
            'blendshape_frames':_shape_inventory(mesh['blendshapes'],mesh['vertex_count']),
            'active_weights':weights.tolist()}


def _asset_rows(env,prefab):
    meshes={o.path_id:o for o in env.objects if o.type.name=='Mesh'}
    rows=asset_renderers(env,prefab)
    for row in rows:
        obj=meshes[row['mesh_path_id']]
        row['blendshapes']=_frames(obj,len(row['data']['verts']))
        tt=obj.read_typetree()
        row['raw_authored_submeshes']=tt['m_SubMeshes']
    return rows


def _origin_candidate(rows, mesh, expected=None):
    found=[r for r in rows if r['mesh_name']==mesh['mesh_name']]
    if expected:
        found=[r for r in found if r['renderer_path_id']==expected['renderer_path_id'] and r['mesh_path_id']==expected['mesh_path_id']]
    require(len(found)==1,'Missing/ambiguous explicitly selected authored renderer '+mesh['mesh_name'])
    return found[0]


def _authored_check(mesh, asset):
    report=compare(mesh,asset['data'],asset['bone_names'])
    require(report['deformation_correspondence_passed'], 'Selected authored geometry differs from actual SOURCE '+mesh['mesh_name'])
    for key in ('normals','tangents','uv'):
        require(report['array_checks'][key]['passed'], 'Authored source '+key+' differs')
    shapes=_shape_inventory(asset['blendshapes'],mesh['vertex_count'])
    actual=_shape_inventory(mesh['blendshapes'],mesh['vertex_count'])
    require(actual==shapes, 'Authored blendshape names/weights/frames/deltas differ from actual SOURCE '+mesh['mesh_name'])
    uv2=report['array_checks']['uv2']
    if not uv2['passed']:
        require(len(mesh['source']['uv2'])==0 and not np.any(asset['data']['uv1']), 'Unsupported nonempty/mismatched authored UV2')
    color_source=np.asarray(mesh['source_vertex_colors_rgba'],np.float32)
    if not color_source.size:
        require(np.all(asset['data']['colors']==1), 'Absent runtime colors with nondefault asset colors unsupported')
    else:
        require(color_source.shape==asset['data']['colors'].shape and np.array_equal(color_source,asset['data']['colors']), 'Authored vertex colors differ')
    report.update(authored_blendshape_frames_exact=True,active_weights_source_only=True,
                  runtime_uv2_absent=len(mesh['source']['uv2'])==0,uv2_certified=uv2['passed'],
                  runtime_colors_absent=not color_source.size,
                  absent_optional_channels_policy='Recorded as absent; asset extractor defaults are not certified authored runtime channels and are not used by geometry mathematics')
    return report


def stage_from_identity(history, abmx_contract_path, overrides, out_dir):
    """Stopped clean source only; explicit exact-renderer overrides; fresh output only."""
    out=Path(out_dir);require(not out.exists(),'NEW isolated staging directory required')
    require(type(overrides) is dict, 'Exact renderer-path override map required')
    geometry=history['geometry'];require(sha(geometry['path'])==geometry['sha256'] and sha(history['trace'])==history['trace_sha256'], 'Source file hash mismatch')
    snapshot,trace=read(geometry['path']),read(history['trace']);contract=read(abmx_contract_path)
    require(snapshot['snapshot_kind']=='maker_live_skinned_geometry' and snapshot['frame_count']==snapshot['frame_count_end'], 'Stable source geometry required')
    require(trace['active'] is False and trace['trace_complete'] is True, 'Stopped complete earlier identity trace required')
    for e in trace['events']:
        require(serialized(modifier(e['resolved_modifier']))==serialized(modifier(IDENTITY)) and not e['additional_modifiers'], 'Candidate/nonidentity history cannot select assets')
    for row in snapshot['abmx_runtime']['bones']:
        require(serialized(modifier({k:row[k] for k in IDENTITY}))==serialized(modifier(IDENTITY)), 'Candidate snapshot cannot select assets')
    _,cursor,replay=select_cursor_predictions(snapshot,trace,contract,required_names=NAMES)
    require(not replay['inter_call_boundaries'], 'Unidentified source history writer boundary')
    for name in NAMES:
        rows=[r for r in snapshot['abmx_runtime']['bones'] if r['name']==name]
        require(len(rows)==1,'Missing/ambiguous clean source bone')
        _clean(rows[0]['runtime_baseline']['fields'])
    head=snapshot['character']['head_id'];require(type(head) is int and head in (0,1,2,3),'Supported actual head required')
    rig=HeadRig(head,sampling_profile=PROFILE)
    list_path=ROOT/'data/hs2_head/chalist_cache.json';lists=read(list_path);row=lists['by_cat']['210'][str(head)]
    bundle=Path(lists['ab_dir'])/row['MainAB'];prefab=row['MainData']
    default_rows=_asset_rows(UnityPy.load(str(bundle)),prefab)
    source_meshes,skipped=select_head(snapshot)
    require(set(overrides)<={m['renderer_path'] for m in source_meshes}, 'Override renderer absent from actual source scope')
    source_files={str(Path(geometry['path']).resolve()):sha(geometry['path']),str(Path(history['trace']).resolve()):sha(history['trace']),
                  str(Path(abmx_contract_path).resolve()):sha(abmx_contract_path),str(list_path.resolve()):sha(list_path),
                  str(bundle.resolve()):sha(bundle),str(Path(__file__).resolve()):sha(__file__),**native_source_bindings(rig)}
    source_files[str(Path(__file__).with_name('audit_source_assets.py').resolve())]=sha(Path(__file__).with_name('audit_source_assets.py'))
    source_files[str(ROOT/'scripts/hs2_extract_head.py')]=sha(ROOT/'scripts/hs2_extract_head.py')
    source_files[str(ROOT/'tools/native_radial_target/validate_runtime.py')]=sha(ROOT/'tools/native_radial_target/validate_runtime.py')
    prepared=[]
    for mesh in source_meshes:
        path=mesh['renderer_path'];override=overrides.get(path)
        if override:
            require(head==3 and mesh['mesh_name']=='o_tooth', 'This revision permits only explicit head3 tooth replacement')
            require(set(override)=={'archive','archive_sha256','member','member_sha256','prefab','renderer_path_id','mesh_path_id'}, 'Complete archive/member/prefab/pathID selection required')
            require(override['prefab']==prefab and override['member']=='abdata/'+row['MainAB'], 'Wrong head/prefab/archive member')
            for k in ('renderer_path_id','mesh_path_id'):require(type(override[k]) is int,'Exact authored pathID required')
            require(sha(override['archive'])==override['archive_sha256'],'Archive SHA mismatch')
            with zipfile.ZipFile(override['archive']) as z:blob=z.read(override['member'])
            require(hashlib.sha256(blob).hexdigest()==override['member_sha256'],'Archive member SHA mismatch')
            asset=_origin_candidate(_asset_rows(UnityPy.load(blob),prefab),mesh,override)
            origin={'kind':'explicit_zipmod',**override}
            source_files[str(Path(override['archive']).resolve())]=sha(override['archive'])
            data=copy.deepcopy(asset['data'])
        else:
            asset=_origin_candidate(default_rows,mesh)
            npz=Path(rig.data_dir)/('o_head_mesh.npz' if mesh['mesh_name']=='o_head' else 'submeshes/'+mesh['mesh_name']+'.npz')
            with np.load(npz,allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
            names=rig.skin_bone_names if mesh['mesh_name']=='o_head' else [rig.bones[str(pid)]['name'] for pid in data['skin_bone_pids']]
            check=compare(mesh,data,names)
            require(check['deformation_correspondence_passed'], 'Unselected cache mismatch; explicit source override required '+mesh['mesh_name'])
            for k in ('normals','tangents','uv'):require(check['array_checks'][k]['passed'], 'Cache authored channel differs')
            origin={'kind':'existing_same_head_cache','npz':str(npz.resolve()),'npz_sha256':sha(npz),
                    'authored_bundle':str(bundle.resolve()),'authored_bundle_sha256':sha(bundle),'prefab':prefab,
                    'renderer_path_id':asset['renderer_path_id'],'mesh_path_id':asset['mesh_path_id']}
            source_files[str(npz.resolve())]=sha(npz)
        authored=_authored_check(mesh,asset)
        require(all(name in rig.name2pid for name in mesh['bone_names']), 'Missing skin bone in fixed cached rig')
        data['skin_bone_pids']=np.asarray([rig.name2pid[name] for name in mesh['bone_names']])
        prepared.append((mesh,data,origin,authored))
    check_files(source_files)
    # No filesystem mutation until every source-only selection has passed.
    out.mkdir(parents=True);(out/'meshes').mkdir()
    entries=[];stage_files={}
    for i,(mesh,data,origin,authored) in enumerate(prepared):
        npz=out/'meshes'/f'{i:02}_{mesh["mesh_name"]}.npz';np.savez(npz,**data)
        shapes=out/'meshes'/f'{i:02}_{mesh["mesh_name"]}_blendshapes.json';_save(shapes,mesh['blendshapes'])
        for p in (npz,shapes):stage_files[str(p.resolve())]=sha(p)
        entries.append({'mesh_name':mesh['mesh_name'],'renderer_path':mesh['renderer_path'],'renderer_id':mesh['renderer_id'],
                        'renderer_transform_id':mesh['renderer_transform_id'],'mesh_instance_id':mesh['mesh_instance_id'],
                        'source_geometry_sha256':mesh['source_geometry_sha256'],'bone_names':mesh['bone_names'],
                        'bone_transform_ids':mesh['bone_transform_ids'],'origin':origin,'source_inventory':_mesh_inventory(mesh),
                        'stage_npz':str(npz.resolve()),'stage_npz_sha256':sha(npz),'blendshapes_path':str(shapes.resolve()),
                        'blendshapes_sha256':sha(shapes),'all_staged_arrays':{k:_array_digest(v,np.int64 if k in ('faces','bone_idx') else (str if k=='skin_bone_pids' else np.float32)) for k,v in data.items() if k!='skin_bone_pids'},
                        'skin_bone_pids':data['skin_bone_pids'].tolist(),'authored_correspondence':authored})
    result={'schema_version':1,'revision':REVISION,'head_id':head,'sampling_profile':PROFILE,
            'abmx_contract':{'path':str(Path(abmx_contract_path).resolve()),'sha256':sha(abmx_contract_path)},
            'native59':snapshot['character']['shape_value_face'],'source_actor_id':snapshot['character']['transform_id'],
            'source_bridge_mvid':trace['metadata']['bridge_mvid'],'source_history':{'trace':str(Path(history['trace']).resolve()),
                'trace_sha256':history['trace_sha256'],'geometry':{'path':str(Path(geometry['path']).resolve()),'sha256':geometry['sha256']}},
            'source_cursor':cursor,'source_files':source_files,'stage_files':stage_files,'meshes':entries,'skipped':skipped,
            'source_game_assembly_sha256':snapshot['game']['game_assembly_sha256'],
            'explicit_overrides':copy.deepcopy(overrides),'actual_source_only':True,'candidate_inputs_used':False,
            'shared_cache_modified':False,'global_zipmod_redirect':False,'active_shape_source_policy':'All authored frames compared; actual source active weights frozen nuisance',
            'full_runtime_surface_quality_or_likeness_certified':False}
    result['contract_binding_sha256']=digest_json(result)
    path=out/'asset_contract.json';_save(path,result)
    return {'path':str(path.resolve()),'sha256':sha(path),'revision':REVISION,'mesh_count':len(entries)}


class AssetProvider:
    def __init__(self, contract_path, contract_sha256, contract, source, *, _capability):
        require(_capability is _CAPABILITY,'Use prepare_asset_provider; JSON trust tokens refused')
        self._contract_path=str(Path(contract_path).resolve());self._contract_sha=contract_sha256
        self._contract=copy.deepcopy(contract);self._digest=digest_json(contract)
        # The large original payload is pinned by its actual file SHA, not
        # repeatedly serialized as a mutable trusted in-memory target.
        self._by_path={e['renderer_path']:e for e in self._contract['meshes']}
        require(len(self._by_path)==len(contract['meshes']),'Duplicate source renderer paths')

    def __reduce__(self):raise TypeError('Prepare new nonserializable asset provider from frozen source contract')

    def _check(self):
        require(sha(self._contract_path)==self._contract_sha and digest_json(self._contract)==self._digest
                ,'Asset contract/context mutated')
        check_files(self._contract['source_files']);check_files(self._contract['stage_files'])

    def verify_snapshot_assets(self,snapshot):
        self._check();c=self._contract
        require(snapshot['frame_count']==snapshot['frame_count_end'],'Frame-unstable geometry')
        require(type(snapshot['character']['head_id']) is int and snapshot['character']['head_id']==c['head_id']
                and type(snapshot['character']['transform_id']) is int and snapshot['character']['transform_id']==c['source_actor_id'], 'Source head/actor asset binding changed')
        require(snapshot['native_face_drivers']['bridge_mvid']==c['source_bridge_mvid']
                and snapshot['game']['game_assembly_sha256']==c['source_game_assembly_sha256'], 'Source bridge/game assembly binding changed')
        require(np.array_equal(np.asarray(snapshot['character']['shape_value_face'],np.float32),np.asarray(c['native59'],np.float32)), 'Bound native59 changed')
        meshes,skipped=select_head(snapshot)
        require({m['renderer_path'] for m in meshes}==set(self._by_path),'Visible source renderer set changed')
        checks={}
        for mesh in meshes:
            e=self._by_path[mesh['renderer_path']]
            require(all(type(mesh[k]) is int for k in ('renderer_id','renderer_transform_id','mesh_instance_id'))
                    and all(type(v) is int for v in mesh['bone_transform_ids']), 'Exact renderer/skin bone IDs required')
            for k in ('mesh_name','renderer_id','renderer_transform_id','mesh_instance_id','source_geometry_sha256','bone_names','bone_transform_ids'):
                require(mesh[k]==e[k],'Actual source asset identity/palette changed: '+k)
            require(_mesh_inventory(mesh)==e['source_inventory'], 'Actual source arrays/blendshape frames/active weights changed '+mesh['mesh_name'])
            checks[mesh['renderer_path']]={'passed':True,'mesh_name':mesh['mesh_name'],'origin_kind':e['origin']['kind'],
                'source_geometry_sha256':e['source_geometry_sha256'],'authored_frames_and_source_weights_bound':True}
        return {'passed':True,'revision':REVISION,'contract_sha256':self._contract_sha,'meshes':checks,'skipped':skipped,
                'target_generated_from_candidate':False,'native_or_transform_pose_checked_separately':True}

    def cached_mesh(self,rig,source_mesh):
        self._check();require(rig.head_id==self._contract['head_id'] and rig.sampling_profile==PROFILE,'Wrong rig head/profile')
        e=self._by_path.get(source_mesh['renderer_path']);require(e is not None,'Renderer absent from explicit provider')
        require(_mesh_inventory(source_mesh)==e['source_inventory'] and source_mesh['source_geometry_sha256']==e['source_geometry_sha256'], 'Provider source mesh changed')
        with np.load(e['stage_npz'],allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
        sub=copy.copy(rig)
        for field,key in (('verts','verts'),('faces','faces'),('bone_idx','bone_idx'),('bone_w','bone_w'),('bindpose','bindpose')):setattr(sub,field,data[key])
        sub.skin_bone_names=[rig.bones[str(pid)]['name'] for pid in data['skin_bone_pids']]
        from tools.abmx_replay.validate_geometry import cache_correspondence
        cache_correspondence(sub,source_mesh)
        return sub

    def _math_function(self,name):
        # Isolated globals: no monkeypatch or mutation of the frozen validator.
        from tools.native_radial_target import validate_runtime as math
        g=dict(math.__dict__);g['cached_mesh']=self.cached_mesh
        independent=types.FunctionType(math.independent_target.__code__,g,math.independent_target.__name__,
                                      math.independent_target.__defaults__,math.independent_target.__closure__)
        independent.__kwdefaults__=math.independent_target.__kwdefaults__
        g['independent_target']=independent
        original_certify=math.certify
        def certify(snapshot,descriptor):
            self.verify_snapshot_assets(snapshot)
            return original_certify(snapshot,descriptor)
        g['certify']=certify
        original=getattr(math,name)
        function=types.FunctionType(original.__code__,g,original.__name__,original.__defaults__,original.__closure__)
        function.__kwdefaults__=original.__kwdefaults__
        return function

    def independent_target(self,artifact,contract,*,source_gaze_policy='reject'):
        self._check();h=artifact['compiler_inputs']['history']
        require(sha(h['geometry']['path'])==self._contract['source_history']['geometry']['sha256']
                and sha(h['trace'])==self._contract['source_history']['trace_sha256'], 'Compiler/provider earlier source differs')
        target=self._math_function('independent_target')(artifact,contract,source_gaze_policy=source_gaze_policy)
        require(all(r['native_source_model_passed'] for r in target['evidence']['source_mesh_checks'].values()),
                'Full visible source baseline FK/LBS not verified at original tolerance')
        target['evidence']['explicit_asset_provider']={'revision':REVISION,'contract_path':self._contract_path,
            'contract_sha256':self._contract_sha,'frozen_math_module_unmodified':True,'default_cache_fallback':False}
        return target

    def runtime_audit(self,manifest_path,abmx_contract_path,out_dir,*,source_gaze_policy='reject'):
        self._check()
        manifest=read(manifest_path);descriptor=manifest.get('asset_provider')
        require(type(descriptor) is dict and str(Path(descriptor['path']).resolve())==self._contract_path
                and descriptor['sha256']==self._contract_sha, 'Manifest/provider descriptor path or SHA mismatch')
        # The target function must enforce same earlier source even in audit.
        from tools.native_radial_target import validate_runtime as math
        fn=self._math_function('audit');fn.__globals__['independent_target']=self.independent_target
        report=fn(manifest_path,abmx_contract_path,out_dir,source_gaze_policy=source_gaze_policy)
        evidence={'revision':REVISION,'contract_path':self._contract_path,'contract_sha256':self._contract_sha,
                  'provider_implementation_sha256':sha(__file__),'unchanged_math_implementation_sha256':sha(math.__file__),
                  'explicit_new_wrapper_revision':True,'not_an_original_frozen_validator_certification':True,
                  'default_cache_fallback':False,'runtime_math_passed':report['passed']}
        _save(Path(out_dir)/'asset_provider_revision.json',evidence)
        return {**report,'explicit_asset_provider_revision':evidence}


def _revalidate_contract_sources(c):
    """Reopen authored assets before observer start; self-digest is never sufficient."""
    history=c['source_history'];snapshot=read(history['geometry']['path']);trace=read(history['trace'])
    require(sha(history['geometry']['path'])==history['geometry']['sha256'] and sha(history['trace'])==history['trace_sha256'],
            'Frozen identity source hash mismatch')
    require(trace['active'] is False and trace['trace_complete'] is True, 'Source is not stopped complete history')
    require(sha(c['abmx_contract']['path'])==c['abmx_contract']['sha256'], 'Bound installed replay contract changed')
    _,cursor,replay=select_cursor_predictions(snapshot,trace,read(c['abmx_contract']['path']),required_names=NAMES)
    require(cursor==c['source_cursor'] and not replay['inter_call_boundaries'], 'Source cursor or ordered identity evidence differs')
    for event in trace['events']:
        require(serialized(modifier(event['resolved_modifier']))==serialized(modifier(IDENTITY)) and not event['additional_modifiers'],
                'Candidate/nonidentity source contract rejected')
    for bone in snapshot['abmx_runtime']['bones']:
        require(serialized(modifier({k:bone[k] for k in IDENTITY}))==serialized(modifier(IDENTITY)), 'Candidate source geometry rejected')
        if bone['name'] in NAMES:_clean(bone['runtime_baseline']['fields'])
    require(snapshot['character']['head_id']==c['head_id'] and snapshot['character']['transform_id']==c['source_actor_id'], 'Contract/source head or actor mismatch')
    require(c['sampling_profile']==PROFILE and c['source_bridge_mvid']==trace['metadata']['bridge_mvid']
            and snapshot['native_face_drivers']['bridge_mvid']==c['source_bridge_mvid']
            and c['source_game_assembly_sha256']==snapshot['game']['game_assembly_sha256'], 'Declared source profile/bridge/game differs')
    require(np.array_equal(np.asarray(snapshot['character']['shape_value_face'],np.float32),np.asarray(c['native59'],np.float32)), 'Contract/source native differs')
    rig=HeadRig(c['head_id'],sampling_profile=PROFILE)
    list_path=ROOT/'data/hs2_head/chalist_cache.json';lists=read(list_path);row=lists['by_cat']['210'][str(c['head_id'])]
    bundle=Path(lists['ab_dir'])/row['MainAB'];prefab=row['MainData'];defaults=_asset_rows(UnityPy.load(str(bundle)),prefab)
    mandatory={str(Path(history['geometry']['path']).resolve()):history['geometry']['sha256'],
               str(Path(history['trace']).resolve()):history['trace_sha256'],
               str(Path(c['abmx_contract']['path']).resolve()):c['abmx_contract']['sha256'],
               str(list_path.resolve()):sha(list_path),str(bundle.resolve()):sha(bundle),
               str(Path(__file__).resolve()):sha(__file__),**native_source_bindings(rig)}
    for relative in ('tools/native_radial_target/audit_source_assets.py','scripts/hs2_extract_head.py','tools/native_radial_target/validate_runtime.py'):
        mandatory[str((ROOT/relative).resolve())]=sha(ROOT/relative)
    require(all(c['source_files'].get(p)==digest for p,digest in mandatory.items()), 'Mandatory frozen implementation/native/history/asset source dependency missing or changed')
    source_by_path={m['renderer_path']:m for m in select_head(snapshot)[0]}
    require({e['renderer_path'] for e in c['meshes']}==set(source_by_path), 'Contract source renderer set differs')
    for entry in c['meshes']:
        mesh=source_by_path[entry['renderer_path']];origin=entry['origin']
        if origin['kind']=='explicit_zipmod':
            override=c['explicit_overrides'].get(entry['renderer_path'])
            require(override is not None and origin=={'kind':'explicit_zipmod',**override}
                    and c['head_id']==3 and entry['mesh_name']=='o_tooth', 'Explicit override declaration differs')
            require(origin['prefab']==prefab and origin['member']=='abdata/'+row['MainAB'] and sha(origin['archive'])==origin['archive_sha256'], 'Wrong authored archive/head binding')
            require(c['source_files'].get(str(Path(origin['archive']).resolve()))==origin['archive_sha256'], 'Archive dependency removed from pinned source list')
            with zipfile.ZipFile(origin['archive']) as z:blob=z.read(origin['member'])
            require(hashlib.sha256(blob).hexdigest()==origin['member_sha256'], 'Wrong authored member hash')
            asset=_origin_candidate(_asset_rows(UnityPy.load(blob),prefab),mesh,origin)
            expected_data=asset['data']
        else:
            require(origin['kind']=='existing_same_head_cache' and entry['renderer_path'] not in c['explicit_overrides'], 'Unsupported or silent cache fallback')
            require(origin['authored_bundle']==str(bundle.resolve()) and origin['authored_bundle_sha256']==sha(bundle)
                    and origin['prefab']==prefab, 'Wrong default authored head source')
            asset=_origin_candidate(defaults,mesh,origin)
            expected_path=Path(rig.data_dir)/('o_head_mesh.npz' if mesh['mesh_name']=='o_head' else 'submeshes/'+mesh['mesh_name']+'.npz')
            require(str(Path(origin['npz']).resolve())==str(expected_path.resolve()) and sha(expected_path)==origin['npz_sha256'], 'Wrong head cache origin')
            require(c['source_files'].get(str(expected_path.resolve()))==origin['npz_sha256'], 'Cache dependency removed from pinned source list')
            with np.load(expected_path,allow_pickle=False) as z:expected_data={k:np.array(z[k]) for k in z.files}
        authored=_authored_check(mesh,asset)
        require(authored==entry['authored_correspondence'], 'Authored comparison claims differ from actual reopened source')
        expected_data=dict(expected_data);expected_data['skin_bone_pids']=np.asarray([rig.name2pid[name] for name in mesh['bone_names']])
        with np.load(entry['stage_npz'],allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
        require(set(data)==set(expected_data) and all(np.array_equal(data[k],expected_data[k]) for k in data), 'Staged arrays differ from reopened original authored/cache sources')
        inventory={k:_array_digest(v,np.int64 if k in ('faces','bone_idx') else np.float32) for k,v in data.items() if k!='skin_bone_pids'}
        require(inventory==entry['all_staged_arrays'] and data['skin_bone_pids'].tolist()==entry['skin_bone_pids'], 'Staged array inventory/palette mapping changed')
        require(c['stage_files'].get(entry['stage_npz'])==entry['stage_npz_sha256']==sha(entry['stage_npz'])
                and c['stage_files'].get(entry['blendshapes_path'])==entry['blendshapes_sha256']==sha(entry['blendshapes_path']),
                'Mandatory staged data dependency missing or changed')
        require(_shape_inventory(read(entry['blendshapes_path']),mesh['vertex_count'])==_shape_inventory(mesh['blendshapes'],mesh['vertex_count'])
                and [r['current_weight'] for r in read(entry['blendshapes_path'])]==[r['current_weight'] for r in mesh['blendshapes']], 'Staged source-only blendshape payload differs')
    return snapshot


def prepare_asset_provider(contract_path,contract_sha256):
    require(sha(contract_path)==contract_sha256,'Asset contract SHA mismatch')
    c=read(contract_path)
    require(c.get('schema_version')==1 and c.get('revision')==REVISION and c.get('actual_source_only') is True
            and c.get('candidate_inputs_used') is False,'Explicit identity source asset revision required')
    copy_contract=copy.deepcopy(c);binding=copy_contract.pop('contract_binding_sha256')
    require(digest_json(copy_contract)==binding,'Contract semantic payload changed')
    check_files(c['source_files']);check_files(c['stage_files'])
    source=_revalidate_contract_sources(c)
    provider=AssetProvider(contract_path,contract_sha256,c,source,_capability=_CAPABILITY)
    provider.verify_snapshot_assets(source)
    for mesh in select_head(source)[0]:
        provider.cached_mesh(HeadRig(c['head_id'],sampling_profile=PROFILE),mesh)
    return provider


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--history',required=True)
    p.add_argument('--abmx-contract',required=True);p.add_argument('--overrides',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();print(json.dumps(stage_from_identity(read(a.history),a.abmx_contract,read(a.overrides),a.out_dir)))


if __name__=='__main__':main()
