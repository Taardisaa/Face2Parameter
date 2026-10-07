"""Strict V2 declaration/helper wrapper around immutable V1 asset mathematics.

No V1 source, cache, report or candidate target is rewritten. This adds exact
JSON declaration types and independently reconstructed local helper bindings.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import torch
import UnityPy

from tools.abmx_multibone.geometry import select_cursor_predictions
from tools.abmx_replay.model import serialized
from tools.abmx_replay.validate_trace import read, sha
from tools.abmx_stable_lowering.compiler import check_files, digest_json
from tools.native_radial_target import asset_provider as v1
from tools.native_radial_target.compiler import require

ROOT=Path(__file__).resolve().parents[2]
REVISION='identity_source_assets_strict_v2'
_CAPABILITY=object()
_HELPER_ROOTS=('tools/native_radial_target/asset_provider.py',
               'tools/native_radial_target/asset_provider_strict.py',
               'tools/native_radial_target/compiler.py',
               'tools/native_radial_target/validate_runtime.py',
               'tools/abmx_multibone/geometry.py',
               'tools/abmx_replay/model.py',
               'tools/abmx_replay/validate_trace.py',
               'tools/abmx_replay/validate_geometry.py',
               'tools/abmx_stable_lowering/compiler.py')


def exact_equal(actual,expected,label='declaration'):
    """Numerical equality never substitutes for exact recursive JSON types."""
    expected=serialized(expected)
    require(type(actual) is type(expected), 'Exact JSON type mismatch: '+label)
    if type(expected) is dict:
        require(set(actual)==set(expected) and all(type(k) is str for k in actual), 'Object keys differ: '+label)
        for k in expected:exact_equal(actual[k],expected[k],label+'.'+k)
    elif type(expected) is list:
        require(len(actual)==len(expected), 'List length differs: '+label)
        for i,(a,b) in enumerate(zip(actual,expected)):exact_equal(a,b,f'{label}[{i}]')
    else:
        require(actual==expected, 'Declaration value differs: '+label)


def _module_file(name):
    p=ROOT.joinpath(*name.split('.'))
    if p.with_suffix('.py').is_file():return p.with_suffix('.py').resolve()
    if (p/'__init__.py').is_file():return (p/'__init__.py').resolve()
    # Frozen runtime math deliberately inserts these local helper search paths.
    if '.' not in name:
        for base in ('tools/geometry_quality','tools/base_comparison'):
            candidate=ROOT/base/(name+'.py')
            if candidate.is_file():return candidate.resolve()
    return None


def helper_sources():
    """Conservative project-local import closure, independently reconstructed."""
    pending=[(ROOT/p).resolve() for p in _HELPER_ROOTS];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        require(path.is_file() and path.is_relative_to(ROOT), 'Required local helper missing')
        seen.add(path)
        rel=path.relative_to(ROOT).with_suffix('');package=list(rel.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                name='.'.join(prefix+([node.module] if node.module else []))
                names=[name]+[name+'.'+a.name if name else a.name for a in node.names]
            for name in names:
                imported=_module_file(name)
                if imported is not None and imported not in seen:pending.append(imported)
    return {str(p):sha(p) for p in sorted(seen)}


def _versions():
    return {'numpy':str(np.__version__),'torch':str(torch.__version__),'UnityPy':str(UnityPy.__version__)}


def _int(value,label,nonnegative=False):
    require(type(value) is int and (not nonnegative or value>=0), 'Exact integer required: '+label)


def _snapshot_types(snapshot):
    exact_equal(snapshot.get('schema_version'),1,'snapshot.schema_version')
    for k in ('frame_count','frame_count_end'):_int(snapshot[k],k,True)
    character=snapshot['character']
    for k in ('transform_id','head_root_transform_id','head_id'):_int(character[k],'character.'+k)
    require(type(character['shape_value_face']) is list and len(character['shape_value_face'])==59,
            'Exact full59 native list required')
    for value in character['shape_value_face']:
        require(type(value) in (float,int) and np.isfinite(value), 'Finite numeric native value, not bool, required')
    cursor=snapshot['abmx_trace_cursor']
    for k in ('frame','observed_calls','completed_calls','last_completed_sequence','pending_calls','dropped_events','observer_error_count'):
        _int(cursor[k],'actual cursor.'+k,True)
    require(type(cursor['active']) is bool and type(cursor['session_id']) is str, 'Exact actual cursor bool/session required')
    for mesh in snapshot['meshes']:
        for k in ('renderer_id','renderer_transform_id','mesh_instance_id','object_id','vertex_count','triangle_count'):
            _int(mesh[k],'mesh.'+k,k.endswith('count'))
        require(type(mesh['enabled']) is bool and type(mesh['active_in_hierarchy']) is bool,'Exact renderer state booleans required')
        for value in mesh['bone_transform_ids']:_int(value,'actual skin bone ID')
    for t in snapshot['transforms']:
        _int(t['id'],'transform ID')
        if t['parent_id'] is not None:_int(t['parent_id'],'parent transform ID')


def _validate_v1_types(c):
    exact_equal(c['schema_version'],1,'V1.schema_version')
    exact_equal(c['revision'],v1.REVISION,'V1.revision')
    for k in ('actual_source_only','candidate_inputs_used','shared_cache_modified','global_zipmod_redirect'):
        exact_equal(c[k],k=='actual_source_only','V1.'+k)
    source=read(c['source_history']['geometry']['path']);trace=read(c['source_history']['trace'])
    _snapshot_types(source)
    _int(c['head_id'],'declared head_id');_int(c['source_actor_id'],'declared source_actor_id')
    exact_equal(c['head_id'],source['character']['head_id'],'V1.head_id')
    exact_equal(c['source_actor_id'],source['character']['transform_id'],'V1.source_actor_id')
    exact_equal(c['source_bridge_mvid'],trace['metadata']['bridge_mvid'],'V1.source_bridge_mvid')
    exact_equal(c['sampling_profile'],v1.PROFILE,'V1.sampling_profile')
    _,cursor,_=select_cursor_predictions(source,trace,read(c['abmx_contract']['path']),required_names=v1.NAMES)
    exact_equal(c['source_cursor'],cursor,'V1.source_cursor')
    by_path={m['renderer_path']:m for m in v1.select_head(source)[0]}
    for entry in c['meshes']:
        mesh=by_path[entry['renderer_path']]
        for k in ('mesh_name','renderer_path','renderer_id','renderer_transform_id','mesh_instance_id','source_geometry_sha256','bone_names','bone_transform_ids'):
            exact_equal(entry[k],mesh[k],'V1.mesh.'+k)
        exact_equal(entry['source_inventory'],v1._mesh_inventory(mesh),'V1.mesh.source_inventory')
        for k in ('renderer_path_id','mesh_path_id'):_int(entry['origin'][k],'authored.'+k)
    return source,cursor


def _descriptor(descriptor):
    require(type(descriptor) is dict and type(descriptor.get('path')) is str and type(descriptor.get('sha256')) is str,
            'Exact immutable V1 descriptor required')
    require(sha(descriptor['path'])==descriptor['sha256'],'V1 descriptor SHA mismatch')
    c=read(descriptor['path'])
    if 'revision' in descriptor:exact_equal(descriptor['revision'],v1.REVISION,'V1 descriptor revision')
    if 'mesh_count' in descriptor:exact_equal(descriptor['mesh_count'],len(c['meshes']),'V1 descriptor mesh_count')
    return {'path':str(Path(descriptor['path']).resolve()),'sha256':descriptor['sha256'],
            'revision':v1.REVISION,'mesh_count':len(c['meshes'])},c


def wrap_existing_v1(v1_descriptor,out_dir):
    """New source binding for a correctly typed historical V1, never retroactive V1 repair."""
    out=Path(out_dir);require(not out.exists(),'NEW strict contract output directory required')
    descriptor,c=_descriptor(v1_descriptor)
    source,cursor=_validate_v1_types(c)
    v1.prepare_asset_provider(descriptor['path'],descriptor['sha256'])
    helpers=helper_sources()
    result={'schema_version':2,'revision':REVISION,'v1_descriptor':descriptor,
            'source_files':helpers,'dependency_scope':'Conservative project-local Python import closure of provider, native compiler, replay, FK/LBS/digest and unchanged runtime mathematics; third-party code not hashed',
            'third_party_versions':_versions(),'source_binding':{'head_id':c['head_id'],'actor_id':c['source_actor_id'],
                'sampling_profile':c['sampling_profile'],'bridge_mvid':c['source_bridge_mvid'],'source_cursor':cursor,
                'source_geometry_sha256':sha(c['source_history']['geometry']['path'])},
            'actual_source_only':True,'candidate_target_inputs':False,'v1_frozen_files_modified':False}
    _snapshot_types(source)
    result['contract_binding_sha256']=digest_json(result)
    out.mkdir(parents=True);path=out/'strict_asset_contract.json';v1._save(path,result)
    return {'path':str(path.resolve()),'sha256':sha(path),'revision':REVISION,'mesh_count':descriptor['mesh_count']}


def stage_strict_from_identity(history,abmx_contract_path,overrides,out_dir):
    out=Path(out_dir);require(not out.exists(),'NEW strict stage directory required')
    require(sha(history['geometry']['path'])==history['geometry']['sha256'],'Source snapshot hash mismatch')
    _snapshot_types(read(history['geometry']['path']))
    descriptor=v1.stage_from_identity(history,abmx_contract_path,overrides,out/'v1_stage')
    return wrap_existing_v1(descriptor,out/'strict')


class StrictAssetProvider:
    def __init__(self,path,digest,c,provider,*,_capability):
        require(_capability is _CAPABILITY,'Use prepare_strict_asset_provider')
        self._path=str(Path(path).resolve());self._sha=digest;self._contract=copy.deepcopy(c)
        self._digest=digest_json(c);self._provider=provider

    def __reduce__(self):raise TypeError('Prepare strict asset context from immutable source contract')

    def _check(self):
        require(sha(self._path)==self._sha and digest_json(self._contract)==self._digest,'Strict contract/context mutated')
        check_files(self._contract['source_files'])
        exact_equal(self._contract['third_party_versions'],_versions(),'package versions')
        d=self._contract['v1_descriptor'];require(sha(d['path'])==d['sha256'],'Frozen nested V1 changed')

    def verify_snapshot_assets(self,snapshot):
        self._check();_snapshot_types(snapshot)
        result=self._provider.verify_snapshot_assets(snapshot)
        return {**result,'revision':REVISION,'strict_contract_sha256':self._sha,'exact_declaration_types':True,
                'complete_bound_project_helper_closure':True,'UnityPy_decoded_during_observer':False}

    def cached_mesh(self,rig,source_mesh):
        self._check();return self._provider.cached_mesh(rig,source_mesh)

    def independent_target(self,artifact,contract,*,source_gaze_policy='reject'):
        self._check();target=self._provider.independent_target(artifact,contract,source_gaze_policy=source_gaze_policy)
        target['evidence']['strict_asset_provider']={'revision':REVISION,'contract_path':self._path,
            'contract_sha256':self._sha,'nested_v1_descriptor':self._contract['v1_descriptor'],
            'candidate_after_target_inputs':False,'original_mathematical_tolerances_unchanged':True}
        return target

    def runtime_audit(self,manifest_path,abmx_contract_path,out_dir,*,source_gaze_policy='reject'):
        self._check();manifest=read(manifest_path);d=manifest.get('asset_provider')
        require(type(d) is dict and type(d.get('path')) is str and str(Path(d['path']).resolve())==self._path
                and type(d.get('sha256')) is str and d['sha256']==self._sha,'Actual manifest strict descriptor mismatch')
        if 'revision' in d:exact_equal(d['revision'],REVISION,'manifest revision')
        if 'mesh_count' in d:exact_equal(d['mesh_count'],self._contract['v1_descriptor']['mesh_count'],'manifest mesh_count')
        artifact=read(manifest['compiler_artifact']['path'])
        source=read(artifact['compiler_inputs']['history']['geometry']['path']);self.verify_snapshot_assets(source)
        for case in manifest['cases']:
            identity=case['identity_capture']['paired_geometry'];require(sha(identity['path'])==identity['sha256'],'Actual identity hash mismatch')
            self.verify_snapshot_assets(read(identity['path']))
            for window in case['windows']:
                g=window['geometry'];require(sha(g['path'])==g['sha256'],'Actual window hash mismatch')
                self.verify_snapshot_assets(read(g['path']))
        out=Path(out_dir);require(not out.exists(),'NEW strict runtime report directory required');out.mkdir(parents=True)
        # Only the provider descriptor changes in an explicit isolated copy.
        # Snapshot, trace, source, compiled patches and restoration data are untouched.
        temporary=copy.deepcopy(manifest);temporary['asset_provider']=self._contract['v1_descriptor']
        derived=out/'v1_math_manifest.json';v1._save(derived,temporary)
        result=self._provider.runtime_audit(derived,abmx_contract_path,out/'unchanged_math',source_gaze_policy=source_gaze_policy)
        strict={'revision':REVISION,'contract_path':self._path,'contract_sha256':self._sha,
                'original_manifest_path':str(Path(manifest_path).resolve()),'original_manifest_sha256':sha(manifest_path),
                'derived_manifest_path':str(derived.resolve()),'derived_manifest_sha256':sha(derived),
                'only_manifest_change':'asset_provider descriptor explicitly translated to bound nested V1',
                'posthoc_readonly':True,'v1_reports_or_sources_modified':False,'candidate_target_inputs':False}
        result={**result,'strict_asset_provider':strict,'manifest_path':strict['original_manifest_path'],
                'manifest_sha256':strict['original_manifest_sha256']}
        v1._save(out/'summary.json',result)
        return result


def prepare_strict_asset_provider(strict_contract_path,strict_contract_sha256):
    require(sha(strict_contract_path)==strict_contract_sha256,'Strict contract SHA mismatch')
    c=read(strict_contract_path);exact_equal(c.get('schema_version'),2,'strict schema')
    exact_equal(c.get('revision'),REVISION,'strict revision')
    for k in ('actual_source_only','candidate_target_inputs','v1_frozen_files_modified'):
        exact_equal(c[k],k=='actual_source_only','strict.'+k)
    payload=copy.deepcopy(c);binding=payload.pop('contract_binding_sha256')
    require(digest_json(payload)==binding,'Strict semantic contract payload changed')
    exact_equal(c['source_files'],helper_sources(),'independently reconstructed helper dependencies')
    check_files(c['source_files']);exact_equal(c['third_party_versions'],_versions(),'package versions')
    descriptor,original=_descriptor(c['v1_descriptor']);exact_equal(c['v1_descriptor'],descriptor,'canonical nested V1 descriptor')
    _,cursor=_validate_v1_types(original)
    expected={'head_id':original['head_id'],'actor_id':original['source_actor_id'],'sampling_profile':original['sampling_profile'],
              'bridge_mvid':original['source_bridge_mvid'],'source_cursor':cursor,'source_geometry_sha256':sha(original['source_history']['geometry']['path'])}
    exact_equal(c['source_binding'],expected,'strict source binding')
    provider=v1.prepare_asset_provider(descriptor['path'],descriptor['sha256'])
    return StrictAssetProvider(strict_contract_path,strict_contract_sha256,c,provider,_capability=_CAPABILITY)
