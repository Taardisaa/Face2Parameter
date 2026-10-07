"""Strict, source-bound FaceScape TU sample loader and dimensionless protocol.

Reports contain provenance/statistics, never the licensed full vertex arrays.
No pickle, candidate-dependent alignment, unit inference or legacy full_mesh.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RAW_KIND = "raw_external_full_head_v1"
SHAPE_KIND = "dimensionless_external_shape_v1"
MEMBER = "sample_tu_model/1_neutral.obj"
README_MEMBER = "sample_tu_model/readme"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def check(descriptor):
    require(type(descriptor) is dict and type(descriptor.get('path')) is str
            and type(descriptor.get('sha256')) is str and type(descriptor.get('bytes')) is int,
            'Typed absolute source descriptor required')
    path = Path(descriptor['path'])
    require(path.is_absolute() and binding(path) == descriptor, 'Source file bytes/path/hash differ')
    return path


def read(path):
    return json.loads(Path(path).read_bytes())


def exact(a, b):
    require(type(a) is type(b), 'Contract declaration type differs')
    if isinstance(a, dict):
        require(a.keys() == b.keys(), 'Contract keys differ')
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, list):
        require(len(a) == len(b), 'Contract length differs')
        for x, y in zip(a, b):
            exact(x, y)
    else:
        require(a == b, 'Contract declaration differs')


def freeze(path, value):
    path = Path(path).resolve()
    require(not path.exists(), 'New frozen contract path required')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    return binding(path)


def array_digest(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()


def readonly(value, dtype):
    array = np.asarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RawTarget:
    vertices: np.ndarray
    faces: np.ndarray
    texcoords: np.ndarray
    normals: np.ndarray
    face_uv_indices: np.ndarray
    face_normal_indices: np.ndarray
    face_obj_signed_indices: np.ndarray
    face_line_numbers: np.ndarray
    declarations: tuple


def parse_obj(path):
    """Preserve order and independent v/vt/vn indices; no weld/triangulate/crop."""
    vertices, uvs, normals, faces, fuv, fnorm, signed, face_lines, declarations = ([] for _ in range(9))
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            tokens = line.split('#', 1)[0].split()
            if not tokens:
                continue
            tag, values = tokens[0], tokens[1:]
            if tag in ('v', 'vt', 'vn'):
                require(len(values) == (2 if tag == 'vt' else 3), 'Unsupported vertex/UV/normal arity')
                row = [float(v) for v in values]
                require(np.isfinite(row).all(), 'Nonfinite authored OBJ array')
                {'v': vertices, 'vt': uvs, 'vn': normals}[tag].append(row)
            elif tag == 'f':
                require(len(values) == 3, 'Nontriangle OBJ face: no automatic triangulation')
                resolved, originals = [], []
                for token in values:
                    pieces = token.split('/')
                    require(1 <= len(pieces) <= 3 and pieces[0], 'Malformed face corner')
                    pieces += [''] * (3-len(pieces))
                    row, original = [], []
                    for column, (text, count) in enumerate(zip(pieces, [len(vertices), len(uvs), len(normals)])):
                        if not text:
                            require(column > 0, 'Missing vertex index')
                            row.append(-1); original.append(0)
                        else:
                            # Reject float/bool-like textual indices; exact decimal integer only.
                            require(text.lstrip('+-').isdigit(), 'Invalid OBJ integer index')
                            index = int(text)
                            require(index != 0, 'OBJ indices are nonzero')
                            converted = index-1 if index > 0 else count+index
                            require(0 <= converted < count, 'OBJ index out of range/forward reference')
                            row.append(converted); original.append(index)
                    resolved.append(row); originals.append(original)
                require(len({r[0] for r in resolved}) == 3, 'Repeated vertex index in triangle')
                faces.append([r[0] for r in resolved]); fuv.append([r[1] for r in resolved]); fnorm.append([r[2] for r in resolved])
                signed.append(originals); face_lines.append(line_number)
            elif tag in ('mtllib', 'usemtl', 'o', 'g', 's'):
                declarations.append((line_number, tag, tuple(values)))
            else:
                raise ValueError('Unsupported OBJ directive: '+tag)
    require(vertices and faces, 'Complete nonempty vertex and triangle surface required')
    return RawTarget(readonly(vertices, '<f8'), readonly(faces, '<i8'), readonly(np.asarray(uvs).reshape(-1,2), '<f8'),
                     readonly(np.asarray(normals).reshape(-1,3), '<f8'), readonly(fuv, '<i8'), readonly(fnorm, '<i8'),
                     readonly(signed, '<i8'), readonly(face_lines, '<i8'), tuple(declarations))


def surface_moments(vertices, faces):
    """Exact area-integrated centroid/RMS radius, independent of vertex density."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all(), 'Finite Nx3 surface required')
    require(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in 'iu' and len(faces)
            and faces.min() >= 0 and faces.max() < len(vertices), 'Ordered valid triangle topology required')
    t = vertices[faces]
    areas = np.linalg.norm(np.cross(t[:,1]-t[:,0], t[:,2]-t[:,0]), axis=1)/2
    total = float(areas.sum())
    require(np.isfinite(total) and total > 0, 'Positive finite total surface area required')
    center = np.sum(areas[:,None]*t.mean(axis=1), axis=0)/total
    centered = t-center
    second = ((centered*centered).sum(axis=(1,2)) + (centered[:,0]*centered[:,1]).sum(1)
              + (centered[:,0]*centered[:,2]).sum(1) + (centered[:,1]*centered[:,2]).sum(1))/6
    radius = float(np.sqrt(np.sum(areas*second)/total))
    require(np.isfinite(radius) and radius > 0, 'Positive finite exact RMS radius required')
    return {'area':total, 'centroid':center.tolist(), 'rms_radius':radius,
            'zero_area_triangle_count':int(np.sum(areas == 0)),
            'method':'exact triangle surface integrals; all faces retained; no per-vertex density weighting'}


def target_statistics(target):
    arrays = {name:array_digest(getattr(target,name),'<f8' if name in ('vertices','texcoords','normals') else '<i8')
              for name in ['vertices','faces','texcoords','normals','face_uv_indices','face_normal_indices','face_obj_signed_indices','face_line_numbers']}
    return {'vertex_count':len(target.vertices), 'triangle_count':len(target.faces),
            'texture_coordinate_count':len(target.texcoords), 'normal_count':len(target.normals),
            'bbox_min':target.vertices.min(0).tolist(), 'bbox_max':target.vertices.max(0).tolist(),
            'arrays_sha256':arrays, 'surface_moments':surface_moments(target.vertices,target.faces),
            'face_index_convention':'zero-based resolved v/vt/vn separately; original signed OBJ v/vt/vn corner indices also retained; -1 missing UV/normal',
            'declarations':[[line,tag,list(tokens)] for line,tag,tokens in target.declarations]}


def verify_acquisition(obj_path, readme_path, receipt_path, extraction_path):
    receipt, extraction = read(receipt_path), read(extraction_path)
    archive = Path(receipt['archive'])
    require(type(receipt['bytes']) is int and receipt['bytes'] == archive.stat().st_size
            and sha(archive) == receipt['sha256'], 'Archive acquisition bytes/SHA differ')
    files=[]
    for member, path in [(MEMBER,Path(obj_path).resolve()),(README_MEMBER,Path(readme_path).resolve())]:
        matches=[r for r in extraction['asset_receipts'] if r['member']==member]
        require(len(matches)==1, 'Unique neutral TU member receipt required')
        expected=matches[0]
        require(Path(expected['path']).resolve()==path and expected['sha256']==sha(path)
                and type(expected['bytes']) is int and expected['bytes']==path.stat().st_size,
                'Extracted member receipt differs')
        with tarfile.open(archive,'r:gz') as tar:
            found=tar.getmember(member)
            require(found.isfile() and found.size==path.stat().st_size, 'Expected regular archive member')
            data=tar.extractfile(found)
            require(hashlib.sha256(data.read()).hexdigest()==expected['sha256'], 'Archive member/extracted bytes differ')
        files.append({**binding(path),'archive_member':member})
    license_text=Path(readme_path).read_text(encoding='utf-8')
    require('Please do NOT distribute' in license_text and 'TU-model' in license_text, 'Expected source sample license/TU role')
    return {'archive':binding(archive),'receipt':binding(receipt_path),'extraction_receipt':binding(extraction_path),
            'obj':files[0],'readme':files[1]}


def raw_contract(obj_path, readme_path, receipt_path, extraction_path):
    provenance=verify_acquisition(obj_path,readme_path,receipt_path,extraction_path)
    target=parse_obj(obj_path)
    extraction=read(extraction_path)
    require(len(target.vertices)==extraction['TU_vertices'] and len(target.faces)==extraction['TU_faces'], 'Neutral TU counts differ')
    return {'schema_version':1,'kind':RAW_KIND,'provenance':provenance,'loader':binding(__file__),
            'units':'unknown_raw_sample_units','coordinate_frame':'authored_OBJ_axes_unmodified_unknown_physical_registration',
            'sample_identity':{'subject_id':None,'expression':'neutral','filename_prefix_semantics':'expression example index, not asserted subject ID'},
            'target_role':'registered TU base mesh; not high-detail raw scan or calibrated HS2 surface',
            'surface_scope':'all authored triangles/vertices; no region mask, crop, welding, decimation, displacement or topology change',
            'anger_tuple_correspondence':'not asserted; raw anger PLY/cameras/images excluded',
            'redistribution':'Please do NOT distribute; raw data stay in local gitignored tools/zips; contract contains no full arrays',
            'statistics':target_statistics(target),'legacy_full_mesh_compatible':False,'human_likeness_acceptance_defined':False}


def freeze_raw_target(obj_path, readme_path, receipt_path, extraction_path, out_contract):
    return freeze(out_contract,raw_contract(obj_path,readme_path,receipt_path,extraction_path))


def load_raw_target(contract_path, expected_sha256):
    require(sha(contract_path)==expected_sha256, 'Frozen raw target contract SHA differs')
    contract=read(contract_path)
    require(type(contract.get('schema_version')) is int and contract['schema_version']==1 and contract.get('kind')==RAW_KIND,
            'Explicit raw_external_full_head_v1 only; legacy contracts refused')
    p=contract['provenance']
    check(contract['loader'])
    for key in ['archive','receipt','extraction_receipt']:
        check(p[key])
    for key in ['obj','readme']:
        check({k:v for k,v in p[key].items() if k!='archive_member'})
    rebuilt=raw_contract(p['obj']['path'],p['readme']['path'],p['receipt']['path'],p['extraction_receipt']['path'])
    exact(contract,rebuilt)
    return parse_obj(p['obj']['path'])


def proper_rigid(matrix):
    matrix=np.asarray(matrix,dtype=np.float64)
    require(matrix.shape==(4,4) and np.isfinite(matrix).all() and np.allclose(matrix[3],[0,0,0,1],atol=1e-12,rtol=0), 'Declared proper rigid matrix required')
    r=matrix[:3,:3]
    require(np.allclose(r.T@r,np.eye(3),atol=1e-10,rtol=0) and abs(np.linalg.det(r)-1)<1e-10,
            'No scale, reflection or affine alignment in rigid declaration')
    return matrix


def geometry_hash(vertices, faces):
    digest=hashlib.sha256()
    for array,dtype in [(vertices,'<f8'),(faces,'<i8')]:
        array=np.asarray(array,dtype=dtype);digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def load_common_reference(reference_path, expected_sha256):
    require(sha(reference_path)==expected_sha256,'Frozen common-reference SHA differs')
    source=read(reference_path);metadata=source['metadata']
    require(source.get('kind')=='full_mesh' and type(metadata['head_id']) is int and metadata['head_id']==0
            and metadata['native59']==[.5]*59 and metadata['abmx']=={} and metadata['sampling_profile']=='vanilla'
            and metadata['units']=='hs2_cache_units' and metadata['coordinate_frame']=='hs2_cached_head_fk',
            'Exact head0 neutral shared reference required, not target-person contract')
    require(source['extra_groups']==[] and 'reference only' in source['reference_role'],'Declared reference scope required')
    exact(metadata['scope'],{'surface_group':'o_head','expression_blendshapes':'excluded','external_ancestors':'excluded','body_pose':'cached_rest'})
    vertices=np.asarray(source['vertices'],dtype=np.float64);faces=np.asarray(source['faces'])
    moments=surface_moments(vertices,faces)
    require(geometry_hash(vertices,faces)==metadata['content_sha256']
            and array_digest(faces,'<i8')==metadata['topology_sha256'],'Reference content/topology digest differs')
    for row in metadata['asset']['files']:
        require(sha(row['path'])==row['sha256'],'Reference cached asset changed')
    for path,digest in source['reference_helper_sources'].items():
        require(sha(path)==digest,'Reference helper changed')
    return vertices,faces,{'descriptor':binding(reference_path),'head_id':0,'native59':[.5]*59,
                           'reference_metadata':metadata,'reference_helper_sources':source['reference_helper_sources'],
                           'vertices_sha256':array_digest(vertices,'<f8'),'faces_sha256':array_digest(faces,'<i8'),
                           'surface_moments':moments}


def freeze_dimensionless_protocol(raw_contract_path, raw_sha256, reference_path, reference_sha256,
                                  target_rigid, orientation_provenance, out_contract):
    target=load_raw_target(raw_contract_path,raw_sha256)
    matrix=proper_rigid(target_rigid)
    require(type(orientation_provenance) is str and orientation_provenance.strip(), 'Earlier explicit orientation provenance required')
    _,_,reference=load_common_reference(reference_path,reference_sha256)
    contract={'schema_version':1,'kind':SHAPE_KIND,'raw_target_contract':binding(raw_contract_path),'loader':binding(__file__),
              'target_rigid':matrix.tolist(),'orientation_provenance':orientation_provenance,
              'target_normalization':surface_moments(target.vertices,target.faces),'hs2_common_reference':reference,
              'normalization_rule':'target=(raw-target_area_centroid) rotated by declared proper rigid / target_area_RMS; candidates=(vertices-fixed_head0_area_centroid)/fixed_head0_area_RMS; rigid translation cancels through centroid',
              'units':'dimensionless_fixed_protocol_not_mm_or_HS2_units','scope':'entire target TU surface and complete cached HS2 o_head; semantic equivalence of surface groups not certified',
              'candidate_rules':{'per_candidate_scale_fit':False,'per_base_scale_fit':False,'per_candidate_rigid_fit':False,
                                 'per_base_rigid_fit':False,'candidate_recentring':False,'target_changes_during_search':False,
                                 'candidate_head_ids':[0,1,2,3],'target_crop_or_mask':False},
              'physical_distance_tolerances_inherited':False,'distance_acceptance_threshold':None,
              'intended_metric':'common area-weighted bidirectional full-surface RMS/P95/max and normals; not implemented search or anatomy proof',
              'pose_orientation_semantically_certified':False,'raw_anger_photos_or_cameras_used':False}
    return freeze(out_contract,contract)


def load_dimensionless_target(protocol_path, expected_sha256):
    require(sha(protocol_path)==expected_sha256, 'Frozen dimensionless contract SHA differs')
    protocol=read(protocol_path)
    require(protocol.get('kind')==SHAPE_KIND and type(protocol.get('schema_version')) is int and protocol['schema_version']==1,
            'Explicit new dimensionless contract only; old full_mesh refused')
    check(protocol['loader']);raw_path=check(protocol['raw_target_contract'])
    raw=load_raw_target(raw_path,protocol['raw_target_contract']['sha256'])
    exact(protocol['target_normalization'],surface_moments(raw.vertices,raw.faces))
    reference_descriptor=protocol['hs2_common_reference']['descriptor'];check(reference_descriptor)
    _,_,reference=load_common_reference(reference_descriptor['path'],reference_descriptor['sha256']);exact(protocol['hs2_common_reference'],reference)
    expected_rules={'per_candidate_scale_fit':False,'per_base_scale_fit':False,'per_candidate_rigid_fit':False,
                    'per_base_rigid_fit':False,'candidate_recentring':False,'target_changes_during_search':False,
                    'candidate_head_ids':[0,1,2,3],'target_crop_or_mask':False}
    exact(protocol['candidate_rules'],expected_rules)
    require(protocol['units']=='dimensionless_fixed_protocol_not_mm_or_HS2_units'
            and protocol['physical_distance_tolerances_inherited'] is False and protocol['distance_acceptance_threshold'] is None,
            'Units/physical tolerance/acceptance claim changed')
    require(type(protocol['orientation_provenance']) is str and protocol['orientation_provenance'].strip()
            and protocol['pose_orientation_semantically_certified'] is False and protocol['raw_anger_photos_or_cameras_used'] is False,
            'Undeclared semantic calibration or cross-sample correspondence')
    matrix=proper_rigid(protocol['target_rigid']);norm=protocol['target_normalization']
    vertices=(raw.vertices-np.asarray(norm['centroid'])) @ matrix[:3,:3].T / norm['rms_radius']
    return readonly(vertices,'<f8'),raw.faces,protocol


def _common_coordinates(vertices, protocol):
    """Numeric primitive, not a candidate provenance certificate."""
    require(protocol.get('kind')==SHAPE_KIND,'Explicit new protocol required')
    vertices=np.asarray(vertices,dtype=np.float64)
    require(vertices.ndim==2 and vertices.shape[1]==3 and np.isfinite(vertices).all(),'Finite complete candidate array required')
    reference=protocol['hs2_common_reference']['surface_moments']
    return (vertices-np.asarray(reference['centroid']))/reference['rms_radius']


def _validated_candidate_conversion(vertices, faces, protocol, candidate_metadata):
    """Validated new-contract conversion, fixed shared head0 anchor/scalar."""
    require(type(candidate_metadata['head_id']) is int and candidate_metadata['head_id'] in [0,1,2,3]
            and candidate_metadata['units']=='hs2_cache_units' and candidate_metadata['coordinate_frame']=='hs2_cached_head_fk',
            'Unknown candidate head/units/frame refused')
    exact(candidate_metadata['scope'],{'surface_group':'o_head','expression_blendshapes':'excluded','external_ancestors':'excluded','body_pose':'cached_rest'})
    require(candidate_metadata['sampling_profile'] in ['vanilla','slider_unlocker_18_2'],'Unknown native sampling profile')
    native=candidate_metadata.get('native59')
    require(type(native) is list and len(native)==59 and all(type(v) in (int,float) for v in native)
            and np.isfinite(native).all(),'Complete finite native59 provenance required')
    require(candidate_metadata['content_sha256']==geometry_hash(vertices,faces)
            and candidate_metadata['topology_sha256']==array_digest(faces,'<i8'),'Candidate full content/topology changed')
    require(candidate_metadata.get('pose_removal')=='none; cached FK frame, no fitted rotation/translation/scale',
            'Candidate-dependent rigid/scale fitting refused')
    asset=candidate_metadata['asset'];require(asset.get('provenance') and asset.get('files'),'Unknown candidate asset provenance')
    expected_names={'o_head_mesh.npz','skeleton.json','anmShapeHead.json','enums.json','customhead.json','update_eqns.json'}
    require({Path(row['path']).name for row in asset['files']}==expected_names and len(asset['files'])==6,
            'Mandatory cached head/equation dependencies missing')
    expected_head='head_'+str(candidate_metadata['head_id'])
    for row in asset['files']:
        if Path(row['path']).name in {'o_head_mesh.npz','skeleton.json','anmShapeHead.json'}:
            require(Path(row['path']).parent.name==expected_head,'Candidate files bind another head')
    for row in asset['files']:
        require(sha(row['path'])==row['sha256'],'Candidate cached asset changed')
    surface_moments(vertices,faces)  # validation only; these moments do NOT alter common normalization.
    return readonly(_common_coordinates(vertices,protocol),'<f8'),readonly(faces,'<i8')


class PreparedComparison:
    """Process-local verified protocol; JSON objects cannot replace its binding."""
    def __init__(self, protocol_path, expected_sha256):
        vertices,faces,protocol=load_dimensionless_target(protocol_path,expected_sha256)
        self.target_vertices=vertices
        self.target_faces=faces
        self._protocol=copy.deepcopy(protocol)
        self._protocol_file=binding(protocol_path)
        self._source_files=[self._protocol_file,protocol['raw_target_contract'],protocol['loader'],
                            protocol['hs2_common_reference']['descriptor']]
        raw=read(protocol['raw_target_contract']['path'])
        self._source_files += [{k:v for k,v in raw['provenance'][key].items() if k!='archive_member'} for key in ['obj','readme','receipt','extraction_receipt']]

    @property
    def protocol(self):
        return copy.deepcopy(self._protocol)

    def transform_candidate(self,vertices,faces,candidate_metadata):
        for descriptor in self._source_files:
            check(descriptor)
        return _validated_candidate_conversion(vertices,faces,self._protocol,candidate_metadata)


def prepare_dimensionless_comparison(protocol_path, expected_sha256):
    return PreparedComparison(protocol_path,expected_sha256)


def transform_candidate(vertices,faces,context,candidate_metadata):
    require(isinstance(context,PreparedComparison),'A source-verified process-local new protocol context is required; JSON/old contracts refused')
    return context.transform_candidate(vertices,faces,candidate_metadata)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obj',type=Path,required=True);parser.add_argument('--readme',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--extraction',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True);parser.add_argument('--freeze-dimensionless',action='store_true')
    parser.add_argument('--reference',type=Path);parser.add_argument('--reference-sha256')
    args=parser.parse_args();require(not args.out_dir.exists(),'New output directory required')
    raw=freeze_raw_target(args.obj,args.readme,args.receipt,args.extraction,args.out_dir/'raw_target_contract.json')
    target=load_raw_target(raw['path'],raw['sha256'])
    result={'raw_contract':raw,'loader_verified_counts':[len(target.vertices),len(target.faces)],'full_arrays_serialized':False}
    if args.freeze_dimensionless:
        require(args.reference is not None and args.reference_sha256 is not None,'Explicit frozen common reference path/SHA required')
        dim=freeze_dimensionless_protocol(raw['path'],raw['sha256'],args.reference,args.reference_sha256,np.eye(4),'Predeclared identity rotation, authored axes preserved; not estimated from target/candidate and anatomical orientation not certified',args.out_dir/'dimensionless_protocol.json')
        vertices,faces,protocol=load_dimensionless_target(dim['path'],dim['sha256'])
        reference,reference_faces,_=load_common_reference(args.reference,args.reference_sha256)
        normalized_reference=_common_coordinates(reference,protocol)
        result.update({'dimensionless_protocol':dim,'target_normalized_moments':surface_moments(vertices,faces),
                       'common_reference_normalized_moments':surface_moments(normalized_reference,reference_faces),
                       'comparison_or_optimizer_run':False,'unit_or_orientation_calibration_claimed':False})
    freeze(args.out_dir/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
