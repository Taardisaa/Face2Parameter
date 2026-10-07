"""Differentiable ALL30 native_radius control semantics, separate from ABMX.

Logical t=p_native*Length+offset. Installed physical Length is always one.
This intentionally differs from vanilla historical-radius first Apply.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from src.face_data_utils.utils import BONE_NAME_LIST
from src.hs2_deform_torch import TorchHeadRig,euler_zxy_quat,qmul
from src.hs2_mesh_deform import _fk_world,_qmul,euler_zxy_quat as numpy_euler
from tools.abmx_replay.model import IDENTITY,modifier,serialized

MODE='native_radial_target_v1'
PROFILE='slider_unlocker_18_2'
NAMES=tuple(BONE_NAME_LIST)
if len(NAMES)!=30 or len(set(NAMES))!=30:raise ValueError('Source ALL30 ordering changed')


def identity_patch_map():
    return {name:{k:list(v) if isinstance(v,list) else v for k,v in IDENTITY.items()} for name in NAMES}


def patch_array(logical_patches):
    if not isinstance(logical_patches,dict) or set(logical_patches)!=set(NAMES):
        raise ValueError('All30 explicit logical modifiers required; no absent-bone/static fallback')
    rows=[]
    for name in NAMES:
        m=modifier(logical_patches[name])
        if np.any(m['scale']<=0) or m['length']<=0:
            raise ValueError('Positive logical scale/Length quality policy required')
        rows.append(np.r_[m['scale'],m['length'],m['position'],m['rotation']])
    return np.asarray(rows,dtype=np.float32)


def assert_coverage(rig):
    for name in NAMES:
        if sum(b['name']==name for b in rig.bones.values())!=1:
            raise ValueError('Missing/ambiguous source bone: '+name)
    if len(rig._topo)!=len(set(rig._topo)) or set(rig._topo)!=set(rig.bones):
        raise ValueError('Invalid source topology')
    seen=set()
    for pid in rig._topo:
        parent=rig.bones[pid]['parent']
        if parent in rig.bones and parent not in seen:raise ValueError('Source skeleton is not parent-first')
        seen.add(pid)


def native_source_bindings(rig):
    from tools.abmx_replay.validate_trace import sha
    root=Path(__file__).resolve().parents[2]
    paths=[root/'src/hs2_mesh_deform.py',root/'src/hs2_deform_torch.py',root/'src/hs2_sampling.py',
        root/'src/face_data_utils/utils.py',root/'src/face_data_utils/parameter_flags.json',
        root/'data/hs2_head/enums.json',root/'data/hs2_head/customhead.json',root/'data/hs2_head/update_eqns.json',Path(__file__)]
    paths.extend(p for p in Path(rig.data_dir).rglob('*') if p.is_file())
    return {str(p.resolve()):sha(p) for p in paths}


def native_numpy(rig,native59):
    assert_coverage(rig)
    a=np.asarray(native59,dtype=np.float64)
    if a.shape!=(59,) or not np.isfinite(a).all():raise ValueError('Complete finite native59 required')
    world=_fk_world(rig,a,None)
    result={}
    for name in NAMES:
        pid=rig.name2pid[name];parent=rig.bones[pid]['parent']
        matrix=np.linalg.solve(world[parent],world[pid]) if parent in world else world[pid]
        scale=np.linalg.norm(matrix[:3,:3],axis=0)
        if np.any(scale<=0) or np.linalg.det(matrix[:3,:3])<=0:raise ValueError('Nonpositive/sheared native local quality policy')
        rot=matrix[:3,:3]/scale
        if not np.allclose(rot.T@rot,np.eye(3),atol=1e-8,rtol=0):raise ValueError('Native local shear unsupported')
        from scipy.spatial.transform import Rotation
        result[name]={'local_position':matrix[:3,3].tolist(),'local_rotation_xyzw':Rotation.from_matrix(rot).as_quat().tolist(),'local_scale':scale.tolist()}
    return result


def target_numpy(native_locals,logical_patches,*,semantic_mode):
    if semantic_mode!=MODE:raise ValueError('Explicit native_radial_target_v1 required')
    array=patch_array(logical_patches)
    if set(native_locals)!=set(NAMES):raise ValueError('Complete30 native locals required')
    result={}
    for name,m in zip(NAMES,array):
        n=native_locals[name]
        p=np.asarray(n['local_position'],dtype=np.float64)
        q=np.asarray(n['local_rotation_xyzw'],dtype=np.float64)
        s=np.asarray(n['local_scale'],dtype=np.float64)
        result[name]={'local_position':(p*float(m[3])+m[4:7].astype(float)).tolist(),
            # patch_array records installed float32 values; continuous reference
            # then evaluates them in float64, matching the declared Torch dtype.
            'local_rotation_xyzw':_qmul(q,numpy_euler(*map(float,m[7:]))).tolist(),
            'local_scale':(s*m[:3].astype(float)).tolist()}
    return result


class NativeRadialTargetRig(TorchHeadRig):
    """Explicit differentiable semantic adapter with full source ALL30 ordering.

    Native/keyframe arithmetic and continuous logical targets use declared dtype.
    Runtime compilation separately performs installed float32 lowering/replay.
    Inherited parent-first FK and skin() connect o_head and every cached submesh.
    """
    def __init__(self,rig,*,semantic_mode,source_bindings,device='cpu',dtype=torch.float64):
        if semantic_mode!=MODE or rig.sampling_profile!=PROFILE or dtype not in (torch.float32,torch.float64):
            raise ValueError('Explicit semantic mode/installed profile/float dtype required')
        if not source_bindings:raise ValueError('Source-bound native predictor required')
        from tools.abmx_stable_lowering.compiler import check_files
        check_files(source_bindings)
        if not set(native_source_bindings(rig)).issubset(source_bindings):raise ValueError('Incomplete native/core/ALL30 mask asset bindings')
        assert_coverage(rig)
        super().__init__(rig,device=device,dtype=dtype)
        if tuple(self.ab_names)!=NAMES or len(self.ab_slot)!=30:
            raise ValueError('All30 ordering/rig slot coverage required')
        self.semantic_mode=semantic_mode;self.source_bindings=dict(source_bindings)
        self.logical_indices=torch.tensor([self.name2bone[n] for n in NAMES],device=self.device,dtype=torch.long)

    def validate_sources(self):
        from tools.abmx_stable_lowering.compiler import check_files
        check_files(self.source_bindings)

    def logical_tensor(self,logical_patches,batch=1):
        if type(batch) is not int or batch<1:raise ValueError('Explicit positive batch required')
        return torch.as_tensor(patch_array(logical_patches),device=self.device,dtype=self.dtype)[None].expand(batch,-1,-1)

    def ab_tensor(self,*a,**kw):
        raise ValueError('Raw ABMX input ambiguous; use explicit logical_tensor in native_radial_target_v1')

    def from_card(self,*a,**kw):
        raise ValueError('Card inference needs explicit semantic mode,full59/base,logical30 and source context')

    def local_transforms(self,shape_face=None,ab=None):
        if (shape_face is None or shape_face.ndim!=2 or shape_face.shape[1]!=59 or shape_face.dtype!=self.dtype or
            shape_face.device!=self.device or not torch.isfinite(shape_face).all()):
            raise ValueError('Explicit full59 native tensor on declared device/dtype required')
        if ab is None or ab.shape!=(len(shape_face),30,10) or ab.dtype!=self.dtype or ab.device!=self.device or not torch.isfinite(ab).all():
            raise ValueError('Explicit complete ALL30 logical tensor on declared device/dtype required')
        if (ab[...,:3]<=0).any() or (ab[...,3]<=0).any():raise ValueError('Positive scale/Length policy required')
        pos,quat,scale=super().local_transforms(shape_face,None)
        ids=self.logical_indices
        p=pos.index_select(1,ids)*ab[...,3:4]+ab[...,4:7]
        s=scale.index_select(1,ids)*ab[...,:3]
        q=qmul(quat.index_select(1,ids),euler_zxy_quat(ab[...,7],ab[...,8],ab[...,9]))
        if any(not torch.isfinite(v).all() for v in (p,s,q)) or (s<=0).any():raise ValueError('Nonfinite/nonpositive logical target')
        return pos.index_copy(1,ids,p),quat.index_copy(1,ids,q),scale.index_copy(1,ids,s)
