"""CPU head forward computation from recorded asset/pose/persistent state.

The installed ABMX Apply implementation is reused, including non-unit length,
private history, flags, ordered modifiers and rotation exclusions. This is a
fixed-pose head protocol; it does not predict an animation frame or shade AIT.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from src.hs2_mesh_deform import HeadRig, _fk_world, _trs
from src.hs2_sampling import sample_keyframes
from tools.abmx_replay.model import (CACHE_FIELDS, IDENTITY, replay_apply,
                                     serialized, unity_zero, unity_sqrmag, FLAG_FIELDS)
from tools.unity_parity.geometry import skin_world

PREFIX='KKABMX.Core.BoneModifier.'
FIELDS={'ScaleModifier':'scale','LengthModifier':'length',
        'PositionModifier':'position','RotationModifier':'rotation'}


def _data(value):
    return {short:copy.deepcopy(value['KKABMX.Core.BoneModifierData.'+long])
            for long,short in FIELDS.items()}


def reset(local, cache):
    """Installed BoneModifier.Reset; flags not silently cleared."""
    if not cache['_hasBaseline']:
        return
    local['local_scale']=copy.deepcopy(cache['_sclBaseline'])
    local['local_rotation_xyzw']=copy.deepcopy(cache['_rotBaseline'])
    position=np.asarray(cache['_posBaseline'],np.float32)
    if not unity_zero(np.asarray(cache['_positionBaseline'],np.float32)):
        if cache['_lenModNeedsPositionRestore'] or unity_zero(position):
            position=np.asarray(cache['_positionBaseline'],np.float32)
            cache['_lenModNeedsPositionRestore']=False
        else:
            position=position/np.float32(np.sqrt(unity_sqrmag(position)))*np.float32(cache['_lenBaseline'])
    local['local_position']=position.tolist()


def collect(local, cache):
    """Installed CollectBaseline preserves already established length history."""
    for key,field in (('_sclBaseline','local_scale'),('_posBaseline','local_position'),
                      ('_rotBaseline','local_rotation_xyzw')):
        cache[key]=copy.deepcopy(local[field])
    if unity_zero(np.asarray(cache['_positionBaseline'],np.float32)):
        cache['_positionBaseline']=copy.deepcopy(local['local_position'])
        cache['_lenBaseline']=float(np.float32(np.sqrt(unity_sqrmag(np.asarray(local['local_position'],np.float32)))))
        cache['_lenModNeedsPositionRestore']=False
    cache['_hasBaseline']=True


class CapturedHeadForward:
    def __init__(self, geometry, context):
        self.identity=context.get('identity',context)
        game=Path(geometry['game']['game_root']) if 'game_root' in geometry['game'] else Path('E:/HoneySelect2_ArcticFox')
        sources={'game_sha256':game/'HoneySelect2_Data/Managed/Assembly-CSharp.dll',
                 'slider_unlocker_sha256':game/'BepInEx/Plugins/HS2_BepisPlugins/HS2_SliderUnlocker.dll',
                 'abmx_sha256':game/'BepInEx/Plugins/HS2ABMX.dll',
                 'HS2_AdditionalAccessoryControls_sha256':game/'BepInEx/Plugins/HS2_Plugins/OrangeSpork/HS2_AdditionalAccessoryControls.dll',
                 'PushUpAI_sha256':game/'BepInEx/Plugins/HS2_Plugins/Mikke/PushUpHS2_2.1.4.dll',
                 'HS2_AdvIKPlugin_sha256':game/'BepInEx/Plugins/HS2_Plugins/OrangeSpork/HS2_AdvIKPlugin.dll'}
        self.source_provenance=[]
        for key,path in sources.items():
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=self.identity['reviewed_sources'][key]:
                raise ValueError('Installed source hash changed: '+key)
            self.source_provenance.append({'path':str(path),'sha256':digest})
        if any(s['current_weight']!=0 and any('delta_vertices' not in frame for frame in s['frames'])
               for mesh in geometry['meshes'] for s in mesh.get('blendshapes',[])):
            from src.hs2_assets import ChaList
            from src.hs2_blendshape_assets import recover_frames
            lists=ChaList()
            selected=lists.resolve('fo_head',geometry['character']['head_id'])
            geometry,self.asset_provenance=recover_frames(geometry,lists.bundle_path(selected),selected['MainData'])
        else:
            self.asset_provenance={'recovered_meshes':[],'scope':'all active frame deltas already recorded'}
        self.geometry=geometry
        self.transforms={r['id']:r for r in geometry['transforms']}
        self.rig=HeadRig(geometry['character']['head_id'],sampling_profile='slider_unlocker_18_2')
        self.names={}
        for row in self.transforms.values():
            self.names.setdefault(row['name'],[]).append(row['id'])
        self.modifiers=[]
        for row in self.identity['abmx_ordered_modifiers']:
            cache={key:copy.deepcopy(row[PREFIX+key]) for key in CACHE_FIELDS}
            self.modifiers.append({'id':row['bone_instance_id'],
                'name':row[PREFIX+'<BoneName>k__BackingField'], 'cache':cache,
                'coordinate_modifiers':[_data(d) if d else None for d in row[PREFIX+'<CoordinateModifiers>k__BackingField']]})
        self.anchor=self.unique('cf_J_Head')
        self.anchor_inverse=np.asarray(self.transforms[self.anchor]['world_to_local']).reshape(4,4)
        self.excluded=geometry['abmx_runtime']['rotation_excluded_bones']
        # Preserve renderer identities: duplicated o_tang names must not collide.
        self.meshes=geometry['meshes']
        self.needed=set()
        for mesh in self.meshes:
            for bone in mesh['bone_transform_ids']:
                while bone is not None and bone not in self.needed:
                    self.needed.add(bone)
                    bone=self.transforms[bone]['parent_id']
        self.additional,self.effect_decisions=self._head_effects()
        self._body_tables=None

    def unique(self,name):
        ids=self.names.get(name,[])
        if len(ids)!=1:
            raise ValueError('One captured transform required: '+name)
        return ids[0]

    def _head_effects(self):
        """Dependency analysis avoids dropping an effect just because it is body-related.

        Effects disjoint from *all* surface palettes and ancestors cannot move
        these surfaces. Active head effects are source-derived below; unknown
        dependencies/active breathing remain explicit unsupported branches.
        """
        additional={}
        decisions=[]
        needed_names={self.transforms[i]['name'] for i in self.needed}
        for effect in self.identity['effect_state']:
            typ=effect['type']; state=effect['state']; values={}
            if typ=='AdditionalAccessoryControls.AdditionalAccessoryBoneEffect':
                used=state[typ+'.UsedBones']
                hidden=state[typ+'.<HiddenBones>k__BackingField']
                if ((state[typ+'.<ResetLeftEar>k__BackingField'] and 'cf_J_EarBase_s_L' not in hidden) or
                    (state[typ+'.<ResetRightEar>k__BackingField'] and 'cf_J_EarBase_s_R' not in hidden)):
                    raise ValueError('Accessory producer requests full refresh beyond frozen protocol')
                for name in used:
                    if name in hidden:
                        values[name]=_data(state[typ+('.earHiddenBoneModifier' if name in ('cf_J_EarBase_s_L','cf_J_EarBase_s_R') else '.hiddenBoneModifier')])
                affected=set(used)
            elif typ=='AdvIKPlugin.Algos.ResizeBoneEffect':
                affected={'cf_N_height'}
                values['cf_N_height']=_data(state[typ+'.resizeModifier'])
            elif typ=='AdvIKPlugin.Algos.BreathingBoneEffect':
                affected=set(state[typ+'.affectedBones'])
                if self.identity['effect_globals']['MakerBreathing'] and affected & needed_names:
                    raise ValueError('Active breathing source-time/FrameEffects must be ported; never disabled as a fallback')
            elif typ.startswith('PushUpAI.'):
                # All lists are captured from the actual producer, not handpicked head exclusions.
                affected=set(state.get('PushUpAI.PushUpEffect.Bones',[]))
                for key in ('rootBones','bones2','bonesSize','nippleBones'):
                    affected.update(state.get(typ+'.'+key,[]))
                if affected & needed_names:
                    raise ValueError('PushUp effect overlaps a requested surface; full producer computation required: '+typ)
            else:
                raise ValueError('Unreviewed BoneEffect: '+typ)
            decisions.append({'type':typ,'affected_surface_ancestors':sorted(affected & needed_names),
                              'disjoint_from_surface_dependency':not bool(affected & needed_names)})
            for name,data in values.items():
                # Native BuildEffects resolves the FIRST modifier object for this name.
                if data!=IDENTITY:
                    additional.setdefault(name,[]).append(data)
        return additional,decisions

    def _face_update(self,local,native):
        values=np.asarray(native,float)
        if values.shape!=(59,) or not np.isfinite(values).all():
            raise ValueError('All 59 finite native values required')
        if self.identity['mouth_shape_mask']:
            values=values.copy()
            values[[3,6,11,47,48,49,50,51,52,53]]=.5
        world=_fk_world(self.rig,values,None)
        field_axes={}
        for e in self.rig.eqns:
            name=self.rig.enums['dst'][e['dst']]
            field_axes.setdefault(name,[]).append(e)
        for name,equations in field_axes.items():
            if not any(i in self.needed for i in self.names.get(name,[])):
                continue
            pid=self.rig.name2pid.get(name)
            if pid is None:
                raise ValueError('Native equation missing cached bone '+name)
            parent=self.rig.bones[pid]['parent']
            m=np.linalg.inv(world[parent]) @ world[pid] if parent in world else world[pid]
            scale=np.linalg.norm(m[:3,:3],axis=0)
            quat=Rotation.from_matrix(m[:3,:3]/scale).as_quat()
            live=local[self.unique(name)]
            for equation in equations:
                if equation['target']=='pos':
                    live['local_position'][equation['axis']]=float(np.float32(m[equation['axis'],3]))
                elif equation['target']=='rot':
                    live['local_rotation_xyzw']=quat.astype(np.float32).tolist()
                elif equation['target']=='scl':
                    live['local_scale']=scale.astype(np.float32).tolist()

    def _body_update(self,local):
        # Installed ShapeBodyInfoFemale.DstName intersect ALL captured surface
        # palettes+ancestors is exactly these two. The remaining body writers
        # (including bust-mask updates) are disjoint from this requested head.
        if self._body_tables is None:
            import UnityPy
            from scripts.hs2_extract_head import CUSTOMSHAPE, parse_anm_shape
            rows=[]
            for reader in UnityPy.load(CUSTOMSHAPE).objects:
                if reader.type.name!='TextAsset':
                    continue
                asset=reader.read_typetree()
                if asset['m_Name']!='cf_custombody':
                    continue
                text=asset['m_Script']
                if not isinstance(text,str):
                    text=bytes(text).decode('utf-8')
                for line in text.splitlines():
                    if line.strip():
                        fields=line.split('\t')
                        rows.append((int(fields[0]),fields[1],[v!='0' for v in fields[2:11]]))
            self._body_tables=(rows,parse_anm_shape(CUSTOMSHAPE,'cf_anmShapeBody'))
        rows,keys=self._body_tables
        wanted={'cf_s_Head_h','cf_s_Head_s','cf_s_height'}
        scales={name:np.ones(3,np.float32) for name in wanted}
        for category,name,use in rows:
            if name not in wanted:
                continue
            _,_,s=sample_keyframes(keys[name],float(self.identity['body_values'][category]),
                                  profile='slider_unlocker_18_2',bone_name=name,sample_rotation=False)
            for axis in range(3):
                if use[6+axis]:
                    scales[name][axis]=s[axis]
        local[self.unique('cf_N_height')]['local_scale']=scales['cf_s_height'].tolist()
        local[self.unique('cf_J_Head_s')]['local_scale']=(scales['cf_s_Head_h']*scales['cf_s_Head_s']).tolist()

    def evaluate(self,patches=(),*,native=None,apply_count=1):
        if type(apply_count)!=int or not 1<=apply_count<=32:
            raise ValueError('Declared Apply count 1..32 required')
        local={i:{k:copy.deepcopy(r[k]) for k in ('local_position','local_rotation_xyzw','local_scale')}
               for i,r in self.transforms.items()}
        modifiers=copy.deepcopy(self.modifiers)
        allowed=set(IDENTITY)|{'name','remove'}
        if any(not isinstance(p,dict) or set(p)-allowed or 'name' not in p for p in patches):
            raise ValueError('Explicit supported patch fields required')
        if len({p['name'] for p in patches})!=len(patches):
            raise ValueError('Duplicate patch bone names')
        if native is not None:
            self._face_update(local,native)
        removed=False
        for patch in patches:
            match=next((m for m in modifiers if m['name']==patch['name']),None)
            if patch.get('remove'):
                if match is not None:
                    reset(local[match['id']],match['cache'])
                    modifiers.remove(match); removed=True
                continue
            if match is None:
                ident=self.unique(patch['name'])
                # Native constructor defaults, followed by CollectBaseline on
                # the current copied transform; append preserves list order.
                cache={k:False for k in FLAG_FIELDS}
                cache.update({'_sclBaseline':[1.,1.,1.],'_posBaseline':[0.,0.,0.],
                              '_positionBaseline':[0.,0.,0.],'_rotBaseline':[0.,0.,0.,0.],
                              '_lenBaseline':0.})
                collect(local[ident],cache)
                match={'id':ident,'name':patch['name'],'cache':cache,
                       'coordinate_modifiers':[copy.deepcopy(IDENTITY)]}
                modifiers.append(match)
            if match['id'] not in local:
                raise ValueError('Requested patch targets a transform outside captured surfaces')
            for field in IDENTITY:
                if field in patch:
                    match['coordinate_modifiers'][0][field]=copy.deepcopy(patch[field])
        if removed:
            self._body_update(local)
            self._face_update(local,native if native is not None else self.identity['native59'])
        if native is not None or self.identity['abmx_baseline_refresh_pending']:
            if native is None:
                native=self.identity['native59']
            dst=set(self.rig.enums['dst'])|{'cf_N_height','cf_J_Head_s'}
            refreshing=[m for m in modifiers if m['name'] in dst and m['id'] in local]
            for m in refreshing:
                reset(local[m['id']],m['cache'])
            self._face_update(local,native)
            self._body_update(local)
            for m in refreshing:
                collect(local[m['id']],m['cache'])
        branches=[]
        for step in range(apply_count):
            for name in self.additional:
                if name not in {m['name'] for m in modifiers}:
                    ident=self.unique(name)
                    cache={k:False for k in FLAG_FIELDS}
                    cache.update({'_sclBaseline':[1.,1.,1.],'_posBaseline':[0.,0.,0.],
                                  '_positionBaseline':[0.,0.,0.],'_rotBaseline':[0.,0.,0.,0.],
                                  '_lenBaseline':0.})
                    collect(local[ident],cache)
                    modifiers.append({'id':ident,'name':name,'cache':cache,
                                      'coordinate_modifiers':[copy.deepcopy(IDENTITY)]})
            first_by_name={}
            for index,m in enumerate(modifiers):
                first_by_name.setdefault(m['name'],index)
            for index,m in enumerate(modifiers):
                if m['id'] not in self.needed:
                    branches.append({'step':step,'modifier_index':index,'name':m['name'],
                                     'branches':['disjoint_from_all_requested_surface_bones_and_ancestors']})
                    continue
                if m['name'].startswith(('cf_J_SiriDam','cf_J_Mune00')):
                    reset(local[m['id']],m['cache']); collect(local[m['id']],m['cache'])
                # Disjoint modifier transformations are retained but cannot change skinning dependencies.
                result=replay_apply(before=local[m['id']],cache=m['cache'],
                    coordinate_modifiers=m['coordinate_modifiers'],coordinate=(0 if self.identity['abmx_coordinate']=='Unknown' else int(self.identity['abmx_coordinate'])),
                    additional_modifiers=(self.additional.get(m['name'],[]) if first_by_name[m['name']]==index else []),bone_exists=True,
                    rotation_excluded=m['name'] in self.excluded,
                    is_during_h_scene=self.identity['abmx_during_h_scene'])
                local[m['id']]=result['after']; m['cache']=result['cache_after']
                branches.append({'step':step,'modifier_index':index,'name':m['name'],'branches':result['branches']})
        world={}
        def fk(i):
            if i not in world:
                r=local[i]; matrix=_trs(r['local_position'],r['local_rotation_xyzw'],r['local_scale'])
                parent=self.transforms[i]['parent_id']
                world[i]=fk(parent) @ matrix if parent is not None else matrix
            return world[i]
        transforms={i:{'local_to_world':fk(i).reshape(-1).tolist()} for i in self.needed}
        surfaces=[]
        for mesh in self.meshes:
            verts,active=skin_world(mesh,transforms,influences=4)
            surfaces.append({'renderer_path':mesh['renderer_path'],'mesh_name':mesh['mesh_name'],
                'world_vertices':verts,'head_vertices':(np.c_[verts,np.ones(len(verts))] @ self.anchor_inverse.T)[:,:3],
                'triangles':np.asarray(mesh['source']['triangles']).reshape(-1,3),'active_blendshapes':active})
        return {'surfaces':surfaces,'branches':branches,'effect_decisions':self.effect_decisions,
                'modifiers':serialized(modifiers),'bone_world':world}
