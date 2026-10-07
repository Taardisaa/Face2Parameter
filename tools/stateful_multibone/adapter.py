"""Native59 -> four independent installed ABMX caches -> parent-first FK/LBS.

Recorded validation uses each bone's actual prefix count. New candidates require
a declared common count; neither observations' after transforms nor fitted call
counts are candidate inputs. Persistent length history remains measured.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.hs2_abmx_torch import FLAGS, apply_sequence, tensor_cache
from src.hs2_deform_torch import TorchHeadRig
from tools.abmx_multibone.geometry import DEFAULT_BONES, select_cursor_predictions
from tools.abmx_multibone.run import restored_flags, verify_protocol
from tools.abmx_replay.validate_trace import trs_errors

IDENTITY = np.array([1., 1., 1., 1., 0., 0., 0., 0., 0., 0.])


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def modifier_vector(modifier):
    result = np.r_[modifier['scale'], modifier['length'], modifier['position'], modifier['rotation']]
    if result.shape != (10,) or not np.isfinite(result).all():
        raise ValueError('Complete finite resolved modifier required')
    return result


class MultiNativeBaselineProtocol:
    """An exact observed clean boundary for all four bones, never a cache reset."""

    def __init__(self, trace, contract, snapshot, *, trace_path, contract_path,
                 snapshot_path, common_apply_count=None):
        for path, expected in ((contract['assembly_path'], contract['assembly_sha256']),
                               (contract['unity_core_path'], contract['unity_core_sha256'])):
            if sha(path) != expected:
                raise ValueError('Installed source assembly changed')
        if any(sha(row['path']) != row['sha256'] for row in contract['sources']):
            raise ValueError('Installed decompiled source changed')
        # Revalidate complete trace and each exact cursor independently with NumPy.
        _, binding, replay = select_cursor_predictions(snapshot, trace, contract)
        if replay['inter_call_boundaries']:
            raise ValueError('Candidate sequence cannot cross any external local/cache boundary')
        sequence = binding['cursor']['last_completed_sequence']
        prefix = trace['events'][:sequence]
        self.head_id = snapshot['character']['head_id']
        self.reference_native = np.asarray(snapshot['character']['shape_value_face'], dtype=np.float64)
        if self.reference_native.shape != (59,) or not np.isfinite(self.reference_native).all():
            raise ValueError('Explicit actual native59 required')
        self.fields, self.reference_before, self.reference_modifiers = {}, {}, {}
        self.observed_counts, starts, identities = {}, {}, {}
        for name in DEFAULT_BONES:
            events = [e for e in prefix if e['bone_name'] == name]
            first = events[0]
            all_events = [e for e in trace['events'] if e['bone_name'] == name]
            if any(e['modifier_instance_identity'] != first['modifier_instance_identity'] or
                   e['before']['bone_transform_id'] != first['before']['bone_transform_id'] for e in all_events):
                raise ValueError('Bone/modifier identity changed in complete trace for '+name)
            fields = first['before']['cache']['fields']
            if fields['_hasBaseline'] is not True or any(fields[key] for key in FLAGS if key != '_hasBaseline'):
                raise ValueError('Explicit clean first-before flags required for '+name)
            if not np.array_equal(modifier_vector(first['resolved_modifier']), IDENTITY):
                raise ValueError('First observed call must be identity for '+name)
            if any(e['additional_modifiers'] or e['is_during_h_scene'] or e['no_rotation_excluded'] for e in all_events):
                raise ValueError('Unsupported additional/scene/exclusion protocol for '+name)
            values = [modifier_vector(e['resolved_modifier']) for e in events]
            active = [i for i, value in enumerate(values) if not np.array_equal(value, IDENTITY)]
            if not active or any(not np.array_equal(value, values[active[0]]) for value in values[active[0]:]):
                raise ValueError('Require one constant candidate patch after identity for '+name)
            start = active[0]
            self.fields[name] = copy.deepcopy(fields)
            self.reference_before[name] = {key: copy.deepcopy(first['before'][key]) for key in
                                           ('local_position', 'local_rotation_xyzw', 'local_scale')}
            self.reference_modifiers[name] = values[start].copy()
            self.observed_counts[name] = len(events)-start
            starts[name] = events[start]['sequence']
            identities[name] = {'bone_transform_id': first['before']['bone_transform_id'],
                                'modifier_instance_identity': first['modifier_instance_identity']}
        if common_apply_count is not None and (type(common_apply_count) is not int or not 1 <= common_apply_count <= 10000):
            raise ValueError('New candidates require an explicit integer common Apply count 1..10000')
        self.common_apply_count = common_apply_count
        self.counts = (self.observed_counts.copy() if common_apply_count is None
                       else {name: common_apply_count for name in DEFAULT_BONES})
        self.metadata = {
            'model': 'installed_abmx_4.4.6_four_clean_native_baselines_persistent_history',
            'head_id': self.head_id, 'names': list(DEFAULT_BONES),
            'trace_path': str(Path(trace_path).resolve()), 'trace_sha256': sha(trace_path),
            'contract_path': str(Path(contract_path).resolve()), 'contract_sha256': sha(contract_path),
            'geometry_path': str(Path(snapshot_path).resolve()), 'geometry_sha256': sha(snapshot_path),
            'cursor_sequence': sequence, 'cursor_binding': binding,
            'trace_total_calls_revalidated': len(trace['events']), 'external_boundaries': [],
            'candidate_start_sequences': starts, 'bone_identities': identities,
            'observed_candidate_apply_counts': self.observed_counts,
            'declared_common_apply_count': common_apply_count, 'effective_counts': self.counts,
            'mode': 'recorded_prefix_validation' if common_apply_count is None else 'declared_common_candidate',
            'application_counts_fitted': False, 'actual_after_used_as_prediction_input': False,
            'initial_flags': {name: {key: self.fields[name][key] for key in FLAGS} for name in DEFAULT_BONES},
            'persistent_history': {name: {key: self.fields[name][key] for key in ('_lenBaseline', '_positionBaseline')}
                                   for name in DEFAULT_BONES},
            'baseline_policy': 'native59 predicts local TRS and pos/rotation/scale baseline; measured length/direction fields and flags preserved',
            'sampling_profile': 'slider_unlocker_18_2', 'new_candidate_runtime_certified': False,
            'scope': 'Exact observed four-bone source/cache boundaries; novel native/ABMX inputs or new counts require fresh runtime validation',
        }

    @classmethod
    def from_manifest(cls, manifest_path, contract_path, head_id, window='late', *, common_apply_count=None):
        manifest, contract = read(manifest_path), read(contract_path)
        restored_flags(manifest)
        if sha(manifest['protocol_path']) != manifest['protocol_sha256']:
            raise ValueError('Predeclared protocol source changed')
        declared = read(manifest['protocol_path'])
        rows = [c for c in manifest['cases'] if c['head_id'] == head_id]
        if len(rows) != 1 or set(rows[0]['names']) != set(DEFAULT_BONES):
            raise ValueError('Unique complete four-bone head case required')
        case = rows[0]
        windows = [w for w in case['windows'] if w['window'] == window]
        if len(windows) != 1:
            raise ValueError('Unique recorded window required')
        entry = windows[0]
        if sha(case['trace']) != case['trace_sha256'] or sha(entry['geometry']['path']) != entry['geometry']['sha256']:
            raise ValueError('Producer trace/geometry SHA mismatch')
        snapshot = read(entry['geometry']['path'])
        if snapshot['character']['head_id'] != head_id or entry['head_id'] != head_id:
            raise ValueError('Head ID mismatch')
        result = cls(read(case['trace']), contract, snapshot, trace_path=case['trace'], contract_path=contract_path,
                     snapshot_path=entry['geometry']['path'], common_apply_count=common_apply_count)
        verify_protocol(declared, snapshot, result.metadata['cursor_binding'])
        result.metadata.update(manifest_sha256=sha(manifest_path), predeclared_protocol_sha256=sha(manifest['protocol_path']), window=window)
        return result


class MultiStatefulTorchHeadRig(TorchHeadRig):
    """Float64 native/FK/LBS and independently propagated float32 cache per bone."""

    def __init__(self, rig, protocol, **kwargs):
        if rig.sampling_profile != 'slider_unlocker_18_2':
            raise ValueError('Explicit installed sampling profile required')
        if kwargs.get('dtype', torch.float32) != torch.float64:
            raise ValueError('Validated native/FK/LBS protocol requires float64')
        super().__init__(rig, **kwargs)
        if self.sampling_profile != 'slider_unlocker_18_2' or rig.head_id != protocol.head_id:
            raise ValueError('Sampling profile/head ID does not belong to protocol')
        if any(name not in self.name2bone or name not in self.ab_names for name in DEFAULT_BONES):
            raise ValueError('Every protocol bone must be in cached rig and BONE_NAME_LIST')
        self.protocol = protocol
        self.slots = {name: self.ab_names.index(name) for name in DEFAULT_BONES}
        self.indices = {name: self.name2bone[name] for name in DEFAULT_BONES}
        checks = {}
        with torch.no_grad():
            native = torch.as_tensor(protocol.reference_native[None], device=self.device, dtype=self.dtype)
            pos, quat, scale = super().local_transforms(native)
            for name, index in self.indices.items():
                local = {'local_position': pos[0, index].cpu().tolist(), 'local_rotation_xyzw': quat[0, index].cpu().tolist(),
                         'local_scale': scale[0, index].cpu().tolist()}
                baseline = {key: protocol.fields[name][field] for key, field in
                            (('local_position', '_posBaseline'), ('local_rotation_xyzw', '_rotBaseline'), ('local_scale', '_sclBaseline'))}
                before_error, cache_error = trs_errors(local, protocol.reference_before[name]), trs_errors(local, baseline)
                if not before_error['passed'] or not cache_error['passed']:
                    raise ValueError('Cached native59 fails actual clean before/baseline for '+name)
                checks[name] = {'first_before_error': before_error, 'baseline_trs_error': cache_error}
        self.protocol_metadata = {**protocol.metadata, 'native_boundary_checks': checks,
                                  'transition_arithmetic': 'float32', 'native_fk_lbs_arithmetic': 'float64'}

    def local_states(self, shape_face, ab):
        if shape_face is None or shape_face.ndim != 2 or shape_face.shape[1] != 59:
            raise ValueError('Explicit batched native59 required')
        pos, quat, scale = super().local_transforms(shape_face)
        if ab is None or ab.shape != (len(shape_face), len(self.ab_names), 10) or not torch.isfinite(ab).all():
            raise ValueError('Explicit complete finite batched ABMX array required')
        other = [i for i in range(len(self.ab_names)) if i not in self.slots.values()]
        identity = torch.as_tensor(IDENTITY, device=ab.device, dtype=ab.dtype)
        if (ab[:, other] != identity).any():
            raise ValueError('Other bones require explicit state protocols; static fallback forbidden')
        # Recorded prefix counts certify only those recorded native+patch inputs.
        if self.protocol.common_apply_count is None:
            reference = torch.as_tensor(self.protocol.reference_native, device=shape_face.device, dtype=shape_face.dtype)
            if not torch.equal(shape_face, reference[None].expand_as(shape_face)):
                raise ValueError('New native candidate requires a declared common Apply count')
            for name, slot in self.slots.items():
                modifier = torch.as_tensor(self.protocol.reference_modifiers[name], device=ab.device, dtype=ab.dtype)
                if not torch.equal(ab[:, slot], modifier[None].expand_as(ab[:, slot])):
                    raise ValueError('New ABMX candidate requires a declared common Apply count')
        states, caches = {}, {}
        for name, index in self.indices.items():
            local = (pos[:, index].float(), quat[:, index].float(), scale[:, index].float())
            cache = tensor_cache(self.protocol.fields[name], device=self.device, batch=len(shape_face))
            cache = {**cache, '_posBaseline': local[0], '_rotBaseline': local[1], '_sclBaseline': local[2]}
            predicted, caches[name] = apply_sequence(local, cache, ab[:, self.slots[name]].float(), self.protocol.counts[name])
            states[name] = predicted
            indices = torch.tensor([index], device=self.device)
            pos, quat, scale = tuple(value.index_copy(1, indices, override.to(value.dtype)[:, None])
                                     for value, override in zip((pos, quat, scale), predicted))
        return (pos, quat, scale), states, caches

    def local_transforms(self, shape_face=None, ab=None):
        return self.local_states(shape_face, ab)[0]

    def ab_tensor(self, ab_data, batch=1):
        if not isinstance(ab_data, dict) or set(ab_data) != set(DEFAULT_BONES):
            raise ValueError('Exactly four explicit protocol bone modifiers required; unknown/missing names forbidden')
        if type(batch) is not int or batch < 1:
            raise ValueError('Positive integer batch required')
        for value in ab_data.values():
            if not isinstance(value, dict) or set(value) != {'scale', 'length', 'position', 'rotation'}:
                raise ValueError('Complete modifier fields required')
            modifier_vector(value)
        return super().ab_tensor(ab_data, batch=batch)
