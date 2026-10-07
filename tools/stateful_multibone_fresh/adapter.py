"""Earlier clean identity history -> predeclared changed native59 -> four Apply sequences.

Observed per-bone counts are declared nuisance inputs for this diagnostic, not
a controlled common-N protocol. Candidate before/after TRS are targets only.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from src.hs2_abmx_torch import FLAGS, apply_sequence, tensor_cache
from src.hs2_deform_torch import TorchHeadRig
from tools.abmx_multibone.geometry import DEFAULT_BONES, select_cursor_predictions
from tools.abmx_multibone.run import restored_flags, verify_protocol
from tools.abmx_replay.validate_trace import trs_errors
from tools.stateful_multibone.adapter import IDENTITY, modifier_vector, read, sha


def persistent_equal(fields, reference):
    return all(np.array_equal(np.asarray(fields[key], dtype=np.float32), np.asarray(reference[key], dtype=np.float32))
               for key in ('_lenBaseline', '_positionBaseline'))


def clean(fields, name):
    if fields['_hasBaseline'] is not True or any(fields[key] for key in FLAGS if key != '_hasBaseline'):
        raise ValueError('Actual clean flags required for '+name)


def local_target(state):
    return {key: copy.deepcopy(state[key]) for key in ('local_position', 'local_rotation_xyzw', 'local_scale')}


class EarlierHistoryProtocol:
    """Capture lifecycle is proved separately from predicting candidate locals."""

    def __init__(self, source_trace, source_snapshot, candidate_trace, candidate_snapshot, contract, declared):
        for path, digest in ((contract['assembly_path'], contract['assembly_sha256']),
                             (contract['unity_core_path'], contract['unity_core_sha256'])):
            if sha(path) != digest:
                raise ValueError('Installed source assembly changed')
        if any(sha(row['path']) != row['sha256'] for row in contract['sources']):
            raise ValueError('Installed source decompilation changed')
        _, source_binding, source_replay = select_cursor_predictions(source_snapshot, source_trace, contract)
        _, candidate_binding, candidate_replay = select_cursor_predictions(candidate_snapshot, candidate_trace, contract)
        if source_replay['inter_call_boundaries'] or candidate_replay['inter_call_boundaries']:
            raise ValueError('External local/cache boundary unsupported')
        if (source_trace['metadata']['stopped_frame'] >= candidate_trace['metadata']['started_frame'] or
                source_binding['cursor']['frame'] >= candidate_trace['metadata']['started_frame']):
            raise ValueError('History must be measured and trace stopped before candidate observation')
        for key in ('transform_id', 'head_id'):
            if source_snapshot['character'][key] != candidate_snapshot['character'][key]:
                raise ValueError('Candidate history belongs to another character/head')
        if source_trace['metadata']['session_id'] == candidate_trace['metadata']['session_id']:
            raise ValueError('History and candidate need distinct observation sessions')
        if len(declared['names']) != 4 or set(declared['names']) != set(DEFAULT_BONES):
            raise ValueError('Exactly four declared bone names required')
        if declared.get('sampling_profile') != 'slider_unlocker_18_2' or declared.get('normalized_whole_head_gate') != 1e-5:
            raise ValueError('Explicit installed sampling profile and fixed 1e-5 gate required')
        if declared.get('common_apply_count') is not None or declared.get('common_count_runtime_certified') is not False:
            raise ValueError('This fresh batch does not certify a controlled common Apply count')
        verify_protocol(declared, candidate_snapshot, candidate_binding)
        self.head_id = candidate_snapshot['character']['head_id']
        self.source_native = np.asarray(source_snapshot['character']['shape_value_face'], dtype=np.float64)
        # Prediction input is the predeclared vector, not native readback from candidate snapshot.
        self.candidate_native = np.asarray(declared['native59'], dtype=np.float64)
        if self.source_native.shape != (59,) or self.candidate_native.shape != (59,) or not np.isfinite(self.source_native).all() or not np.isfinite(self.candidate_native).all():
            raise ValueError('Complete finite source and predeclared candidate native59 required')
        if not np.array_equal(self.source_native.astype(np.float32), np.asarray(declared['source_history_expected_native59'], dtype=np.float32)):
            raise ValueError('Earlier source native59 differs from predeclared source history')
        for key in ('candidate_expected_native59', 'baseline_expected_native59'):
            if not np.array_equal(self.candidate_native, np.asarray(declared[key], dtype=np.float64)):
                raise ValueError('Predeclared candidate/baseline native59 inconsistent')
        if np.array_equal(self.source_native.astype(np.float32), self.candidate_native.astype(np.float32)):
            raise ValueError('Fresh varied-native diagnostic needs an actual changed native vector')
        self.fields, self.source_before, self.candidate_before, self.candidate_baseline = {}, {}, {}, {}
        self.patches = {row['name']: modifier_vector(row) for row in declared['patches']}
        self.counts, self.identities, starts = {}, {}, {}
        candidate_sequence = candidate_binding['cursor']['last_completed_sequence']
        for name in DEFAULT_BONES:
            source_events = [e for e in source_trace['events'] if e['bone_name'] == name]
            candidate_events = [e for e in candidate_trace['events'] if e['bone_name'] == name]
            first, fresh_first = source_events[0], candidate_events[0]
            fields = first['before']['cache']['fields']
            clean(fields, name)
            clean(fresh_first['before']['cache']['fields'], name)
            identity = (first['before']['bone_transform_id'], first['modifier_instance_identity'])
            self.identities[name] = {'bone_transform_id': identity[0], 'modifier_instance_identity': identity[1]}
            for stage, events in (('source', source_events), ('candidate', candidate_events)):
                for event in events:
                    if (event['before']['bone_transform_id'], event['modifier_instance_identity']) != identity:
                        raise ValueError('History/candidate bone or modifier identity changed for '+name)
                    if event['additional_modifiers'] or event['is_during_h_scene'] or event['no_rotation_excluded']:
                        raise ValueError('Unsupported scene/additional/exclusion branch for '+name)
                    for phase in ('before', 'after'):
                        if not persistent_equal(event[phase]['cache']['fields'], fields):
                            raise ValueError('Persistent history changed in '+stage+'/'+phase+'/'+name)
                    if stage == 'source':
                        if not np.array_equal(modifier_vector(event['resolved_modifier']), IDENTITY):
                            raise ValueError('Candidate-before history trace must be entirely identity')
                        clean(event['before']['cache']['fields'], name)
                        clean(event['after']['cache']['fields'], name)
            if not np.array_equal(modifier_vector(fresh_first['resolved_modifier']), IDENTITY):
                raise ValueError('Candidate trace requires initial identity call for '+name)
            # Persistent fields also cross-bind both geometry caches, not just trace rows.
            for snapshot in (source_snapshot, candidate_snapshot):
                entries = [row for row in snapshot['abmx_runtime']['bones'] if row['name'] == name]
                if len(entries) != 1 or not persistent_equal(entries[0]['runtime_baseline']['fields'], fields):
                    raise ValueError('Snapshot/history private fields differ for '+name)
            prefix = [e for e in candidate_events if e['sequence'] <= candidate_sequence]
            values = [modifier_vector(e['resolved_modifier']) for e in prefix]
            active = [i for i, value in enumerate(values) if not np.array_equal(value, IDENTITY)]
            if not active or any(not np.array_equal(value.astype(np.float32), self.patches[name].astype(np.float32)) for value in values[active[0]:]):
                raise ValueError('One predeclared constant candidate patch required for '+name)
            all_values = [modifier_vector(e['resolved_modifier']) for e in candidate_events]
            if any(not np.array_equal(value.astype(np.float32), self.patches[name].astype(np.float32)) for value in all_values[active[0]:]):
                raise ValueError('Candidate patch changed later in complete trace for '+name)
            self.counts[name] = len(prefix)-active[0]
            starts[name] = prefix[active[0]]['sequence']
            self.fields[name] = copy.deepcopy(fields)  # only earlier source is a prediction cache input
            self.source_before[name] = local_target(first['before'])
            self.candidate_before[name] = local_target(fresh_first['before'])
            fresh_fields = fresh_first['before']['cache']['fields']
            self.candidate_baseline[name] = {key: copy.deepcopy(fresh_fields[field]) for key, field in
                                            (('local_position', '_posBaseline'), ('local_rotation_xyzw', '_rotBaseline'), ('local_scale', '_sclBaseline'))}
        self.metadata = {'head_id': self.head_id, 'names': list(DEFAULT_BONES), 'source_cursor': source_binding,
                         'candidate_cursor': candidate_binding, 'source_total_calls': len(source_trace['events']),
                         'candidate_total_calls': len(candidate_trace['events']), 'identities': self.identities,
                         'source_native59': self.source_native.tolist(), 'predeclared_candidate_native59': self.candidate_native.tolist(),
                         'candidate_start_sequences': starts, 'observed_candidate_apply_counts': self.counts,
                         'persistent_history_source': 'earlier completed clean identity trace and geometry',
                         'persistent_history_float32_unchanged_in_all_calls_and_snapshots': True,
                         'persistent_history': {name: {key: fields[key] for key in ('_lenBaseline', '_positionBaseline')}
                                                for name, fields in self.fields.items()},
                         'candidate_native_and_baseline_trs_source': 'predeclared native59 through cached native predictor',
                         'actual_before_used_as_prediction_input': False, 'actual_after_used_as_prediction_input': False,
                         'application_counts_fitted': False, 'counts_controlled': False, 'common_N_runtime_certified': False,
                         'arbitrary_candidate_runtime_certified': False, 'sampling_profile': 'slider_unlocker_18_2'}

    @classmethod
    def from_manifest(cls, manifest_path, contract_path, head_id, window='late'):
        manifest, contract = read(manifest_path), read(contract_path)
        restored_flags(manifest)
        if sha(manifest['protocol_path']) != manifest['protocol_sha256']:
            raise ValueError('Predeclared candidate protocol SHA mismatch')
        cases = [case for case in manifest['cases'] if case['head_id'] == head_id]
        if len(cases) != 1:
            raise ValueError('Unique head case required')
        case = cases[0]
        if len(case['names']) != 4 or set(case['names']) != set(DEFAULT_BONES):
            raise ValueError('Complete four-bone case required')
        selected = [row for row in case['windows'] if row['window'] == window]
        if len(selected) != 1:
            raise ValueError('Unique candidate window required')
        source, target = case['source_history'], selected[0]
        paths = [(source['trace'], source['trace_sha256']), (source['geometry']['path'], source['geometry']['sha256']),
                 (case['trace'], case['trace_sha256']), (target['geometry']['path'], target['geometry']['sha256'])]
        if any(sha(path) != digest for path, digest in paths):
            raise ValueError('History/candidate source SHA mismatch')
        result = cls(read(source['trace']), read(source['geometry']['path']), read(case['trace']),
                     read(target['geometry']['path']), contract, read(manifest['protocol_path']))
        if result.head_id != head_id or target['head_id'] != head_id:
            raise ValueError('Head ID source mismatch')
        result.metadata.update(source_files={str(path): digest for path, digest in paths},
                               manifest_sha256=sha(manifest_path), contract_sha256=sha(contract_path),
                               predeclared_protocol_sha256=manifest['protocol_sha256'], window=window)
        return result


class FreshObservedCountTorchRig(TorchHeadRig):
    """Observed-count fresh diagnostic, never an unrestricted candidate optimizer."""

    def __init__(self, rig, protocol, **kwargs):
        if rig.head_id != protocol.head_id or rig.sampling_profile != 'slider_unlocker_18_2' or kwargs.get('dtype') != torch.float64:
            raise ValueError('Source head, explicit installed sampler and float64 native/FK/LBS required')
        super().__init__(rig, **kwargs)
        if self.sampling_profile != 'slider_unlocker_18_2' or any(name not in self.name2bone or name not in self.ab_names for name in DEFAULT_BONES):
            raise ValueError('Missing four-bone rig/profile coverage')
        self.protocol, self.slots = protocol, {name: self.ab_names.index(name) for name in DEFAULT_BONES}
        self.indices = {name: self.name2bone[name] for name in DEFAULT_BONES}
        checks = {}
        with torch.no_grad():
            for label, values, before in (('source', protocol.source_native, protocol.source_before),
                                          ('candidate', protocol.candidate_native, protocol.candidate_before)):
                native = torch.as_tensor(values[None], device=self.device, dtype=self.dtype)
                pos, quat, scl = super().local_transforms(native)
                for name, index in self.indices.items():
                    prediction = {key: value[0, index].cpu().tolist() for key, value in
                                  zip(('local_position', 'local_rotation_xyzw', 'local_scale'), (pos, quat, scl))}
                    baseline = ({key: protocol.fields[name][field] for key, field in
                                 (('local_position', '_posBaseline'), ('local_rotation_xyzw', '_rotBaseline'), ('local_scale', '_sclBaseline'))}
                                if label == 'source' else protocol.candidate_baseline[name])
                    check = {'before': trs_errors(prediction, before[name]), 'baseline': trs_errors(prediction, baseline)}
                    checks[label+'/'+name] = check
                    if not check['before']['passed'] or not check['baseline']['passed']:
                        raise ValueError('Native predictor fails '+label+' actual before/baseline for '+name)
        self.protocol_metadata = {**protocol.metadata, 'native_boundary_checks': checks,
                                  'transition_arithmetic': 'float32', 'native_fk_lbs_arithmetic': 'float64'}

    def ab_tensor(self, ab_data, batch=1):
        if not isinstance(ab_data, dict) or set(ab_data) != set(DEFAULT_BONES):
            raise ValueError('Exactly four explicit modifiers required')
        if type(batch) is not int or batch < 1:
            raise ValueError('Positive integer batch required')
        for value in ab_data.values():
            if not isinstance(value, dict) or set(value) != {'scale', 'length', 'position', 'rotation'}:
                raise ValueError('Complete explicit modifier fields required')
            modifier_vector(value)
        return super().ab_tensor(ab_data, batch=batch)

    def local_states(self, shape_face, ab):
        if shape_face is None or shape_face.ndim != 2 or shape_face.shape[1] != 59 or shape_face.dtype != torch.float64:
            raise ValueError('Explicit float64 batched native59 required')
        reference = torch.as_tensor(self.protocol.candidate_native, device=shape_face.device, dtype=shape_face.dtype)
        if not torch.equal(shape_face, reference[None].expand_as(shape_face)):
            raise ValueError('Observed-count diagnostic accepts only predeclared fresh candidate native59')
        if ab is None or ab.shape != (len(shape_face), len(self.ab_names), 10) or not torch.isfinite(ab).all():
            raise ValueError('Complete finite ABMX tensor required')
        other = [i for i in range(len(self.ab_names)) if i not in self.slots.values()]
        identity = torch.as_tensor(IDENTITY, device=ab.device, dtype=ab.dtype)
        if (ab[:, other] != identity).any():
            raise ValueError('Other bone static fallback forbidden')
        for name, slot in self.slots.items():
            expected = torch.as_tensor(self.protocol.patches[name], device=ab.device, dtype=ab.dtype)
            if not torch.equal(ab[:, slot], expected[None].expand_as(ab[:, slot])):
                raise ValueError('Only predeclared fresh patches accepted under observed counts')
        pos, quat, scl = super().local_transforms(shape_face)
        states, caches = {}, {}
        for name, index in self.indices.items():
            local = (pos[:, index].float(), quat[:, index].float(), scl[:, index].float())
            cache = tensor_cache(self.protocol.fields[name], device=self.device, batch=len(shape_face))
            # No candidate actual cache fields are substituted here.
            cache = {**cache, '_posBaseline': local[0], '_rotBaseline': local[1], '_sclBaseline': local[2]}
            states[name], caches[name] = apply_sequence(local, cache, ab[:, self.slots[name]].float(), self.protocol.counts[name])
            target = torch.tensor([index], device=self.device)
            pos, quat, scl = tuple(value.index_copy(1, target, override.to(value.dtype)[:, None])
                                  for value, override in zip((pos, quat, scl), states[name]))
        return (pos, quat, scl), states, caches

    def local_transforms(self, shape_face=None, ab=None):
        return self.local_states(shape_face, ab)[0]
