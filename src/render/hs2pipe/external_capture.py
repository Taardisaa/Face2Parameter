"""Capture-sourced inputs of a generated frame program (provenance ``external:capture``).

A generated frame needs objects whose bytes the scene state does not carry
(meshes, material textures, samplers, the camera target allocation, the
constant-buffer residue of earlier frames).  They are taken from the
mechanical conversion of the paired capture (``capture_to_program.py``) and
always labelled ``external:capture`` with the reason.  Nothing here is
generated or adjusted: the objects are imported verbatim (same ids, blobs
hard-linked), and binding / buffer queries return the captured values.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from src.render.frame_program import sha256

TABLES = ('resources', 'views', 'shaders', 'input_layouts', 'blend_states', 'depth_states', 'raster_states',
          'samplers')
DRAW_OPS = ('draw', 'draw_indexed', 'draw_instanced', 'draw_indexed_instanced')
BIND_CONSTANT_BUFFER = 4


class CaptureProgram:
    """Index over a converted capture program directory."""

    def __init__(self, root):
        self.root = Path(root)
        self.program = json.loads((self.root / 'program.json').read_text(encoding='utf-8'))
        self.by_id = {}
        for t in TABLES:
            for e in self.program[t]:
                self.by_id[e['id']] = (t, e)
        self.commands = self.program['commands']
        self.max_id = max(self.by_id)
        self._cb_at = None

    # ---------------------------------------------------------------- lookup
    def obj(self, ident):
        return self.by_id[ident][1]

    def event(self, index):
        return (self.commands[index].get('provenance') or {}).get('event')

    def command_at_event(self, event):
        hits = [i for i, c in enumerate(self.commands) if (c.get('provenance') or {}).get('event') == event]
        if len(hits) != 1:
            raise KeyError('event %r: %d commands' % (event, len(hits)))
        return hits[0]

    def blob(self, h):
        data = (self.root / 'blobs' / h).read_bytes()
        if sha256(data) != h:
            raise ValueError('capture blob hash mismatch ' + h)
        return data

    def shader_sha(self, ident):
        return None if ident is None else self.obj(ident)['blob']

    # ---------------------------------------------------- constant buffers
    def _cb_index(self):
        """Contents of every constant buffer bound by each draw, after the
        commands before it (initial data, then map_write / update_subresource
        in command order)."""
        if self._cb_at is not None:
            return self._cb_at
        cur = {}
        for t, e in self.by_id.values():
            if t == 'resources' and e['kind'] == 'buffer' and e['desc']['BindFlags'] & BIND_CONSTANT_BUFFER:
                init = e.get('initial')
                cur[e['id']] = bytearray(self.blob(init[0]['blob'])) if init else bytearray(e['desc']['ByteWidth'])
        at = {}
        first_write = {}
        writes = {}
        for i, c in enumerate(self.commands):
            op = c['op']
            if op == 'map_write' and c['resource'] in cur:
                first_write.setdefault(c['resource'], (i, bytes(cur[c['resource']])))
                data = self.blob(c['blob'])
                if len(data) != c['end'] - c['start']:
                    raise ValueError('map_write blob length differs from its range')
                before = bytes(cur[c['resource']])
                cur[c['resource']][c['start']:c['end']] = data
                writes.setdefault(c['resource'], []).append((i, before))
            elif op == 'update_subresource' and c['resource'] in cur:
                if c['box'] is not None:
                    raise NotImplementedError('boxed UpdateSubresource on a constant buffer')
                first_write.setdefault(c['resource'], (i, bytes(cur[c['resource']])))
                writes.setdefault(c['resource'], []).append((i, bytes(cur[c['resource']])))
                cur[c['resource']][:] = self.blob(c['blob'])
            elif op in DRAW_OPS or op == 'dispatch':
                snap = {}
                for st, b in c['state'].items():
                    if not isinstance(b, dict) or 'constant_buffers' not in b:
                        continue
                    for cb in b['constant_buffers']:
                        if 'buffer' in cb and cb['buffer'] in cur:
                            snap[(st, cb['slot'])] = (cb['buffer'], bytes(cur[cb['buffer']]))
                at[i] = snap
        self._cb_at = at
        self._first_write = first_write
        self._writes = writes
        self._draw_indices = sorted(at)
        return at

    def upload_index(self, draw_index, buffer_id):
        """Index of the write of ``buffer_id`` that belongs to this draw (after the
        previous draw/dispatch), else ``draw_index``."""
        self._cb_index()
        import bisect
        k = bisect.bisect_left(self._draw_indices, draw_index)
        prev = self._draw_indices[k - 1] if k > 0 else -1
        hits = [i for i, _ in self._writes.get(buffer_id, []) if prev < i < draw_index]
        return hits[-1] if hits else draw_index

    def contents_before_upload(self, draw_index, buffer_id):
        """Bytes of the buffer before this draw's own upload."""
        self._cb_index()
        u = self.upload_index(draw_index, buffer_id)
        if u == draw_index:
            for bid, data in self._cb_at[draw_index].values():
                if bid == buffer_id:
                    return data
            raise KeyError('buffer %d not bound by draw %d' % (buffer_id, draw_index))
        return next(b for i, b in self._writes[buffer_id] if i == u)

    def writes_between(self, buffer_id, a, b):
        """Capture writes of ``buffer_id`` with command index in (a, b)."""
        self._cb_index()
        return [i for i, _ in self._writes.get(buffer_id, []) if a < i < b]

    def bound_at(self, command_index):
        """{(stage, slot): (buffer id, bytes)} of one draw/dispatch."""
        return self._cb_index().get(command_index)

    def bound_cbuffer(self, command_index, stage, slot):
        """(buffer id, bytes) bound at ``stage``/``slot`` by a draw."""
        return self._cb_index()[command_index].get((stage, slot))

    def frame_start_contents(self, buffer_id):
        """Bytes of a capture constant buffer before its first in-frame write
        (cross-frame residue of Unity's shadow copy)."""
        self._cb_index()
        if buffer_id in self._first_write:
            return self._first_write[buffer_id][1]
        e = self.obj(buffer_id)
        init = e.get('initial')
        return self.blob(init[0]['blob']) if init else bytes(e['desc']['ByteWidth'])

    # --------------------------------------------------------------- import
    def import_into(self, builder, *, reason):
        """Copy every object of the capture program into ``builder`` (ids kept,
        blobs hard-linked).  The builder must have been created with
        ``first_id > self.max_id``."""
        for t in TABLES:
            for e in self.program[t]:
                entry = copy.deepcopy(e)
                prov = dict(entry.get('provenance') or {})
                prov['external'] = 'external:capture'
                prov['reason'] = reason
                entry['provenance'] = prov
                if entry['id'] in builder._ids:
                    raise ValueError('id collision %d' % entry['id'])
                builder._ids.add(entry['id'])
                builder.program[t].append(entry)
                for h in _entry_blobs(t, entry):
                    _link_blob(builder, self.root, h)
        return self.max_id


def _entry_blobs(table, e):
    if table == 'resources':
        return [i['blob'] for i in (e.get('initial') or [])]
    if table == 'shaders':
        return [e['blob']]
    if table == 'input_layouts':
        return [e['signature_blob']]
    return []


def _link_blob(builder, src_root, h):
    store = builder.blobs
    if h in store.sizes:
        return
    dst = store.dir / h
    src = Path(src_root) / 'blobs' / h
    if not dst.exists():
        try:
            os.link(src, dst)
            store.linked += 1
        except OSError:
            # file system without hard links (exFAT data drive): verified byte copy
            data = src.read_bytes()
            if sha256(data) != h:
                raise ValueError('capture blob hash mismatch ' + h)
            dst.write_bytes(data)
    store.sizes[h] = dst.stat().st_size


def link_command_blobs(builder, capture, cmd):
    """Hard-link the blobs a spliced capture command references."""
    hs = []
    if 'blob' in cmd:
        hs.append(cmd['blob'])
    st = cmd.get('state') or {}
    for b in st.values():
        if isinstance(b, dict):
            for cb in b.get('constant_buffers', []):
                if 'blob' in cb:
                    hs.append(cb['blob'])
    for h in hs:
        _link_blob(builder, capture.root, h)
